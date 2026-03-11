"""
surrogate/train_surrogate.py
─────────────────────────────
CircuitCAT surrogate 모델의 전체 생애주기 관리:

  1. save_to_db()        ← run.py가 시뮬 결과마다 호출
  2. should_retrain()    ← 50개 쌓일 때마다 True
  3. train()             ← 증분 학습 (이전 모델 위에서 fine-tune)
  4. predict()           ← BO loop에서 ngspice 대신 호출
  5. get_bottleneck_hint()← LLM 프롬프트에 삽입할 GNN 분석 결과

run.py 연동 방법:
  from surrogate.train_surrogate import CircuitSurrogate
  surrogate = CircuitSurrogate(db_dir="data/circuit_tag")

  # 시뮬 후 저장 (변경 1)
  surrogate.save_to_db(netlist_text, sim_output, task_id)

  # BO loop (변경 2)
  pred, unc = surrogate.predict(candidate_netlist)
  if unc < surrogate.UNCERTAINTY_THRESHOLD:
      perf = pred   # ngspice skip
  else:
      perf = run_ngspice(candidate_netlist)
      surrogate.save_to_db(candidate_netlist, perf, task_id)

  # LLM 프롬프트 hint (변경 3)
  hint = surrogate.get_bottleneck_hint(netlist_text)
  prompt = base_prompt + hint
"""

import os
import sys
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import numpy as np

# ──────────────────────────────────────────
# 선택적 의존성
# ──────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# surrogate/ 디렉토리가 루트에 있을 때와 서브모듈일 때 모두 대응
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

try:
    from netlist_parser import parse_netlist, circuit_graph_to_pyg
    from circuit_tag_dataset import (
        BagOfWordsEncoder, extract_performance_from_sim_output,
        PERFORMANCE_KEYS,
    )
    from circuit_cat import CircuitCAT, CircuitCATTrainer
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("CircuitSurrogate")

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────

PERF_KEYS = ["gain_dB", "ugf_MHz", "phase_margin", "power_uW",
             "cutoff_MHz", "osc_freq_hz", "conversion_gain", "success"]

@dataclass
class SurrogateConfig:
    # 학습 트리거
    cold_start_size:     int   = 30     # 최소 이 수 이상 쌓여야 첫 학습
    retrain_interval:    int   = 20     # 이 수만큼 새 데이터 쌓이면 재학습
    # 모델
    d_hidden:            int   = 128
    dropout:             float = 0.1
    gate_reg_weight:     float = 0.01
    # 학습
    lr:                  float = 1e-3
    epochs_cold:         int   = 300    # cold start 학습
    epochs_finetune:     int   = 100    # 증분 fine-tune
    batch_size:          int   = 16
    # 예측 불확실도
    mc_samples:          int   = 30
    uncertainty_threshold: float = 0.15  # 이 미만이면 ngspice skip


# ──────────────────────────────────────────
# DB: (netlist, 성능) 쌍 저장/로드
# ──────────────────────────────────────────

class CircuitDB:
    """
    JSON Lines 형식으로 (netlist, performance) 쌍을 저장.
    파일 I/O만 사용 — SQLite / Redis 불필요.
    """

    def __init__(self, db_dir: str):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.db_dir / "circuits.jsonl"
        self._cache: list[dict] = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._cache = []
        if self.db_file.exists():
            with open(self.db_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._cache.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        self._loaded = True

    def save(self, netlist_text: str, sim_output_or_perf, task_id: int = 0):
        """
        sim_output_or_perf: ngspice stdout 문자열 또는 dict {"gain_dB": 60.0, ...}
        """
        # 성능 dict 정규화
        if isinstance(sim_output_or_perf, str):
            perf = extract_performance_from_sim_output(sim_output_or_perf) \
                   if HAS_MODULES else {}
        else:
            perf = dict(sim_output_or_perf)

        # 중복 방지
        netlist_hash = hashlib.md5(netlist_text.encode()).hexdigest()[:12]
        self._load()
        for rec in self._cache:
            if rec.get("hash") == netlist_hash:
                log.debug(f"Duplicate netlist skipped ({netlist_hash})")
                return

        record = {
            "hash":         netlist_hash,
            "task_id":      task_id,
            "timestamp":    time.time(),
            "netlist":      netlist_text,
            "performance":  perf,
        }
        with open(self.db_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._cache.append(record)
        log.info(f"DB에 저장 (총 {len(self._cache)}개) | task={task_id} | "
                 f"gain={perf.get('gain_dB', '?'):.1f}dB "
                 f"PM={perf.get('phase_margin', '?'):.1f}°"
                 if isinstance(perf.get('gain_dB'), float) else
                 f"DB에 저장 (총 {len(self._cache)}개)")

    def load_all(self) -> list[dict]:
        self._load()
        return list(self._cache)

    def __len__(self):
        self._load()
        return len(self._cache)

    def new_since(self, last_count: int) -> int:
        return len(self) - last_count


# ──────────────────────────────────────────
# Fallback: torch 없을 때 numpy surrogate
# ──────────────────────────────────────────

class _NumpySurrogate:
    """torch/pyg 없을 때 사용하는 경량 numpy MLP surrogate."""

    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 8, seed: int = 0):
        rng = np.random.RandomState(seed)
        s = lambda a: np.sqrt(2.0 / a)
        self.W1 = rng.randn(d_in, d_hidden).astype(np.float32) * s(d_in)
        self.b1 = np.zeros(d_hidden, np.float32)
        self.W2 = rng.randn(d_hidden, d_hidden).astype(np.float32) * s(d_hidden)
        self.b2 = np.zeros(d_hidden, np.float32)
        self.W3 = rng.randn(d_hidden, d_out).astype(np.float32) * s(d_hidden)
        self.b3 = np.zeros(d_out, np.float32)
        self.x_mean = self.x_std = None
        self.y_mean = self.y_std = None

    def _forward(self, x, drop=0.0, rng=None):
        relu = lambda z: np.maximum(0, z)
        def dropout(z):
            if drop > 0 and rng is not None:
                return z * (rng.rand(*z.shape) > drop).astype(np.float32) / (1 - drop)
            return z
        return dropout(relu(dropout(relu(x @ self.W1 + self.b1)) @ self.W2 + self.b2)) @ self.W3 + self.b3

    def fit(self, X, y, lr=1e-3, epochs=300, batch=32):
        self.x_mean, self.x_std = X.mean(0), X.std(0) + 1e-6
        self.y_mean, self.y_std = y.mean(0), y.std(0) + 1e-6
        Xn = (X - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std
        rng = np.random.RandomState(0)
        for ep in range(epochs):
            idx = rng.permutation(len(Xn))
            for s in range(0, len(Xn), batch):
                xb = Xn[idx[s:s+batch]]
                yb = yn[idx[s:s+batch]]
                h1 = np.maximum(0, xb @ self.W1 + self.b1)
                h2 = np.maximum(0, h1 @ self.W2 + self.b2)
                d  = h2 @ self.W3 + self.b3 - yb
                dW3= h2.T @ (2*d/d.size); db3 = (2*d/d.size).sum(0)
                dh2= (2*d/d.size) @ self.W3.T * (h2>0)
                dW2= h1.T @ dh2; db2 = dh2.sum(0)
                dh1= dh2 @ self.W2.T * (h1>0)
                dW1= xb.T @ dh1; db1 = dh1.sum(0)
                for p, g in [(self.W3,dW3),(self.b3,db3),(self.W2,dW2),
                             (self.b2,db2),(self.W1,dW1),(self.b1,db1)]:
                    p -= lr * np.clip(g, -1, 1)
        log.info(f"  NumpySurrogate 학습 완료 (epochs={epochs})")

    def predict(self, x):
        xn = (x - self.x_mean) / self.x_std
        rng = np.random.RandomState(42)
        preds = np.stack([self._forward(xn, drop=0.2, rng=rng) for _ in range(30)])
        mean = preds.mean(0) * self.y_std + self.y_mean
        std  = preds.std(0)  * self.y_std
        return mean, std


# ──────────────────────────────────────────
# Feature extractor (TAG 없을 때 hand-crafted)
# ──────────────────────────────────────────

import re as _re

def _extract_structural_features(netlist_text: str) -> np.ndarray:
    """
    netlist 문자열 → numpy feature vector.
    CircuitCAT (GNN) 없을 때 사용.
    실제 CircuitCAT이 이것보다 훨씬 잘 함.
    """
    lines = [l.strip() for l in netlist_text.splitlines()
             if l.strip() and not l.startswith("*")]

    n_mos  = sum(1 for l in lines if l.upper().startswith("M"))
    n_res  = sum(1 for l in lines if l.upper().startswith("R"))
    n_cap  = sum(1 for l in lines if l.upper().startswith("C"))
    n_src  = sum(1 for l in lines if l.upper()[0:1] in ("V","I"))

    wl = [float(m.group(1))/float(lm.group(1))
          for l in lines if l.upper().startswith("M")
          for m  in [_re.search(r"W=([\d.]+)u", l, _re.I)] if m
          for lm in [_re.search(r"L=([\d.]+)u", l, _re.I)] if lm]
    max_wl  = max(wl)  if wl else 0.0
    mean_wl = float(np.mean(wl)) if wl else 0.0

    caps = [float(m.group(1)) for l in lines if l.upper().startswith("C")
            for m in [_re.search(r"([\d.]+)p", l)] if m]
    total_cap = sum(caps)

    res  = [float(m.group(1)) for l in lines if l.upper().startswith("R")
            for m in [_re.search(r"(\d+\.?\d*)$", l.split()[-1])] if m]
    total_res = sum(res) / 1e3

    return np.array([
        n_mos/10., n_res/5., n_cap/5., n_src/4.,
        max_wl/500., mean_wl/300.,
        total_cap/20., total_res/100.,
        float(n_mos > 4),   # 다단 구조 여부
        float(n_cap > 1),   # 보상 커패시터 존재
    ], dtype=np.float32)


# ──────────────────────────────────────────
# 메인: CircuitSurrogate
# ──────────────────────────────────────────

class CircuitSurrogate:
    """
    AnalogCoderPro의 run.py가 사용하는 단일 인터페이스.

    사용법:
        surrogate = CircuitSurrogate("data/circuit_tag")

        # 시뮬 후 저장
        surrogate.save_to_db(netlist, sim_output, task_id=19)

        # 자동 재학습
        if surrogate.should_retrain():
            surrogate.train()

        # BO loop에서 예측
        pred, unc = surrogate.predict(netlist)

        # LLM 프롬프트용 힌트
        hint = surrogate.get_bottleneck_hint(netlist)
    """

    UNCERTAINTY_THRESHOLD = 0.15  # 이 미만이면 ngspice skip 가능

    def __init__(
        self,
        db_dir: str = "data/circuit_tag",
        model_path: str = "data/surrogate_model.pt",
        config: Optional[SurrogateConfig] = None,
    ):
        self.cfg = config or SurrogateConfig()
        self.db = CircuitDB(db_dir)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self._model = None          # CircuitCAT or NumpySurrogate
        self._encoder = None        # BagOfWordsEncoder
        self._trained_on = 0        # 마지막 학습 시점의 DB 크기
        self._use_torch = HAS_TORCH and HAS_MODULES

        # 저장된 모델 있으면 로드
        self._try_load_model()

        mode = "CircuitCAT (PyG)" if self._use_torch else "NumpyMLP (fallback)"
        log.info(f"CircuitSurrogate 초기화 | 모드: {mode} | DB: {db_dir}")

    # ── 1. 데이터 저장 ───────────────────────

    def save_to_db(
        self,
        netlist_text: str,
        sim_output_or_perf,
        task_id: int = 0,
    ):
        """
        run.py에서 ngspice 시뮬 후 호출.
        자동으로 재학습 여부도 체크.
        """
        self.db.save(netlist_text, sim_output_or_perf, task_id)
        if self.should_retrain():
            log.info(f"재학습 트리거 (DB={len(self.db)}개)")
            self.train()

    # ── 2. 재학습 트리거 ─────────────────────

    def should_retrain(self) -> bool:
        n = len(self.db)
        if n < self.cfg.cold_start_size:
            return False
        if self._model is None:
            return True
        return (n - self._trained_on) >= self.cfg.retrain_interval

    # ── 3. 학습 ──────────────────────────────

    def train(self, force: bool = False):
        records = self.db.load_all()
        n = len(records)
        if n < self.cfg.cold_start_size and not force:
            log.warning(f"데이터 부족 ({n} < {self.cfg.cold_start_size}), 학습 건너뜀")
            return

        is_cold = (self._model is None)
        epochs  = self.cfg.epochs_cold if is_cold else self.cfg.epochs_finetune
        log.info(f"{'Cold start' if is_cold else 'Fine-tune'} 학습 시작 "
                 f"(n={n}, epochs={epochs})")

        if self._use_torch:
            self._train_circuitcat(records, epochs)
        else:
            self._train_numpy(records, epochs)

        self._trained_on = n
        self._save_model()
        log.info(f"학습 완료 | 모델 저장: {self.model_path}")

    def _train_circuitcat(self, records: list[dict], epochs: int):
        if self._encoder is None:
            self._encoder = BagOfWordsEncoder()

        data_list = []
        for rec in records:
            try:
                cg = parse_netlist(rec["netlist"])
                if not cg.components:
                    continue
                cg.performance = rec["performance"]
                data = circuit_graph_to_pyg(
                    cg,
                    text_encoder=self._encoder,
                    performance_keys=PERF_KEYS,
                )
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
                data_list.append(data)
            except Exception as e:
                log.debug(f"  데이터 변환 실패: {e}")

        if not data_list:
            log.warning("변환 가능한 데이터 없음")
            return

        # 모델 초기화 또는 재사용 (증분 학습)
        if self._model is None:
            self._model = CircuitCAT(
                d_text_in=self._encoder.dim,
                d_hidden=self.cfg.d_hidden,
                dropout=self.cfg.dropout,
                gate_reg_weight=self.cfg.gate_reg_weight,
            )
        trainer = CircuitCATTrainer(
            self._model, lr=self.cfg.lr
        )

        loader = DataLoader(data_list, batch_size=self.cfg.batch_size, shuffle=True)

        for ep in range(epochs):
            ep_losses = []
            ep_alphas = []
            for batch in loader:
                perf_targets = batch.y if hasattr(batch, 'y') and batch.y is not None \
                               else None
                losses = trainer.train_step(
                    batch,
                    node_labels=batch.role,
                    perf_targets=perf_targets,
                )
                ep_losses.append(losses["total"])
                ep_alphas.append(losses["alpha_mean"])

            if (ep + 1) % 50 == 0:
                mean_loss  = np.mean(ep_losses)
                mean_alpha = np.mean(ep_alphas)
                log.info(f"  ep {ep+1:4d}/{epochs} | "
                         f"loss={mean_loss:.4f} | α_mean={mean_alpha:.3f}")
                # Collapse 경고
                if abs(mean_alpha - 0.5) > 0.4:
                    log.warning(f"  ⚠️  Gate collapse 감지 α={mean_alpha:.3f} "
                                f"→ 학습률 낮추기 권장")

    def _train_numpy(self, records: list[dict], epochs: int):
        """torch 없을 때 fallback."""
        X = np.stack([_extract_structural_features(r["netlist"]) for r in records])
        y = np.stack([
            np.array([r["performance"].get(k, 0.0) for k in PERF_KEYS], dtype=np.float32)
            for r in records
        ])
        d_in = X.shape[1]
        if self._model is None:
            self._model = _NumpySurrogate(d_in=d_in, d_hidden=64, d_out=len(PERF_KEYS))
        self._model.fit(X, y, lr=self.cfg.lr, epochs=epochs)

    # ── 4. 예측 ──────────────────────────────

    def predict(self, netlist_text: str) -> tuple[dict, float]:
        """
        Returns:
            perf_pred   : {"gain_dB": 62.3, "ugf_MHz": 14.1, ...}
            uncertainty : float (높을수록 ngspice 검증 필요)
        """
        if self._model is None:
            log.warning("모델 미학습 — 더미 예측 반환")
            return {k: 0.0 for k in PERF_KEYS}, 1.0

        if self._use_torch:
            return self._predict_circuitcat(netlist_text)
        else:
            return self._predict_numpy(netlist_text)

    def _predict_circuitcat(self, netlist_text: str) -> tuple[dict, float]:
        cg   = parse_netlist(netlist_text)
        data = circuit_graph_to_pyg(cg, text_encoder=self._encoder)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        self._model.eval()
        preds = []
        with torch.no_grad():
            for _ in range(self.cfg.mc_samples):
                # MC Dropout: eval 모드에서도 dropout 활성화
                for m in self._model.modules():
                    if isinstance(m, nn.Dropout):
                        m.train()
                _, perf_pred, _, alpha, _ = self._model(data)
                preds.append(perf_pred.squeeze(0).numpy())

        preds  = np.stack(preds)          # [S, num_metrics]
        mean   = preds.mean(0)
        std    = preds.std(0)
        unc    = float(std.mean())

        perf_dict = {k: float(mean[i]) for i, k in enumerate(PERF_KEYS)}
        return perf_dict, unc

    def _predict_numpy(self, netlist_text: str) -> tuple[dict, float]:
        feat  = _extract_structural_features(netlist_text).reshape(1, -1)
        mean, std = self._model.predict(feat)
        unc   = float(std.mean())
        perf  = {k: float(mean[0, i]) for i, k in enumerate(PERF_KEYS)}
        return perf, unc

    # ── 5. LLM 프롬프트 힌트 생성 ────────────

    def get_bottleneck_hint(self, netlist_text: str) -> str:
        """
        CircuitCAT의 α (gate) 값으로
        어떤 소자가 성능에 가장 큰 영향을 미치는지 분석.

        반환값을 AnalogCoderPro의 optimize_template에 삽입.

        예시 출력:
          [GNN Analysis]
          High-impact components (α > 0.7 → text features dominant):
            - M6: likely bottleneck for gain/bandwidth (α=0.87)
            - CC: compensation cap, critical for phase margin (α=0.81)
          Suggestion: increase W6 or reduce CC to improve UGF.
        """
        if self._model is None or not self._use_torch:
            return ""

        try:
            cg   = parse_netlist(netlist_text)
            data = circuit_graph_to_pyg(cg, text_encoder=self._encoder)
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

            self._model.eval()
            with torch.no_grad():
                _, _, _, alpha, _ = self._model(data)

            alphas = alpha.squeeze(1).numpy()   # [N]
            names  = data.node_names            # [N]

            # α > 0.7: 텍스트(소자 특성)이 지배적 → 해당 소자가 bottleneck
            HIGH_ALPHA = 0.7
            bottlenecks = sorted(
                [(names[i], float(alphas[i])) for i in range(len(names))
                 if alphas[i] > HIGH_ALPHA],
                key=lambda x: -x[1]
            )

            if not bottlenecks:
                return "\n[GNN Analysis]\nNo strong bottleneck detected. Topology well-balanced.\n"

            lines = ["\n[GNN Analysis — CircuitCAT]",
                     "High-impact components (text feature dominant, α > 0.7):"]
            for name, a in bottlenecks[:5]:
                # 텍스트 설명도 포함
                comp_idx = names.index(name)
                text_desc = data.x_text[comp_idx][:60] if data.x_text else ""
                lines.append(f"  - {name} (α={a:.2f}): {text_desc}")

            lines.append("Suggestion: focus optimization on the components above.")
            lines.append("")
            return "\n".join(lines)

        except Exception as e:
            log.debug(f"get_bottleneck_hint 실패: {e}")
            return ""

    # ── 6. 상태 리포트 ───────────────────────

    def status(self) -> dict:
        return {
            "db_size":       len(self.db),
            "trained_on":    self._trained_on,
            "model_type":    "CircuitCAT" if self._use_torch else "NumpyMLP",
            "model_exists":  self._model is not None,
            "model_path":    str(self.model_path),
            "should_retrain": self.should_retrain(),
            "perf_keys":     PERF_KEYS,
        }

    def print_status(self):
        s = self.status()
        print("\n" + "="*50)
        print(" CircuitSurrogate 상태")
        print("="*50)
        for k, v in s.items():
            print(f"  {k:20s}: {v}")
        print("="*50 + "\n")

    # ── 내부: 모델 저장/로드 ─────────────────

    def _save_model(self):
        if self._model is None:
            return
        if self._use_torch:
            torch.save({
                "model_state":   self._model.state_dict(),
                "model_config":  asdict(self.cfg),
                "encoder_vocab": self._encoder.VOCAB if self._encoder else [],
                "trained_on":    self._trained_on,
            }, self.model_path)
        else:
            # numpy: pickle 없이 npz
            m = self._model
            np.savez(
                str(self.model_path).replace(".pt", ".npz"),
                W1=m.W1, b1=m.b1, W2=m.W2, b2=m.b2, W3=m.W3, b3=m.b3,
                x_mean=m.x_mean if m.x_mean is not None else np.array([0.0]),
                x_std= m.x_std  if m.x_std  is not None else np.array([1.0]),
                y_mean=m.y_mean if m.y_mean is not None else np.array([0.0]),
                y_std= m.y_std  if m.y_std  is not None else np.array([1.0]),
            )

    def _try_load_model(self):
        if self._use_torch and self.model_path.exists():
            try:
                ckpt = torch.load(self.model_path, map_location="cpu")
                cfg  = SurrogateConfig(**ckpt.get("model_config", {}))
                self._encoder = BagOfWordsEncoder()
                self._model   = CircuitCAT(
                    d_text_in=len(self._encoder.VOCAB),
                    d_hidden=cfg.d_hidden,
                    dropout=cfg.dropout,
                )
                self._model.load_state_dict(ckpt["model_state"])
                self._trained_on = ckpt.get("trained_on", 0)
                log.info(f"저장된 CircuitCAT 모델 로드 (trained_on={self._trained_on})")
            except Exception as e:
                log.warning(f"모델 로드 실패 ({e}) → 새로 학습 필요")

        npz_path = Path(str(self.model_path).replace(".pt", ".npz"))
        if not self._use_torch and npz_path.exists():
            try:
                d = np.load(str(npz_path))
                m = _NumpySurrogate(d_in=d["W1"].shape[0])
                for attr in ("W1","b1","W2","b2","W3","b3",
                             "x_mean","x_std","y_mean","y_std"):
                    setattr(m, attr, d[attr])
                self._model = m
                log.info("저장된 NumpyMLP 모델 로드")
            except Exception as e:
                log.warning(f"NumpyMLP 로드 실패: {e}")


# ──────────────────────────────────────────
# run.py 패치 스니펫 출력
# ──────────────────────────────────────────

RUN_PY_PATCH = '''
# ─── CircuitSurrogate 연동 (run.py에 추가) ───────────────────────────────
from surrogate.train_surrogate import CircuitSurrogate

_surrogate = CircuitSurrogate(
    db_dir="data/circuit_tag",
    model_path="data/surrogate_model.pt",
)

def run_simulation_with_surrogate(netlist_text: str, task_id: int):
    """ngspice를 surrogate로 가속."""
    pred, unc = _surrogate.predict(netlist_text)

    if unc < CircuitSurrogate.UNCERTAINTY_THRESHOLD and _surrogate.status()["trained_on"] > 0:
        # GNN이 확신 → ngspice skip
        print(f"  [Surrogate] ngspice SKIP (uncertainty={unc:.3f})")
        return pred, False   # (perf_dict, used_ngspice)
    else:
        # ngspice 실행
        sim_output = run_ngspice(netlist_text)          # 기존 함수
        _surrogate.save_to_db(netlist_text, sim_output, task_id)  # DB 저장
        perf = parse_sim_output(sim_output)             # 기존 파서
        return perf, True

def get_llm_hint(netlist_text: str) -> str:
    """optimize_template에 삽입할 GNN bottleneck 분석."""
    return _surrogate.get_bottleneck_hint(netlist_text)
# ─────────────────────────────────────────────────────────────────────────
'''


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CircuitSurrogate CLI")
    parser.add_argument("--db",    default="data/circuit_tag", help="DB 디렉토리")
    parser.add_argument("--model", default="data/surrogate_model.pt", help="모델 경로")
    parser.add_argument("--action", choices=["status", "train", "predict", "patch"],
                        default="status")
    parser.add_argument("--netlist", default=None, help="예측할 넷리스트 파일 경로")
    parser.add_argument("--force",   action="store_true", help="강제 재학습")
    args = parser.parse_args()

    surrogate = CircuitSurrogate(db_dir=args.db, model_path=args.model)

    if args.action == "status":
        surrogate.print_status()

    elif args.action == "train":
        if surrogate.should_retrain() or args.force:
            surrogate.train(force=args.force)
        else:
            print(f"재학습 불필요 (DB={len(surrogate.db)}개, "
                  f"last_train={surrogate._trained_on})")

    elif args.action == "predict":
        if args.netlist:
            netlist = Path(args.netlist).read_text()
        else:
            # 빠른 테스트용 내장 넷리스트
            netlist = """Two-Stage Op-Amp
VDD vdd 0 DC 1.8
M1 vout1 vinp vs1 vss NMOS W=10u L=0.18u
M2 vout2 vinn vs1 vss NMOS W=10u L=0.18u
M3 vs1 vbias vss vss NMOS W=5u L=0.5u
M4 vout1 vout1 vdd vdd PMOS W=20u L=0.18u
M5 vout2 vout1 vdd vdd PMOS W=20u L=0.18u
M6 vout vout2 vss vss NMOS W=40u L=0.18u
CC vout2 vout 3p
CL vout 0 10p
"""
        perf, unc = surrogate.predict(netlist)
        print("\n예측 결과:")
        for k, v in perf.items():
            print(f"  {k:20s}: {v:.3f}")
        print(f"  {'uncertainty':20s}: {unc:.4f}")
        should_skip = unc < CircuitSurrogate.UNCERTAINTY_THRESHOLD
        print(f"\n  → ngspice {'SKIP 가능' if should_skip else '검증 필요'} "
              f"(threshold={CircuitSurrogate.UNCERTAINTY_THRESHOLD})")

        hint = surrogate.get_bottleneck_hint(netlist)
        if hint:
            print(hint)

    elif args.action == "patch":
        print("\n아래 코드를 run.py에 추가하세요:\n")
        print(RUN_PY_PATCH)
