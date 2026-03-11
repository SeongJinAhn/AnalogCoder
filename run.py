"""
  python run.py --task_id=19 --num_per_task=3 \\
                --model=claude-sonnet-4-20250514 \\
                --api_key="sk-..." --base_url="https://api.anthropic.com/v1"
"""

import os
import sys
import re
import json
import time
import base64
import logging
import argparse
import subprocess
import traceback
import importlib.util
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# ── OpenAI-compatible client ──────────────────────────────────────────────────
from openai import OpenAI

# [SURROGATE] CircuitSurrogate import
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from surrogate.train_surrogate import CircuitSurrogate
    _surrogate_available = True
except ImportError:
    _surrogate_available = False
    print("[WARN] surrogate 모듈 없음 — ngspice only 모드로 실행")

# ─────────────────────────────────────────────────────────────────────────────
# 로거 설정
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AnalogCoderPro")


# ─────────────────────────────────────────────────────────────────────────────
# 설정 상수
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRY          = 5    # LLM 코드 생성 재시도 횟수
MAX_VLM_DEBUG      = 3    # VLM 파형 디버그 횟수
MAX_BO_ITER        = 20   # Bayesian Optimization 최대 iteration
BO_INIT_RANDOM     = 5    # BO 초기 랜덤 탐색 수
SURROGATE_SKIP_UNC = 0.15 # [SURROGATE] 불확실도 임계값 (낮으면 ngspice skip)


# ─────────────────────────────────────────────────────────────────────────────
# 템플릿 로더
# ─────────────────────────────────────────────────────────────────────────────

def load_template(name: str) -> str:
    """prompt_template*.md 파일 로드."""
    path = Path(__file__).parent / f"{name}.md"
    if not path.exists():
        log.warning(f"템플릿 없음: {path}")
        return ""
    return path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# problem_set.tsv 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_problem_set(path: str = "problem_set.tsv") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df


def get_task(df: pd.DataFrame, task_id: int) -> dict:
    row = df[df["task_id"] == task_id]
    if row.empty:
        raise ValueError(f"task_id={task_id} not found in problem_set.tsv")
    return row.iloc[0].to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# LLM 호출
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    """OpenAI-compatible API 호출. 텍스트 응답 반환."""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.warning(f"LLM 호출 실패 (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM 호출 3회 모두 실패")


def call_vlm(
    client: OpenAI,
    model: str,
    prompt: str,
    image_path: str,
) -> str:
    """VLM 호출 — 파형 이미지 + 텍스트."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lstrip(".")
    media_type = f"image/{ext if ext != 'jpg' else 'jpeg'}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text",       "text": prompt},
            {"type": "image_url",  "image_url": {
                "url": f"data:{media_type};base64,{img_b64}"
            }},
        ]
    }]
    return call_llm(client, model, messages, temperature=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 코드 추출 & 실행
# ─────────────────────────────────────────────────────────────────────────────

def extract_python_code(llm_response: str) -> str:
    """LLM 응답에서 ```python ... ``` 블록 추출."""
    patterns = [
        r"```python\s*(.*?)```",
        r"```\s*(import.*?)```",
    ]
    for pat in patterns:
        m = re.search(pat, llm_response, re.DOTALL)
        if m:
            return m.group(1).strip()
    # 코드블록 없으면 전체 반환
    return llm_response.strip()


def run_python_code(code: str, save_path: Path) -> tuple[bool, str]:
    """
    생성된 PySpice 코드를 파일로 저장 후 subprocess로 실행.
    Returns: (success, stdout+stderr)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(code, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(save_path)],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: 실행 시간 초과 (120s)"
    except Exception as e:
        return False, f"EXECUTION_ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# ngspice 시뮬레이션 & 결과 파싱
# ─────────────────────────────────────────────────────────────────────────────

def run_ngspice(netlist_path: str) -> tuple[bool, str]:
    """
    ngspice를 subprocess로 실행.
    Returns: (success, stdout)
    """
    try:
        result = subprocess.run(
            ["ngspice", "-b", netlist_path],
            capture_output=True, text=True, timeout=60,
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0 and "Error" not in output
        return success, output
    except FileNotFoundError:
        return False, "ngspice not found — conda install -c conda-forge ngspice"
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT: ngspice 60s 초과"
    except Exception as e:
        return False, str(e)


def check_pass(task: dict, sim_output: str, output_dir: Path) -> tuple[bool, str]:
    """
    problem_check/ 디렉토리의 테스트벤치로 pass/fail 판정.
    Returns: (passed, message)
    """
    check_script = Path("problem_check") / f"p{task['task_id']}_check.py"
    if not check_script.exists():
        # 체크 스크립트 없으면 시뮬 성공 여부로 판단
        return "Error" not in sim_output, "No check script — sim output check only"

    try:
        result = subprocess.run(
            [sys.executable, str(check_script), str(output_dir)],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout + result.stderr
        passed = "PASS" in output or result.returncode == 0
        return passed, output
    except Exception as e:
        return False, str(e)


def extract_netlist_from_output(output_dir: Path) -> str:
    """실행 결과 디렉토리에서 .sp 넷리스트 파일 찾기."""
    for sp_file in output_dir.glob("*.sp"):
        return sp_file.read_text(errors="ignore")
    for sp_file in output_dir.glob("*.net"):
        return sp_file.read_text(errors="ignore")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 프롬프트 빌더
# ─────────────────────────────────────────────────────────────────────────────

def build_generation_prompt(task: dict, template: str) -> list[dict]:
    """회로 생성 프롬프트 구성."""
    is_complex = task.get("is_complex", False)
    system = (
        "You are an expert analog circuit designer. "
        "Generate complete, runnable PySpice Python code for the given circuit specification. "
        "The code must: (1) create the circuit netlist, (2) run ngspice simulation, "
        "(3) save waveform plots as PNG, (4) print key performance metrics."
    )
    user_content = template.replace("{TASK_DESCRIPTION}", task.get("description", ""))
    user_content = user_content.replace("{TASK_ID}", str(task["task_id"]))
    user_content = user_content.replace("{SPECIFICATIONS}", task.get("specifications", ""))

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


def build_debug_prompt(
    task: dict,
    error_template: str,
    prev_code: str,
    error_msg: str,
    error_type: str = "execution",   # "execution" | "simulation"
) -> list[dict]:
    """오류 수정 프롬프트 구성."""
    system = (
        "You are an expert analog circuit designer and debugger. "
        "Fix the provided PySpice code based on the error message."
    )
    template_key = "execution_error" if error_type == "execution" else "simulation_error"
    err_template = load_template(template_key)

    user = (
        f"{err_template}\n\n"
        f"## Original Code\n```python\n{prev_code}\n```\n\n"
        f"## Error\n```\n{error_msg[:2000]}\n```\n\n"
        f"Please fix the code and return the complete corrected version."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_vlm_debug_prompt(
    task: dict,
    vlm_template: str,
    prev_code: str,
    sim_output: str,
) -> str:
    """VLM 파형 디버깅 프롬프트."""
    return (
        f"{vlm_template}\n\n"
        f"## Task\n{task.get('description', '')}\n\n"
        f"## Specifications\n{task.get('specifications', '')}\n\n"
        f"## Current Code\n```python\n{prev_code}\n```\n\n"
        f"## Simulation Output\n```\n{sim_output[:1000]}\n```\n\n"
        "Analyze the waveform image and suggest specific fixes to meet the specifications."
    )


def build_optimize_prompt(
    task: dict,
    optimize_template: str,
    current_code: str,
    sim_output: str,
    param_space: dict,
    gnn_hint: str = "",          # [SURROGATE] GNN bottleneck 힌트
) -> list[dict]:
    """BO 최적화 프롬프트 구성."""
    system = (
        "You are an expert analog circuit optimizer. "
        "Extract device parameters for Bayesian Optimization from the given circuit code. "
        "Return ONLY a JSON object with parameter names and their current values."
    )
    user = (
        f"{optimize_template}\n\n"
        f"## Task\n{task.get('description', '')}\n\n"
        f"## Current Circuit Code\n```python\n{current_code}\n```\n\n"
        f"## Simulation Output\n{sim_output[:500]}\n"
        + (f"\n{gnn_hint}" if gnn_hint else "") +
        f"\n\n## Parameter Space\n{json.dumps(param_space, indent=2)}\n\n"
        "Return JSON with parameter names → values to try next."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Optimization
# ─────────────────────────────────────────────────────────────────────────────

class GaussianProcessBO:
    """
    경량 GP-BO (scipy 기반).
    실제 AnalogCoderPro의 BO 로직을 모사.
    """

    def __init__(self, param_bounds: dict):
        """
        param_bounds: {"W1": (5e-6, 50e-6), "L1": (0.18e-6, 1e-6), ...}
        """
        self.bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def _normalize(self, params: dict) -> np.ndarray:
        x = []
        for name in self.param_names:
            lo, hi = self.bounds[name]
            x.append((params[name] - lo) / (hi - lo + 1e-12))
        return np.array(x)

    def _denormalize(self, x: np.ndarray) -> dict:
        return {name: x[i] * (self.bounds[name][1] - self.bounds[name][0]) + self.bounds[name][0]
                for i, name in enumerate(self.param_names)}

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray, length=0.5) -> np.ndarray:
        diff = X1[:, None, :] - X2[None, :, :]
        return np.exp(-0.5 * (diff**2).sum(-1) / length**2)

    def _gp_predict(self, x_new: np.ndarray) -> tuple[float, float]:
        if len(self.X) < 2:
            return 0.0, 1.0
        X = np.stack(self.X)
        y = np.array(self.y)
        noise = 1e-4
        K   = self._rbf_kernel(X, X) + noise * np.eye(len(X))
        K_s = self._rbf_kernel(x_new[None], X)[0]
        K_ss = 1.0
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            mu  = float(K_s @ alpha)
            v   = np.linalg.solve(L, K_s)
            std = float(np.sqrt(max(K_ss - v @ v, 1e-8)))
        except np.linalg.LinAlgError:
            mu, std = 0.0, 1.0
        return mu, std

    def acquisition_ei(self, x: np.ndarray, xi: float = 0.01) -> float:
        """Expected Improvement."""
        mu, std = self._gp_predict(x)
        best = max(self.y) if self.y else 0.0
        z = (mu - best - xi) / (std + 1e-8)
        return float((mu - best - xi) * norm.cdf(z) + std * norm.pdf(z))

    def suggest_next(self) -> dict:
        """다음 탐색 포인트 제안."""
        if len(self.X) < BO_INIT_RANDOM:
            # 초기 랜덤 탐색
            x = np.random.rand(len(self.param_names))
        else:
            # EI 최대화
            best_ei, best_x = -np.inf, np.random.rand(len(self.param_names))
            for _ in range(20):
                x0 = np.random.rand(len(self.param_names))
                res = minimize(
                    lambda x: -self.acquisition_ei(x),
                    x0, method="L-BFGS-B",
                    bounds=[(0, 1)] * len(self.param_names),
                )
                if -res.fun > best_ei:
                    best_ei, best_x = -res.fun, res.x
            x = best_x
        return self._denormalize(x)

    def update(self, params: dict, score: float):
        self.X.append(self._normalize(params))
        self.y.append(score)

    def best(self) -> tuple[dict, float]:
        if not self.y:
            return {}, -np.inf
        idx = np.argmax(self.y)
        return self._denormalize(self.X[idx]), self.y[idx]


def extract_param_space(task: dict) -> dict:
    """task 정보에서 BO 파라미터 공간 추출."""
    # task TSV에 param_space 컬럼이 있으면 파싱
    if "param_space" in task and pd.notna(task.get("param_space")):
        try:
            return json.loads(task["param_space"])
        except Exception:
            pass
    # 기본값: MOSFET W/L + 커패시터
    return {
        "W_input":  (2e-6,  50e-6),
        "L_input":  (0.18e-6, 1e-6),
        "W_output": (10e-6, 200e-6),
        "L_output": (0.18e-6, 1e-6),
        "Cc":       (0.5e-12, 20e-12),
    }


def score_from_sim(sim_output: str, task: dict) -> float:
    """
    시뮬 결과에서 최적화 목적함수 계산.
    높을수록 좋음.
    """
    score = 0.0
    # gain
    m = re.search(r"gain\s*[=:]\s*([\-\d.e+]+)\s*dB", sim_output, re.I)
    if m:
        score += min(float(m.group(1)), 100) / 100  # normalize

    # phase margin
    m = re.search(r"phase.margin\s*[=:]\s*([\d.e+]+)", sim_output, re.I)
    if m:
        pm = float(m.group(1))
        score += 1.0 if 45 <= pm <= 80 else 0.0

    # success
    if re.search(r"PASS|SUCCESS", sim_output, re.I):
        score += 2.0

    return score


def apply_params_to_code(code: str, params: dict) -> str:
    """
    BO 제안 파라미터를 코드에 적용.
    PySpice 코드의 W=, L=, C= 패턴 교체.
    """
    # 간단한 치환: 실제 AnalogCoderPro는 더 정교한 방법 사용
    result = code
    param_map = {
        "W_input":  r"(M[12].*?W=)([\d.e+-]+)(u)",
        "L_input":  r"(M[12].*?L=)([\d.e+-]+)(u)",
        "W_output": r"(M[67].*?W=)([\d.e+-]+)(u)",
        "L_output": r"(M[67].*?L=)([\d.e+-]+)(u)",
    }
    for param, pattern in param_map.items():
        if param in params:
            val_u = params[param] * 1e6  # m → µm
            result = re.sub(pattern, lambda m, v=val_u: f"{m.group(1)}{v:.3f}{m.group(3)}",
                           result)
    if "Cc" in params:
        val_p = params["Cc"] * 1e12
        result = re.sub(r"(CC\s+=\s*)([\d.e+-]+)(p)", f"\\g<1>{val_p:.2f}\\3", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

class AnalogCoderProPipeline:
    """
    AnalogCoderPro 전체 파이프라인.

    Phase 1: LLM 코드 생성 + 실행 오류 수정
    Phase 2: VLM 파형 디버깅
    Phase 3: BO 파라미터 최적화 (+ Surrogate 가속)
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url or "https://api.openai.com/v1",
        )
        self.problem_df = load_problem_set()

        # 템플릿 로드
        self.tpl_gen      = load_template("prompt_template")
        self.tpl_complex  = load_template("prompt_template_comptex")
        self.tpl_optimize = load_template("prompt_template_optimize")
        self.tpl_vlm      = load_template("vlm_debug_prompt")

        # [SURROGATE] Surrogate 초기화
        if _surrogate_available:
            self.surrogate = CircuitSurrogate(
                db_dir=str(Path(args.output_dir) / "circuit_tag_db"),
                model_path=str(Path(args.output_dir) / "surrogate_model.pt"),
            )
            log.info("[SURROGATE] CircuitSurrogate 초기화 완료")
        else:
            self.surrogate = None

        # 결과 저장 루트
        self.output_root = Path(args.output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: 코드 생성 ───────────────────────────────────────────────────

    def phase1_generate(self, task: dict, attempt_dir: Path) -> tuple[bool, str, str]:
        """
        LLM으로 코드 생성 → 실행 → 실행오류 수정 반복.
        Returns: (success, code, execution_output)
        """
        is_complex = task.get("circuit_type", "").lower() in ("mixer", "oscillator", "pll")
        template   = self.tpl_complex if is_complex else self.tpl_gen
        messages   = build_generation_prompt(task, template)
        code       = ""
        exec_out   = ""

        for trial in range(MAX_RETRY):
            log.info(f"  [Phase1] 코드 생성 시도 {trial+1}/{MAX_RETRY}")

            if trial == 0:
                response = call_llm(self.client, self.args.model, messages)
            else:
                # 이전 오류로 수정 요청
                fix_messages = build_debug_prompt(
                    task, "", code, exec_out, error_type="execution"
                )
                response = call_llm(self.client, self.args.model, fix_messages)

            code = extract_python_code(response)
            code_path = attempt_dir / "circuit.py"
            success, exec_out = run_python_code(code, code_path)

            if success:
                log.info(f"  [Phase1] ✓ 실행 성공")
                return True, code, exec_out
            else:
                log.info(f"  [Phase1] ✗ 실행 실패: {exec_out[:100]}")

        return False, code, exec_out

    # ── Phase 2: VLM 파형 디버깅 ─────────────────────────────────────────────

    def phase2_vlm_debug(
        self,
        task: dict,
        attempt_dir: Path,
        code: str,
        sim_output: str,
    ) -> tuple[bool, str, str]:
        """
        파형 이미지를 VLM에 넘겨 디버깅.
        Returns: (success, fixed_code, sim_output)
        """
        for debug_round in range(MAX_VLM_DEBUG):
            # 파형 이미지 찾기
            waveform_imgs = list(attempt_dir.glob("*.png"))
            if not waveform_imgs:
                log.info("  [Phase2] 파형 이미지 없음 — VLM 스킵")
                break

            img_path = str(waveform_imgs[0])
            log.info(f"  [Phase2] VLM 디버깅 round {debug_round+1} | {img_path}")

            vlm_prompt = build_vlm_debug_prompt(task, self.tpl_vlm, code, sim_output)
            vlm_response = call_vlm(self.client, self.args.model, vlm_prompt, img_path)
            fixed_code   = extract_python_code(vlm_response)

            if not fixed_code or fixed_code == code:
                log.info("  [Phase2] VLM이 동일 코드 반환 — 중단")
                break

            code = fixed_code
            code_path = attempt_dir / f"circuit_vlm_{debug_round}.py"
            success, exec_out = run_python_code(code, code_path)

            if not success:
                sim_output = exec_out
                continue

            # 시뮬 체크
            passed, check_msg = check_pass(task, exec_out, attempt_dir)
            if passed:
                log.info(f"  [Phase2] ✓ VLM 수정 후 PASS")
                return True, code, exec_out
            sim_output = exec_out

        return False, code, sim_output

    # ── Phase 3: BO 최적화 ───────────────────────────────────────────────────

    def phase3_bo_optimize(
        self,
        task: dict,
        attempt_dir: Path,
        base_code: str,
    ) -> tuple[bool, str, float]:
        """
        Bayesian Optimization으로 디바이스 파라미터 최적화.
        [SURROGATE] ngspice 앞에 GNN 필터 적용.
        Returns: (success, best_code, best_score)
        """
        param_space = extract_param_space(task)
        bo          = GaussianProcessBO(param_space)
        best_code   = base_code
        best_score  = -np.inf
        ngspice_calls = 0
        surrogate_hits = 0

        log.info(f"  [Phase3] BO 시작 (max_iter={MAX_BO_ITER})")

        # [SURROGATE] LLM 프롬프트에 GNN bottleneck 힌트 삽입
        gnn_hint = ""
        if self.surrogate:
            netlist_text = extract_netlist_from_output(attempt_dir)
            if netlist_text:
                gnn_hint = self.surrogate.get_bottleneck_hint(netlist_text)
                if gnn_hint:
                    log.info(f"  [SURROGATE] GNN hint 생성:\n{gnn_hint}")

        for bo_iter in range(MAX_BO_ITER):
            params = bo.suggest_next()
            candidate_code = apply_params_to_code(base_code, params)
            code_path = attempt_dir / f"bo_iter_{bo_iter}.py"

            # [SURROGATE] 먼저 GNN surrogate로 예측
            use_ngspice = True
            score = None

            if self.surrogate and self.surrogate.status()["trained_on"] > 0:
                netlist_candidate = extract_netlist_from_output(
                    attempt_dir / f"bo_iter_{bo_iter-1}" if bo_iter > 0 else attempt_dir
                ) or ""
                if netlist_candidate:
                    pred, unc = self.surrogate.predict(netlist_candidate)
                    if unc < SURROGATE_SKIP_UNC:
                        # GNN 확신 → ngspice skip
                        score = pred.get("success", 0) + pred.get("gain_dB", 0) / 100
                        use_ngspice = False
                        surrogate_hits += 1
                        log.info(f"  [SURROGATE] iter {bo_iter} | ngspice SKIP "
                                 f"(unc={unc:.3f}) | score≈{score:.3f}")

            if use_ngspice:
                # ngspice 실행
                exec_success, exec_out = run_python_code(candidate_code, code_path)
                if not exec_success:
                    score = -1.0
                else:
                    score = score_from_sim(exec_out, task)
                    ngspice_calls += 1

                    # [SURROGATE] 결과 DB에 저장 (재학습 트리거 포함)
                    if self.surrogate:
                        netlist_new = extract_netlist_from_output(code_path.parent)
                        if netlist_new:
                            self.surrogate.save_to_db(
                                netlist_new, exec_out, task_id=task["task_id"]
                            )

            bo.update(params, score)

            if score > best_score:
                best_score = score
                best_code  = candidate_code
                log.info(f"  [Phase3] iter {bo_iter:3d} | score={score:.4f} ★ NEW BEST "
                         f"| ngspice={ngspice_calls} surrogate={surrogate_hits}")

                # 최선 코드로 pass 체크
                if use_ngspice:
                    passed, check_msg = check_pass(task, exec_out, attempt_dir)
                    if passed:
                        log.info(f"  [Phase3] ✓ BO PASS (iter={bo_iter})")
                        return True, best_code, best_score
            else:
                log.debug(f"  [Phase3] iter {bo_iter:3d} | score={score:.4f}")

        log.info(f"  [Phase3] BO 완료 | best={best_score:.4f} "
                 f"ngspice_calls={ngspice_calls} surrogate_hits={surrogate_hits}")

        # surrogate 절감율 로그
        total_possible = ngspice_calls + surrogate_hits
        if total_possible > 0:
            saving_pct = 100 * surrogate_hits / total_possible
            log.info(f"  [SURROGATE] ngspice 절감율: {saving_pct:.1f}% "
                     f"({surrogate_hits}/{total_possible})")

        return False, best_code, best_score

    # ── 전체 실행 ────────────────────────────────────────────────────────────

    def run_task(self, task_id: int) -> dict:
        """
        단일 task에 대해 num_per_task번 시도.
        Returns: 결과 요약 dict
        """
        task     = get_task(self.problem_df, task_id)
        task_dir = self.output_root / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*60}")
        log.info(f"Task {task_id}: {task.get('description', 'Unknown')}")
        log.info(f"{'='*60}")

        results = []
        for attempt in range(self.args.num_per_task):
            attempt_dir = task_dir / f"attempt_{attempt}"
            attempt_dir.mkdir(exist_ok=True)
            log.info(f"\n[Attempt {attempt+1}/{self.args.num_per_task}]")

            result = self._run_single_attempt(task, attempt_dir)
            results.append(result)

            if result["passed"]:
                log.info(f"✓ Task {task_id} PASSED (attempt {attempt+1})")
                break

        # 요약 저장
        summary = {
            "task_id":      task_id,
            "description":  task.get("description", ""),
            "attempts":     len(results),
            "passed":       any(r["passed"] for r in results),
            "best_score":   max((r.get("score", -np.inf) for r in results), default=-np.inf),
            "results":      results,
        }
        (task_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _run_single_attempt(self, task: dict, attempt_dir: Path) -> dict:
        """단일 attempt 실행."""
        result = {"passed": False, "score": -np.inf, "phase": None}

        try:
            # ── Phase 1: 코드 생성 ──
            p1_ok, code, exec_out = self.phase1_generate(task, attempt_dir)
            if not p1_ok:
                result["phase"] = "phase1_fail"
                result["error"] = exec_out[:200]
                return result

            # 첫 시뮬 pass 체크
            passed, check_msg = check_pass(task, exec_out, attempt_dir)
            if passed:
                result.update({"passed": True, "phase": "phase1", "score": 2.0})
                # [SURROGATE] 성공 회로 저장
                self._save_to_surrogate(attempt_dir, exec_out, task)
                return result

            # ── Phase 2: VLM 디버깅 ──
            p2_ok, code, sim_out = self.phase2_vlm_debug(
                task, attempt_dir, code, exec_out
            )
            if p2_ok:
                result.update({"passed": True, "phase": "phase2", "score": 1.5})
                self._save_to_surrogate(attempt_dir, sim_out, task)
                return result

            # ── Phase 3: BO 최적화 ──
            p3_ok, best_code, best_score = self.phase3_bo_optimize(
                task, attempt_dir, code
            )
            result.update({
                "passed": p3_ok,
                "phase":  "phase3",
                "score":  float(best_score),
            })
            if p3_ok:
                (attempt_dir / "circuit_best.py").write_text(best_code)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(f"  실행 중 예외: {e}")
            log.debug(traceback.format_exc())
            result["error"] = str(e)

        return result

    def _save_to_surrogate(self, attempt_dir: Path, sim_output: str, task: dict):
        """[SURROGATE] 시뮬 결과를 surrogate DB에 저장."""
        if not self.surrogate:
            return
        netlist = extract_netlist_from_output(attempt_dir)
        if netlist:
            self.surrogate.save_to_db(
                netlist, sim_output, task_id=task["task_id"]
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AnalogCoderPro — LLM-based analog circuit design + CircuitSurrogate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task_id",      type=int, required=True,
                        help="Task ID from problem_set.tsv (e.g. 19 for Mixer)")
    parser.add_argument("--num_per_task", type=int, default=3,
                        help="Number of independent generation attempts per task")
    parser.add_argument("--model",        type=str, default="gpt-4o",
                        help="LLM/VLM model name")
    parser.add_argument("--api_key",      type=str, default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI-compatible API key")
    parser.add_argument("--base_url",     type=str, default=None,
                        help="Custom API base URL (e.g. for Claude, local LLM)")
    parser.add_argument("--output_dir",   type=str, default="outputs",
                        help="Directory to save generated circuits and logs")
    parser.add_argument("--no_surrogate", action="store_true",
                        help="Disable CircuitSurrogate (ngspice only)")
    parser.add_argument("--surrogate_status", action="store_true",
                        help="Print surrogate status and exit")
    parser.add_argument("--train_surrogate",  action="store_true",
                        help="Force train surrogate on existing DB and exit")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Surrogate 단독 명령
    if (args.surrogate_status or args.train_surrogate) and _surrogate_available:
        sur = CircuitSurrogate(
            db_dir=str(Path(args.output_dir) / "circuit_tag_db"),
            model_path=str(Path(args.output_dir) / "surrogate_model.pt"),
        )
        if args.surrogate_status:
            sur.print_status()
        if args.train_surrogate:
            sur.train(force=True)
        return

    if not args.api_key:
        log.error("API key 없음. --api_key 또는 OPENAI_API_KEY 환경변수 설정 필요")
        sys.exit(1)

    # no_surrogate 플래그 처리
    if args.no_surrogate:
        global _surrogate_available
        _surrogate_available = False

    pipeline = AnalogCoderProPipeline(args)
    summary  = pipeline.run_task(args.task_id)

    # 최종 결과 출력
    print(f"\n{'='*60}")
    print(f" Task {args.task_id} 결과")
    print(f"{'='*60}")
    print(f"  상태     : {'✓ PASS' if summary['passed'] else '✗ FAIL'}")
    print(f"  시도 횟수: {summary['attempts']}")
    print(f"  최고점수 : {summary['best_score']:.4f}")

    if _surrogate_available and pipeline.surrogate:
        status = pipeline.surrogate.status()
        print(f"\n  [SURROGATE]")
        print(f"  DB 크기  : {status['db_size']}개")
        print(f"  학습여부 : {status['model_exists']}")
    print(f"{'='*60}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
