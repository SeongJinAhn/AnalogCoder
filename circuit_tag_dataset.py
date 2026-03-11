"""
circuit_tag_dataset.py
----------------------
PyTorch Geometric Dataset that wraps AnalogCoderPro's generated circuits
into Text-Attributed Graphs (TAGs) ready for GNN training.

Pipeline:
  AnalogCoderPro run.py output (netlist .py / .sp files)
  ──► parse_netlist()
  ──► extract_performance_from_sim()     (reads ngspice stdout / output files)
  ──► circuit_graph_to_pyg()
  ──► LM text encoder (BERT / sentence-transformers)
  ──► CircuitTAGDataset

Usage:
  dataset = CircuitTAGDataset(root="./data/circuit_tag")
  loader  = DataLoader(dataset, batch_size=16, shuffle=True)
"""

import os
import re
import json
import glob
from pathlib import Path
from typing import Optional, Callable

import torch
from torch_geometric.data import Dataset, Data

from netlist_parser import parse_netlist, circuit_graph_to_pyg


# ─────────────────────────────────────────────
# Performance metric extraction
# ─────────────────────────────────────────────

# Maps problem_set.tsv metrics to regex patterns for ngspice output
METRIC_PATTERNS = {
    # Op-amp
    "gain_db":         r"gain\s*[=:]\s*([\-\d.e+]+)\s*dB",
    "ugf_hz":          r"(?:ugf|unity.gain.frequency|GBW)\s*[=:]\s*([\d.e+]+)",
    "phase_margin_deg":r"phase.margin\s*[=:]\s*([\d.e+]+)",
    "cmrr_db":         r"CMRR\s*[=:]\s*([\d.e+]+)",
    "power_uw":        r"power\s*[=:]\s*([\d.e+]+)\s*[uµ]?W",
    # Filter
    "cutoff_hz":       r"(?:cutoff|fc|f_?3dB)\s*[=:]\s*([\d.e+]+)",
    "stopband_atten":  r"stopband.attenuation\s*[=:]\s*([\d.e+]+)",
    # Oscillator
    "osc_freq_hz":     r"oscillation.freq(?:uency)?\s*[=:]\s*([\d.e+]+)",
    "thd_pct":         r"THD\s*[=:]\s*([\d.e+]+)\s*%",
    # Mixer
    "conversion_gain": r"conversion.gain\s*[=:]\s*([\-\d.e+]+)",
    "iip3_dbm":        r"IIP3\s*[=:]\s*([\-\d.e+]+)",
    # General
    "success":         r"(?:PASS|SUCCESS|✓)",
}

PERFORMANCE_KEYS = [
    "gain_db", "ugf_hz", "phase_margin_deg", "power_uw",
    "cutoff_hz", "osc_freq_hz", "conversion_gain", "success",
]


def extract_performance_from_sim_output(sim_output: str) -> dict:
    """
    Parse ngspice simulation stdout / result file text into a
    performance metric dictionary.
    """
    perf = {}
    for key, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, sim_output, re.IGNORECASE)
        if m:
            if key == "success":
                perf[key] = 1.0
            else:
                try:
                    perf[key] = float(m.group(1))
                except ValueError:
                    pass
    if "success" not in perf:
        if re.search(r"FAIL|ERROR|did not converge", sim_output, re.IGNORECASE):
            perf["success"] = 0.0
    return perf


def load_analogcoder_run_dir(run_dir: str) -> list[dict]:
    """
    Walk an AnalogCoderPro output directory and collect
    (netlist_text, sim_output, task_id, attempt_id) records.

    AnalogCoderPro typically writes:
      <run_dir>/task_<id>/attempt_<n>/circuit.py   (generated PySpice code)
      <run_dir>/task_<id>/attempt_<n>/netlist.sp   (exported SPICE netlist)
      <run_dir>/task_<id>/attempt_<n>/sim_output.txt
    """
    records = []
    for attempt_dir in sorted(glob.glob(os.path.join(run_dir, "**", "attempt_*"), recursive=True)):
        netlist_path = os.path.join(attempt_dir, "netlist.sp")
        sim_path = os.path.join(attempt_dir, "sim_output.txt")

        if not os.path.exists(netlist_path):
            # Try to find any .sp file
            sp_files = glob.glob(os.path.join(attempt_dir, "*.sp"))
            if sp_files:
                netlist_path = sp_files[0]
            else:
                continue

        netlist_text = Path(netlist_path).read_text(errors="ignore")
        sim_output = Path(sim_path).read_text(errors="ignore") if os.path.exists(sim_path) else ""

        # Infer task_id from directory name
        parts = attempt_dir.replace("\\", "/").split("/")
        task_id = next((p.replace("task_", "") for p in parts if p.startswith("task_")), "0")
        attempt_id = next((p.replace("attempt_", "") for p in parts if p.startswith("attempt_")), "0")

        records.append({
            "netlist_text": netlist_text,
            "sim_output":   sim_output,
            "task_id":      task_id,
            "attempt_id":   attempt_id,
            "path":         attempt_dir,
        })
    return records


# ─────────────────────────────────────────────
# Text encoder wrappers
# ─────────────────────────────────────────────

class BagOfWordsEncoder:
    """
    Simple BoW encoder for quick experiments (no heavy LM dependencies).
    Vocab built from component type keywords.
    """
    VOCAB = [
        "resistor", "capacitor", "inductor", "mosfet", "bjt", "diode",
        "voltage", "current", "source", "nmos", "pmos", "npn", "pnp",
        "dc", "ac", "pulse", "sin", "subcircuit", "vcvs", "vccs",
        "drain", "gate", "source", "base", "collector", "emitter",
        "vdd", "vss", "gnd", "out", "in", "bias", "comp", "miller",
        "passive", "active", "controlled", "connected",
    ]

    def __init__(self):
        self.word2idx = {w: i for i, w in enumerate(self.VOCAB)}
        self.dim = len(self.VOCAB)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        vecs = torch.zeros(len(texts), self.dim)
        for i, text in enumerate(texts):
            words = re.findall(r"[a-z]+", text.lower())
            for w in words:
                if w in self.word2idx:
                    vecs[i, self.word2idx[w]] += 1.0
            # L2 normalize
            norm = vecs[i].norm()
            if norm > 0:
                vecs[i] /= norm
        return vecs


class SentenceTransformerEncoder:
    """
    High-quality text encoder using sentence-transformers.
    pip install sentence-transformers
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings  # [N, 384]


# ─────────────────────────────────────────────
# PyG Dataset
# ─────────────────────────────────────────────

class CircuitTAGDataset(Dataset):
    """
    Text-Attributed Graph dataset of analog circuits.

    Each graph represents one circuit netlist:
      - data.x         : text-encoded node features [N, d]
      - data.x_text    : raw text descriptions (list of str)
      - data.edge_index: adjacency [2, E]
      - data.role      : component role labels [N]  (passive/active/source/…)
      - data.y         : performance metrics [1, num_metrics]
      - data.task_id   : circuit type/task identifier (int)
      - data.node_names: component names (list)

    Args:
        root            : directory to store processed files
        raw_run_dirs    : list of AnalogCoderPro run output directories
        text_encoder    : callable(list[str]) -> Tensor; defaults to BoW
        performance_keys: subset of PERFORMANCE_KEYS to include in y
        transform       : optional PyG transform
    """

    def __init__(
        self,
        root: str,
        raw_run_dirs: Optional[list[str]] = None,
        text_encoder: Optional[Callable] = None,
        performance_keys: Optional[list[str]] = None,
        transform=None,
    ):
        self.raw_run_dirs = raw_run_dirs or []
        self.text_encoder = text_encoder or BagOfWordsEncoder()
        self.performance_keys = performance_keys or PERFORMANCE_KEYS
        self._data_list: list[Data] = []
        super().__init__(root, transform=transform)

    @property
    def processed_file_names(self):
        return ["circuit_tag_dataset.pt"]

    def process(self):
        all_records = []
        for run_dir in self.raw_run_dirs:
            all_records.extend(load_analogcoder_run_dir(run_dir))

        print(f"[CircuitTAGDataset] Processing {len(all_records)} records...")

        data_list = []
        for rec in all_records:
            try:
                cg = parse_netlist(rec["netlist_text"])
                if len(cg.components) == 0:
                    continue

                # Extract performance labels
                cg.performance = extract_performance_from_sim_output(rec["sim_output"])

                # Convert to PyG Data
                data = circuit_graph_to_pyg(
                    cg,
                    text_encoder=self.text_encoder,
                    performance_keys=self.performance_keys,
                )

                # Attach metadata
                data.task_id = int(rec["task_id"]) if rec["task_id"].isdigit() else 0
                data.path    = rec["path"]

                data_list.append(data)
            except Exception as e:
                print(f"  [WARN] Failed on {rec.get('path', '?')}: {e}")

        torch.save(data_list, self.processed_paths[0])
        self._data_list = data_list
        print(f"[CircuitTAGDataset] Saved {len(data_list)} graphs.")

    def _load(self):
        if not self._data_list:
            self._data_list = torch.load(self.processed_paths[0])

    def len(self):
        self._load()
        return len(self._data_list)

    def get(self, idx):
        self._load()
        return self._data_list[idx]

    # ── Convenience: build from raw netlist strings (no file I/O) ──

    @classmethod
    def from_netlists(
        cls,
        netlists: list[str],
        sim_outputs: Optional[list[str]] = None,
        task_ids: Optional[list[int]] = None,
        text_encoder: Optional[Callable] = None,
        performance_keys: Optional[list[str]] = None,
    ) -> "CircuitTAGDataset":
        """
        Build an in-memory dataset directly from netlist strings.
        Useful for quick experiments without file I/O.
        """
        encoder = text_encoder or BagOfWordsEncoder()
        perf_keys = performance_keys or PERFORMANCE_KEYS
        data_list = []

        for i, netlist in enumerate(netlists):
            cg = parse_netlist(netlist)
            if len(cg.components) == 0:
                continue
            sim_out = sim_outputs[i] if sim_outputs else ""
            cg.performance = extract_performance_from_sim_output(sim_out)
            data = circuit_graph_to_pyg(cg, text_encoder=encoder, performance_keys=perf_keys)
            data.task_id = task_ids[i] if task_ids else i
            data_list.append(data)

        dummy = cls.__new__(cls)
        dummy._data_list = data_list
        dummy.performance_keys = perf_keys
        dummy.text_encoder = encoder
        return dummy


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    netlists = [
        """Two-Stage Op-Amp
VDD vdd 0 DC 1.8
M1 vout1 vin+ vs1 vss NMOS W=10u L=0.18u
M2 vout2 vin- vs1 vss NMOS W=10u L=0.18u
M3 vs1 vbias vss vss NMOS W=5u L=0.5u
M4 vout1 vout1 vdd vdd PMOS W=20u L=0.18u
M5 vout2 vout1 vdd vdd PMOS W=20u L=0.18u
M6 vout vout2 vss vss NMOS W=40u L=0.18u
CC vout2 vout 3p
RC vout2 rc_mid 200
CL vout 0 10p
""",
        """RC Low-pass Filter
VIN vin 0 AC 1
R1 vin vout 10k
C1 vout 0 1.6p
""",
    ]
    sim_outputs = [
        "gain = 60.2 dB  unity-gain frequency: 12.5 MHz  phase margin = 58.3 deg  PASS",
        "cutoff frequency: fc = 9.95 MHz  SUCCESS",
    ]

    dataset = CircuitTAGDataset.from_netlists(netlists, sim_outputs=sim_outputs)
    for i in range(len(dataset._data_list)):
        d = dataset._data_list[i]
        print(f"\n[Graph {i}] nodes={d.num_nodes}  edges={d.edge_index.shape[1]}")
        print(f"  x.shape = {d.x.shape}")
        print(f"  roles   = {d.role.tolist()}")
        print(f"  y       = {d.y}")
        for name, text in zip(d.node_names, d.x_text):
            print(f"    {name:8s} | {text[:80]}")
