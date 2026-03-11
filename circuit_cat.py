"""
circuit_cat.py
--------------
CircuitCAT: Conflict-Aware TAG model for analog circuit analysis.

Directly inspired by the CAT (Conflict-Aware TAG) architecture but
adapted for circuit netlists:

  * Text stream  : component descriptions (type, value, model, net names)
  * Graph stream : circuit topology via MPNN / GCN
  * Fusion gate  : per-node adaptive α based on text–graph disagreement

The semantic disagreement between, e.g., "Resistor R1 100kΩ connected to [vbias]"
and its structural role (current mirror load vs. bias divider) is exactly the
kind of conflict CAT was designed to exploit.

Architecture:
                    ┌──────────────────┐
  text_attrs ──────►│  Text Encoder    │──► h_text  [N, d_t]
                    └──────────────────┘
                                            ↓
                    ┌──────────────────┐    ┌──────────────────────┐
  edge_index ──────►│  Graph Encoder   │──► h_graph [N, d_g]       │
  (init: h_text)    └──────────────────┘    │  Conflict Gate α_i   │
                                            │  = σ(W·|h_t - h_g|)  │
                                            └──────────────────────┘
                                                     ↓
                                        h_fused = α·h_t + (1-α)·h_g

Tasks supported:
  1. Node classification  — predict component role (passive / active / source)
  2. Graph regression     — predict circuit performance metrics (gain, bandwidth…)
  3. Graph classification — predict circuit type (filter / amplifier / oscillator…)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch


# ─────────────────────────────────────────────
# Conflict-Aware Fusion Gate (core of CAT)
# ─────────────────────────────────────────────

class ConflictGate(nn.Module):
    """
    Per-node adaptive gate based on text–graph disagreement.

    α_i = σ( W_α · |h_text_i − h_graph_i| + b_α )

    High α_i → rely more on text features (component is "unusual" in topology)
    Low  α_i → rely more on graph features (component well-explained by structure)

    Collapse prevention:
      - Gate regularization loss pushes α toward 0.5 during early training
      - Gradient clipping recommended (max_norm=1.0)
    """

    def __init__(self, d_text: int, d_graph: int, d_hidden: int = 64):
        super().__init__()
        # Project to common dimension for difference computation
        self.proj_text  = nn.Linear(d_text,  d_hidden)
        self.proj_graph = nn.Linear(d_graph, d_hidden)
        # Gate MLP: |difference| → scalar in (0, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize gate toward 0.5 (balanced fusion at start)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, h_text: torch.Tensor, h_graph: torch.Tensor):
        """
        Args:
            h_text  : [N, d_text]
            h_graph : [N, d_graph]
        Returns:
            h_fused : [N, d_hidden]
            alpha   : [N, 1]  gate values (for monitoring collapse)
        """
        t = self.proj_text(h_text)
        g = self.proj_graph(h_graph)

        conflict = torch.abs(t - g)              # [N, d_hidden]
        alpha = torch.sigmoid(self.gate_mlp(conflict))   # [N, 1]

        h_fused = alpha * t + (1.0 - alpha) * g  # [N, d_hidden]
        return h_fused, alpha

    def gate_entropy_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Regularization: penalize gates far from 0.5.
        Encourages the gate to be informative rather than trivially all-0 or all-1.

        L_reg = mean( (α - 0.5)^2 )
        """
        return ((alpha - 0.5) ** 2).mean()


# ─────────────────────────────────────────────
# Text Encoder (lightweight, no external LM)
# ─────────────────────────────────────────────

class CircuitTextEncoder(nn.Module):
    """
    Maps pre-computed text feature vectors through an MLP.
    If using sentence-transformers, pass d_in=384; for BoW, d_in=vocab_size.
    """
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out * 2),
            nn.LayerNorm(d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Graph Encoder (MPNN over circuit topology)
# ─────────────────────────────────────────────

class CircuitGraphEncoder(nn.Module):
    """
    2-layer GraphSAGE encoder initialized with text features.

    We use SAGE rather than GCN because:
    - Circuit graphs can have high-degree nodes (VDD/GND buses)
    - SAGE's mean aggregation is more robust to degree imbalance
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = SAGEConv(d_in, d_hidden)
        self.conv2 = SAGEConv(d_hidden, d_out)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_out)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x, edge_index)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h, edge_index))
        return h


# ─────────────────────────────────────────────
# Main CircuitCAT model
# ─────────────────────────────────────────────

class CircuitCAT(nn.Module):
    """
    Conflict-Aware TAG model for analog circuits.

    Args:
        d_text_in   : dimension of input text features (BoW / sentence-transformer)
        d_hidden    : internal hidden dimension
        num_node_classes    : for node classification (component roles)
        num_perf_metrics    : for graph regression (performance prediction)
        num_circuit_types   : for graph classification (circuit type)
        dropout             : dropout rate
        gate_reg_weight     : weight for gate entropy regularization
    """

    def __init__(
        self,
        d_text_in: int = 44,       # BoW encoder dim by default
        d_hidden: int = 128,
        num_node_classes: int = 6,
        num_perf_metrics: int = 8,
        num_circuit_types: int = 10,
        dropout: float = 0.1,
        gate_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.gate_reg_weight = gate_reg_weight

        # ── Text stream ──
        self.text_enc = CircuitTextEncoder(d_text_in, d_hidden, dropout)

        # ── Graph stream (initialized with text features) ──
        self.graph_enc = CircuitGraphEncoder(d_hidden, d_hidden, d_hidden, dropout)

        # ── Conflict-aware fusion gate ──
        self.gate = ConflictGate(d_hidden, d_hidden, d_hidden)

        # ── Task heads ──
        # 1. Node classification (component role prediction)
        self.node_clf = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, num_node_classes),
        )

        # 2. Graph regression (circuit performance prediction)
        self.graph_reg = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_perf_metrics),
        )

        # 3. Graph classification (circuit type)
        self.graph_clf = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, num_circuit_types),
        )

    def forward(self, data: Data):
        """
        Returns:
            node_logits  : [N, num_node_classes]
            perf_pred    : [B, num_perf_metrics]
            type_logits  : [B, num_circuit_types]
            alpha        : [N, 1]   gate values (monitor for collapse)
            gate_reg     : scalar   regularization term
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ── Text stream ──
        h_text = self.text_enc(x)                     # [N, d_hidden]

        # ── Graph stream (message passing over circuit topology) ──
        h_graph = self.graph_enc(h_text, edge_index)  # [N, d_hidden]
        # Residual: preserve original text signal
        h_graph = h_graph + h_text

        # ── Conflict-aware fusion ──
        h_fused, alpha = self.gate(h_text, h_graph)   # [N, d_hidden], [N, 1]
        gate_reg = self.gate.gate_entropy_loss(alpha)

        # ── Node-level head ──
        node_logits = self.node_clf(h_fused)           # [N, num_node_classes]

        # ── Graph-level pooling ──
        h_graph_pooled = global_mean_pool(h_fused, batch)  # [B, d_hidden]

        # ── Graph-level heads ──
        perf_pred   = self.graph_reg(h_graph_pooled)   # [B, num_perf_metrics]
        type_logits = self.graph_clf(h_graph_pooled)   # [B, num_circuit_types]

        return node_logits, perf_pred, type_logits, alpha, gate_reg

    def compute_loss(
        self,
        data: Data,
        node_labels: torch.Tensor = None,
        perf_targets: torch.Tensor = None,
        type_labels: torch.Tensor = None,
        task_weights: dict = None,
    ) -> dict:
        """
        Multi-task loss computation.

        Args:
            node_labels  : [N]       component role labels (long)
            perf_targets : [B, M]    performance metrics (float, NaN = missing)
            type_labels  : [B]       circuit type labels (long)
            task_weights : e.g. {"node": 0.3, "perf": 0.5, "type": 0.2, "gate": 0.01}
        """
        w = task_weights or {"node": 0.3, "perf": 0.5, "type": 0.2}
        node_logits, perf_pred, type_logits, alpha, gate_reg = self(data)

        losses = {}
        total = gate_reg * self.gate_reg_weight
        losses["gate_reg"] = gate_reg.item()

        if node_labels is not None:
            mask = node_labels >= 0
            if mask.any():
                losses["node"] = F.cross_entropy(node_logits[mask], node_labels[mask])
                total = total + w.get("node", 0.3) * losses["node"]

        if perf_targets is not None:
            # Handle NaN metrics: only supervise available metrics
            valid = ~torch.isnan(perf_targets)
            if valid.any():
                losses["perf"] = F.mse_loss(perf_pred[valid], perf_targets[valid])
                total = total + w.get("perf", 0.5) * losses["perf"]

        if type_labels is not None:
            losses["type"] = F.cross_entropy(type_logits, type_labels)
            total = total + w.get("type", 0.2) * losses["type"]

        losses["total"] = total
        losses["alpha_mean"] = alpha.mean().item()
        losses["alpha_std"]  = alpha.std().item()
        return losses


# ─────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────

class CircuitCATTrainer:
    """
    Thin training wrapper with gate collapse monitoring.
    """

    def __init__(self, model: CircuitCAT, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.history = []

    def train_step(self, data: Data, **kwargs) -> dict:
        self.model.train()
        self.optimizer.zero_grad()
        losses = self.model.compute_loss(data, **kwargs)
        losses["total"].backward()
        # Gradient clipping — important to prevent gate collapse
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    @torch.no_grad()
    def eval_step(self, data: Data, **kwargs) -> dict:
        self.model.eval()
        losses = self.model.compute_loss(data, **kwargs)
        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()}

    def check_gate_collapse(self, alpha_mean: float, alpha_std: float, threshold: float = 0.05):
        """Warn if gate is collapsing (all α near 0 or 1, very low std)."""
        if alpha_std < threshold:
            collapsed_to = "text" if alpha_mean > 0.9 else "graph" if alpha_mean < 0.1 else "mid"
            print(f"  ⚠️  Gate collapse warning: α_mean={alpha_mean:.3f}, α_std={alpha_std:.4f}"
                  f"  → collapsed toward {collapsed_to}")
            return True
        return False


# ─────────────────────────────────────────────
# Quick sanity test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from netlist_parser import parse_netlist, circuit_graph_to_pyg
    from circuit_tag_dataset import BagOfWordsEncoder

    sample_netlist = """Two-Stage Op-Amp
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
"""
    encoder = BagOfWordsEncoder()
    cg = parse_netlist(sample_netlist)
    data = circuit_graph_to_pyg(cg, text_encoder=encoder)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    model = CircuitCAT(d_text_in=encoder.dim, d_hidden=64)
    trainer = CircuitCATTrainer(model)

    # Fake labels for test
    node_labels  = data.role
    perf_targets = torch.tensor([[60.2, 12.5e6, 58.3, 0.0, 0.0, 0.0, 0.0, 1.0]])

    print("=== Forward pass ===")
    node_logits, perf_pred, type_logits, alpha, gate_reg = model(data)
    print(f"  node_logits : {node_logits.shape}")
    print(f"  perf_pred   : {perf_pred.shape}")
    print(f"  type_logits : {type_logits.shape}")
    print(f"  alpha (mean/std): {alpha.mean():.3f} / {alpha.std():.4f}")

    print("\n=== Training step ===")
    losses = trainer.train_step(data, node_labels=node_labels, perf_targets=perf_targets)
    for k, v in losses.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    collapsed = trainer.check_gate_collapse(losses["alpha_mean"], losses["alpha_std"])
    if not collapsed:
        print("  ✓ Gate is healthy (not collapsed)")

    print("\n=== 10 training steps ===")
    for step in range(10):
        losses = trainer.train_step(data, node_labels=node_labels, perf_targets=perf_targets)
        if (step + 1) % 5 == 0:
            print(f"  step {step+1:3d} | total={losses['total']:.4f} "
                  f"α_mean={losses['alpha_mean']:.3f} α_std={losses['alpha_std']:.4f}")
    print("Done.")
