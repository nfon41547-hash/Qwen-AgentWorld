"""
Nirnaya-0.4B-0.2a Typed Decision Injection Layer
Integrated with Qwen-AgentWorld & Overlaid Optical-Fiber MoEFX

Architecture:
- Typed Decision Gating: Inspired by Kshitij Thakkar's Nirnaya-0.4B-0.2a (Qwen3Moe Typed Decision Engine).
- Adapts 2048-dim hidden states of AgentWorld into 512-dim Typed Decision latent space.
- Projects decisions across 16 typed decision experts (Top-2 per token routing).
- Emits structured decision logits: [Tool Call, Code Edit, Terminal Exec, Final Patch, Reflection, Direct Response].
- Merges typed decision residual directly into the main trunk via zero-initialized cross-attention projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

class NirnayaTypedDecisionLayer(nn.Module):
    """
    Nirnaya-0.4B Typed Decision Neural Adapter:
    Specialized for discrete, type-safe decision making (Action Gating & Sub-Agent Dispatch).
    Features:
    - Hidden dimension bridge: 2048 (AgentWorld) <-> 512 (Nirnaya Decision Latent)
    - 16 Typed Decision Experts (intermediate size: 768) with Top-2 routing
    - Typed Decision Head: 6 discrete decision classes + Soft Probability Calibration
    """
    def __init__(self, agent_hidden_dim=2048, nirnaya_dim=512, moe_inter_dim=768, num_experts=16, top_k=2):
        super().__init__()
        self.agent_hidden_dim = agent_hidden_dim
        self.nirnaya_dim = nirnaya_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 1. Down-projection into Nirnaya decision manifold
        self.down_proj = nn.Linear(agent_hidden_dim, nirnaya_dim, bias=False)
        self.norm = nn.RMSNorm(nirnaya_dim, eps=1e-6)
        
        # 2. Decision Router (Top-2 over 16 typed experts)
        self.decision_gate = nn.Linear(nirnaya_dim, num_experts, bias=False)
        
        # 3. 16 Typed Decision Experts (Compact sub-networks for typed decisions)
        self.expert_gate = nn.Parameter(torch.randn(num_experts, nirnaya_dim, moe_inter_dim) * 0.02)
        self.expert_up = nn.Parameter(torch.randn(num_experts, nirnaya_dim, moe_inter_dim) * 0.02)
        self.expert_down = nn.Parameter(torch.randn(num_experts, moe_inter_dim, nirnaya_dim) * 0.02)
        
        # 4. Typed Decision Classification Head (6 Core SWE/Agent Action Types)
        # Classes: [0: READ_NAV, 1: REPRO_TEST, 2: SURGICAL_EDIT, 3: VERIFY_PYTEST, 4: REFLECT_PIVOT, 5: SUBMIT_PATCH]
        self.typed_decision_head = nn.Linear(nirnaya_dim, 6, bias=True)
        
        # 5. Up-projection back to main AgentWorld trunk with zero-init gate
        self.up_proj = nn.Linear(nirnaya_dim, agent_hidden_dim, bias=False)
        self.alpha_residual = nn.Parameter(torch.zeros(1))  # Identity at init, plastic during forward

    def forward(self, x, return_decision_logits=False):
        """
        x: [batch, seq_len, agent_hidden_dim] (e.g. 2048)
        returns:
            fused_hidden: [batch, seq_len, agent_hidden_dim]
            decision_logits: [batch, seq_len, 6] (optional)
        """
        b, s, d = x.shape
        
        # Project into Nirnaya 512-dim decision manifold
        h = self.norm(self.down_proj(x))  # [b, s, 512]
        
        # Calculate Router Gating
        router_logits = self.decision_gate(h)  # [b, s, 16]
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        # Flatten for efficient expert gathering
        h_flat = h.view(b * s, self.nirnaya_dim)  # [N, 512]
        topk_weights_flat = topk_weights.view(b * s, self.top_k)  # [N, 2]
        topk_indices_flat = topk_indices.view(b * s, self.top_k)  # [N, 2]
        
        expert_out = torch.zeros_like(h_flat)
        for k in range(self.top_k):
            w = topk_weights_flat[:, k].unsqueeze(-1)  # [N, 1]
            idx = topk_indices_flat[:, k]  # [N]
            
            # Weighted expert forward computation
            # Using SwiGLU: SiLU(x @ W_gate) * (x @ W_up) @ W_down
            g = torch.bmm(h_flat.unsqueeze(1), self.expert_gate[idx]).squeeze(1)
            u = torch.bmm(h_flat.unsqueeze(1), self.expert_up[idx]).squeeze(1)
            act = F.silu(g) * u
            d_out = torch.bmm(act.unsqueeze(1), self.expert_down[idx]).squeeze(1)
            expert_out += w * d_out
            
        expert_out = expert_out.view(b, s, self.nirnaya_dim)
        
        # Extract typed decision logits (Decision Confidence)
        decision_logits = self.typed_decision_head(expert_out)
        
        # Inject back into the main trunk via residual bridge
        out = x + self.alpha_residual * self.up_proj(expert_out)
        
        if return_decision_logits:
            return out, decision_logits
        return out

def export_nirnaya_decision_adapter(output_path=r"C:\Models\Qwen-AgentWorld-35B-A3B-INT4-24SP\nirnaya_decision_layer.safetensors"):
    print("[Nirnaya-0.4B] Initializing Typed Decision Adapter Layer...")
    layer = NirnayaTypedDecisionLayer()
    tensors = {k: v.contiguous() for k, v in layer.state_dict().items()}
    save_file(tensors, output_path)
    print(f"[Nirnaya-0.4B] Exported adapter to {output_path}")

if __name__ == "__main__":
    export_nirnaya_decision_adapter()
