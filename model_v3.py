import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax
from config_v3 import Config


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand(1, device=x.device) < keep).float()
        return x * mask / keep


class StructFeats(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, 32), nn.SiLU(),
            nn.Linear(32, output_dim), nn.LayerNorm(output_dim)
        )

    def forward(self, coords, edge_index, edge_dist):
        L = coords.shape[0]
        src, dst = edge_index
        device = coords.device
        ones = torch.ones_like(edge_dist)
        close = (edge_dist < 8.0).float()

        contact = torch.zeros(L, device=device)
        contact.scatter_add_(0, dst, close)
        contact = contact / (contact.max() + 1e-8)

        centroid = coords.mean(0, keepdim=True)
        depth = torch.norm(coords - centroid, dim=-1)
        depth = depth / (depth.max() + 1e-8)

        s_dist = torch.zeros(L, device=device)
        n_nb = torch.zeros(L, device=device)
        s_dist.scatter_add_(0, dst, edge_dist)
        n_nb.scatter_add_(0, dst, ones)
        avg_d = s_dist / (n_nb + 1e-8)
        avg_d = avg_d / (avg_d.max() + 1e-8)

        cc = torch.zeros(L, device=device)
        tc = torch.zeros(L, device=device)
        cc.scatter_add_(0, dst, close)
        tc.scatter_add_(0, dst, ones)
        ld = cc / (tc + 1e-8)

        sc = torch.zeros(L, 3, device=device)
        sq = torch.zeros(L, 3, device=device)
        sc.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), coords[src])
        sq.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), coords[src] ** 2)
        nc = n_nb.unsqueeze(-1).clamp(min=1)
        var = (sq / nc - (sc / nc) ** 2).clamp(min=0).sum(-1).sqrt()
        var = var / (var.max() + 1e-8)

        sp = torch.arange(L, device=device, dtype=torch.float32) / max(L - 1, 1)

        feats = torch.stack([contact, depth, avg_d, ld, var, sp], dim=-1)
        return self.proj(feats)


class ShellEdgeEmbedding(nn.Module):
    def __init__(self, out_dim=12, bounds=(3.0, 5.0, 8.0, 12.0)):
        super().__init__()
        self.bounds = bounds
        self.emb = nn.Embedding(4, out_dim)

    def forward(self, edge_dist):
        b0, b1, b2, b3 = self.bounds
        tag = torch.full_like(edge_dist, 3, dtype=torch.long)
        tag[(edge_dist >= b0) & (edge_dist < b1)] = 0
        tag[(edge_dist >= b1) & (edge_dist < b2)] = 1
        tag[(edge_dist >= b2) & (edge_dist < b3)] = 2
        return self.emb(tag)


def compute_gnm_corr(coords, edge_index, gamma=1.0, eps=1e-5):
    device = coords.device
    dtype = coords.dtype
    N = coords.size(0)
    src, dst = edge_index

    A = torch.zeros(N, N, device=device, dtype=dtype)
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    A.fill_diagonal_(0.0)

    D = torch.diag(A.sum(dim=-1))
    G = gamma * (D - A)
    G = G + eps * torch.eye(N, device=device, dtype=dtype)

    G_pinv = torch.linalg.pinv(G)

    d = torch.sqrt(torch.clamp(torch.diag(G_pinv), min=eps))
    corr = G_pinv / (d[:, None] * d[None, :] + eps)

    edge_corr = corr[src, dst]
    edge_corr = torch.nan_to_num(edge_corr, nan=0.0, posinf=0.0, neginf=0.0)
    return edge_corr


class EGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, drop_path=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(node_dim)
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        self.attn_gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, h, coords, edge_index, edge_attr):
        src, dst = edge_index
        N = h.shape[0]
        h_n = self.norm(h)
        rel = coords[dst] - coords[src]
        dsq = (rel ** 2).sum(-1, keepdim=True)
        inp = torch.cat([h_n[src], h_n[dst], edge_attr, dsq], dim=-1)

        msg_raw = self.message_net(inp)
        msg = msg_raw * self.attn_gate(msg_raw)

        agg = torch.zeros(N, msg.shape[1], device=h.device, dtype=msg.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        h = h + self.drop_path(self.update_net(torch.cat([h_n, agg], dim=-1)))

        cw = self.coord_mlp(msg)
        cd = torch.zeros_like(coords)
        cd.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), rel * cw)
        coords = coords + cd * 0.1
        return h, coords


class EGNNEncoder(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers=4, drop_path=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, hidden_dim, dpr[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats, coords, edge_index, edge_feats):
        h = self.input_proj(node_feats)
        e = self.edge_proj(edge_feats)
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index, e)
        return self.norm(h), coords


class ESMAdapter(nn.Module):
    def __init__(self, esm_dim, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out = self.proj(x)
        out = self.transformer(out)
        return self.norm(out).squeeze(0)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, query, context):
        L = query.shape[0]
        S = context.shape[0]
        nh = self.num_heads
        hd = self.head_dim

        q_n = self.norm_q(query)
        c_n = self.norm_k(context)

        Q = self.q_proj(q_n).view(L, nh, hd).permute(1, 0, 2)
        K = self.k_proj(c_n).view(S, nh, hd).permute(1, 0, 2)
        V = self.v_proj(c_n).view(S, nh, hd).permute(1, 0, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, V)
        out = out.permute(1, 0, 2).reshape(L, -1)

        query = query + self.drop_path(self.out_proj(out))
        query = query + self.drop_path(self.ffn(self.norm_ffn(query)))
        return query


class BiCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_rounds=2, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.rounds = nn.ModuleList()
        for r in range(num_rounds):
            dp = drop_path * r / max(num_rounds - 1, 1)
            self.rounds.append(nn.ModuleDict({
                'struct2seq': CrossAttentionBlock(dim, num_heads, dropout, dp),
                'seq2struct': CrossAttentionBlock(dim, num_heads, dropout, dp)
            }))

    def forward(self, h_seq, h_struct):
        for rnd in self.rounds:
            h_struct = rnd['struct2seq'](h_struct, h_seq)
            h_seq = rnd['seq2struct'](h_seq, h_struct)
        return h_seq, h_struct


class GTLayer(nn.Module):
    def __init__(self, dim, edge_dim, num_heads=8, ffn_dim=512, dropout=0.1, drop_path=0.0, gnm_gate_init=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(edge_dim, num_heads)

        # learnable gate for GNM bias
        self.gnm_gate = nn.Parameter(torch.tensor(float(gnm_gate_init)))

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout)
        )
        self.attn_drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, h, edge_index, edge_feats, attn_bias=None):
        L = h.shape[0]
        src, dst = edge_index
        nh = self.num_heads
        hd = self.head_dim

        h_n = self.norm1(h)

        Q = self.q_proj(h_n).view(L, nh, hd)
        K = self.k_proj(h_n).view(L, nh, hd)
        V = self.v_proj(h_n).view(L, nh, hd)

        q_dst = Q[dst]
        k_src = K[src]
        v_src = V[src]

        attn_scores = (q_dst * k_src).sum(-1) * self.scale
        e_bias = self.edge_bias(edge_feats)
        attn_scores = attn_scores + e_bias

        if attn_bias is not None:
            gate = torch.clamp(self.gnm_gate, 0.0, 0.2)
            if attn_bias.dim() == 1:
                b = (attn_bias - attn_bias.mean()) / (attn_bias.std() + 1e-6)
                attn_scores = attn_scores + gate * b.unsqueeze(-1)
            else:
                attn_scores = attn_scores + gate * attn_bias

        attn_weights = pyg_softmax(attn_scores, dst, num_nodes=L)
        attn_weights = self.attn_drop(attn_weights)

        weighted_v = v_src * attn_weights.unsqueeze(-1)
        agg = torch.zeros(L, nh, hd, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.view(-1, 1, 1).expand_as(weighted_v), weighted_v)
        agg = agg.view(L, -1)

        h = h + self.drop_path(self.out_proj(agg))
        h = h + self.drop_path(self.ffn(self.norm2(h)))
        return h


class GTEncoder(nn.Module):
    def __init__(self, dim, edge_dim, num_heads=8, ffn_dim=512, num_layers=4, dropout=0.1, drop_path=0.1, gnm_gate_init=0.05):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.layers = nn.ModuleList([
            GTLayer(dim, edge_dim, num_heads, ffn_dim, dropout, dpr[i], gnm_gate_init=gnm_gate_init)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, edge_index, edge_feats, attn_bias=None):
        intermediates = []
        for layer in self.layers:
            h = layer(h, edge_index, edge_feats, attn_bias=attn_bias)
            intermediates.append(h)
        return self.norm(h), intermediates


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim)
        )

    def forward(self, a, b):
        cat = torch.cat([a, b], dim=-1)
        g = self.gate(cat)
        return g * a + (1 - g) * b + self.proj(cat) * 0.1


class SwiGLUDecoder(nn.Module):
    def __init__(self, dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.norm(x)
        a, b = self.fc(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.drop(self.proj(x))
        return self.out(x).squeeze(-1)


class BFactorPredictor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        hdim = config.hidden_dim

        node_input_dim = config.node_feat_dim + config.pos_embed_dim + config.extra_struct_dim
        self.struct_feats = StructFeats(output_dim=config.extra_struct_dim)

        self.shell_edge_emb = ShellEdgeEmbedding(
            out_dim=getattr(config, "shell_embed_dim", 12),
            bounds=getattr(config, "shell_bounds", (3.0, 5.0, 8.0, 12.0))
        )

        edge_feat_dim_base = config.rbf_num + 3 + 1
        edge_feat_dim = edge_feat_dim_base + getattr(config, "shell_embed_dim", 12)

        self.egnn = EGNNEncoder(
            input_dim=node_input_dim,
            edge_dim=edge_feat_dim,
            hidden_dim=hdim,
            num_layers=config.egnn_layers,
            drop_path=config.path_dropout
        )

        self.esm_adapter = ESMAdapter(
            esm_dim=config.esm3_embed_dim,
            hidden_dim=hdim,
            num_layers=config.esm_adapter_layers,
            num_heads=config.esm_adapter_heads,
            dropout=config.dropout
        )

        self.cross_attn = BiCrossAttention(
            dim=hdim,
            num_heads=config.cross_attn_heads,
            num_rounds=config.cross_attn_rounds,
            dropout=config.attn_dropout,
            drop_path=config.path_dropout
        )

        self.fusion = GatedFusion(hdim)

        self.gt = GTEncoder(
            dim=hdim,
            edge_dim=hdim,
            num_heads=config.gt_heads,
            ffn_dim=config.gt_ffn_dim,
            num_layers=config.gt_layers,
            dropout=config.dropout,
            drop_path=config.path_dropout,
            gnm_gate_init=getattr(config, "gnm_gate_init", 0.05)
        )

        self.gt_edge_proj = nn.Linear(edge_feat_dim, hdim)

        dec_hidden = config.pred_hidden_dims[0] if len(config.pred_hidden_dims) > 0 else hdim
        self.decoder = SwiGLUDecoder(
            dim=hdim,
            hidden_dim=dec_hidden,
            dropout=config.dropout
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        coords = data['coords']
        node_feats = data['node_feats']
        edge_index = data['edge_index']
        edge_feats = data['edge_feats']
        edge_dist = data['edge_dist']

        esm_embed = data.get('esm_embedding', None)
        if esm_embed is None:
            esm_embed = data.get('esm3_embedding', None)
        if esm_embed is None:
            esm_embed = data.get('esm_embed', None)
        if esm_embed is None:
            esm_embed = data.get('esm3_embed', None)
        if esm_embed is None:
            raise KeyError(f"Missing ESM embedding. Available keys: {list(data.keys())}")

        if node_feats.dim() == 3 and node_feats.size(0) == 1:
            node_feats = node_feats.squeeze(0)
        if coords.dim() == 3 and coords.size(0) == 1:
            coords = coords.squeeze(0)
        if edge_index.dim() == 3 and edge_index.size(0) == 1:
            edge_index = edge_index.squeeze(0)
        if edge_feats.dim() == 3 and edge_feats.size(0) == 1:
            edge_feats = edge_feats.squeeze(0)
        if edge_dist.dim() == 2 and edge_dist.size(0) == 1:
            edge_dist = edge_dist.squeeze(0)
        if esm_embed.dim() == 3 and esm_embed.size(0) == 1:
            esm_embed = esm_embed.squeeze(0)

        shell_emb = self.shell_edge_emb(edge_dist)
        edge_feats = torch.cat([edge_feats, shell_emb], dim=-1)

        sf = self.struct_feats(coords, edge_index, edge_dist)
        node_input = torch.cat([node_feats, sf], dim=-1)

        expected_dim = self.egnn.input_proj.in_features
        actual_dim = node_input.size(-1)
        if actual_dim < expected_dim:
            pad = node_input.new_zeros(node_input.size(0), expected_dim - actual_dim)
            node_input = torch.cat([node_input, pad], dim=-1)
        elif actual_dim > expected_dim:
            node_input = node_input[:, :expected_dim]

        h_struct, coords_updated = self.egnn(node_input, coords, edge_index, edge_feats)
        h_seq = self.esm_adapter(esm_embed)
        h_seq, h_struct = self.cross_attn(h_seq, h_struct)
        h_fused = self.fusion(h_struct, h_seq)

        gnm_bias = None
        if getattr(self.config, "use_gnm_bias", True):
            gnm_bias = compute_gnm_corr(
                coords=coords_updated,
                edge_index=edge_index,
                gamma=getattr(self.config, "gnm_gamma", 1.0),
                eps=getattr(self.config, "gnm_eps", 1e-5)
            )

        edge_feats_gt = self.gt_edge_proj(edge_feats)
        h_final, _ = self.gt(h_fused, edge_index, edge_feats_gt, attn_bias=gnm_bias)

        pred = self.decoder(h_final)
        return pred

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
