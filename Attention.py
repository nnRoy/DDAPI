import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, q_dim, v_dim, k_dim, dropout = 0. ):
        super(Attention, self).__init__()
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.k_dim = k_dim

        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(q_dim, q_dim)
        self.w_k = nn.Linear(k_dim, q_dim)
        self.w_V = nn.Linear(q_dim, 1)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param.data.uniform_(-0.1, 0.1)
            if 'bias' in name:
                nn.init.constant_(param, 0.)

    def forward(self, Q, K, V, key_pad_mask):
        batch_size, q_len, _ = Q.size()
        k_len = K.size(1)

        mapped_K = self.w_k(K)
        mapped_Q = self.w_q(Q)
        tiled_Q = mapped_Q.unsqueeze(2).repeat(1, 1, k_len, 1)
        tiled_K = mapped_K.unsqueeze(1)

        fc1 = torch.tanh(tiled_K + tiled_Q)
        attn_scores = self.w_V(fc1).squeeze(-1)

        attn_scores = attn_scores.masked_fill_(key_pad_mask.unsqueeze(1), -1e12)

        attn_weights = F.softmax(attn_scores.view(-1, k_len), dim=1).view(batch_size, -1,
                                                                          k_len)  # (batch x q_len x k_len)
        attn_weights = self.dropout(attn_weights)
        attn_ctx = torch.bmm(attn_weights, V)

        combined_attn_ctx = None

        return attn_ctx, attn_weights, combined_attn_ctx

