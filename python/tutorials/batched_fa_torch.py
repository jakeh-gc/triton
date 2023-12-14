import torch

torch.manual_seed(10)
N_CTX = 128
DMODEL = 64
q = torch.randn((N_CTX, DMODEL))
k = torch.randn((N_CTX, DMODEL))
v = torch.randn((N_CTX, DMODEL))

o1 = torch.matmul(torch.softmax(torch.matmul(q, k.T), dim=-1), v)

# Note: BLOCK_M = N_CTX
BLOCK_N = 32
n_warps = 2
q = q * 1.44269504
q3d = q.unsqueeze(0).expand(n_warps, -1, -1)
m = torch.full((n_warps, N_CTX, 1), fill_value=float("-inf"))
l = torch.zeros((n_warps, N_CTX, 1))
acc = torch.zeros((n_warps, N_CTX, DMODEL))
for i in range(0, N_CTX, BLOCK_N):
    ki = torch.stack(k[i:i + BLOCK_N].chunk(n_warps, dim=0))
    vi = torch.stack(v[i:i + BLOCK_N].chunk(n_warps, dim=0))
    qk = torch.matmul(q3d, ki.transpose(-1, -2))
    m_new = torch.maximum(m, qk.max(dim=-1, keepdim=True)[0])
    alpha = torch.exp2(m - m_new)
    p = torch.exp2(qk - m_new)
    acc *= alpha
    acc += torch.matmul(p, vi)
    l = l * alpha + torch.sum(p, dim=-1, keepdim=True)
    m = m_new

# acc.shape = (n_warps, N_CTX, DMODEL)
# l.shape = (n_warps, N_CTX, 1)
# m.shape = (n_warps, N_CTX, 1)
o = torch.zeros((N_CTX, DMODEL))
l_prev = torch.zeros((N_CTX, 1))
exp_m = torch.exp2(m)
o = acc * exp_m
l_prev = l * exp_m
o = torch.sum(o, dim=0)
l_prev = torch.sum(l_prev, dim=0)
o = o / l_prev
# for i in range(n_warps):
#     mi = m[i]
#     li = l[i]
#     acci = acc[i]
#     exp_mi = torch.exp2(mi)
#     l_prev += (li * exp_mi)
#     o += (acci * exp_mi)
# l_prev_final = l_prev * torch.exp2(-m[n_warps-1])
# o = o * torch.exp2(-m[n_warps-1])
# o = o / l_prev_final

print((o1 - o).abs().max())
