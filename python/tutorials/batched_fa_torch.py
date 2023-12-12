import torch

torch.manual_seed(10)
N_CTX = 128
q = torch.randn((N_CTX, 64))
k = torch.randn((N_CTX, 64))
v = torch.randn((N_CTX, 64))

o1 = torch.matmul(torch.softmax(torch.matmul(q, k.T), dim=-1), v)

# Note: BLOCK_M = N_CTX
BLOCK_N = 32
n_warps = 2

q3d = q.unsqueeze(0).expand(n_warps, -1, -1)
m_i = torch.full((n_warps, N_CTX, 1), fill_value=float("-inf"))
l_i = torch.zeros((n_warps, N_CTX, 1))
acc = torch.zeros((n_warps, N_CTX, 64))
for i in range(0, N_CTX, BLOCK_N):
    ki = torch.stack(k[i:i + BLOCK_N].chunk(n_warps, dim=0))
    vi = torch.stack(v[i:i + BLOCK_N].chunk(n_warps, dim=0))
    qk = torch.matmul(q3d, ki.transpose(-1, -2))
    m_i_new = torch.maximum(m_i, qk.max(dim=-1, keepdim=True)[0])
    alpha = torch.exp(m_i - m_i_new)
    p = torch.exp(qk - m_i_new)
    acc *= alpha
    acc += torch.matmul(p, vi)
    l_i = l_i * alpha + torch.sum(p, dim=-1, keepdim=True)
    m_i = m_i_new

o2 = torch.zeros_like(o1)
m_i_final = torch.full((N_CTX, 1), fill_value=float("-inf"))
l_i_final = torch.zeros((N_CTX, 1))
for i in range(n_warps):
    msi = m_i[i]
    lsi = l_i[i]
    osi = acc[i]
    alphai = torch.exp(m_i_final - msi)
    l_i_final = l_i_final * alphai + lsi
    o2 = o2 * alphai + osi
    m_i_final = msi
o2 = o2 / l_i_final

print((o1 - o2).abs().max())
