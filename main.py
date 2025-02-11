import numpy as np
import torch
from utils import *
import math

N = 7
K = 4
factor = 1
p = 0.1

size = (K, N, N)
ones = torch.ones(N)
mutual = factor / N

angles = torch.rand(size) * 2 * torch.pi
T = torch.exp(angles * 1j) * (1 / math.sqrt(N))

T.requires_grad_(True)

optimizer = torch.optim.SGD([T], lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100000, verbose=True)

min_loss = init_loss(T, N, K)
print("Initial loss: ", min_loss.item())
for epoch in range(100000):
    # loss = 0
    _loss = 0

    # Unitary loss
    for k in range(K):
        Tk = T[k]
        Tk_norms = torch.norm(Tk, dim=1)
        _loss += torch.sum((ones - Tk_norms) ** 2)

        for i in range(N - 1):
            for j in range(i + 1, N):
                _loss += (abs(torch.dot(Tk[i], Tk[j].conj())) ** 2)
    
    # Unbiased loss
    for k in range(K - 1):
        for _k in range(k + 1, K):
            for i in range(N):
                for j in range(N):
                    _loss += ((abs(torch.dot(T[k][i], T[_k][j].conj())) ** 2 - mutual) ** 2)

    print("Epoch: ", epoch, " | Loss: ", _loss.item())

    if _loss < min_loss:
        torch.save(T, 'tensor_N7K4.pt')
        min_loss = _loss

    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    scheduler.step(_loss)
