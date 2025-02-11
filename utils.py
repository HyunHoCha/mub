import numpy as np
import torch
import random


def init_loss(T, N, K):
    # T.shape: (K, N, N)

    ones = torch.ones(N)
    mutual = 1 / N

    loss = 0

    # Unitary loss
    for k in range(K):
        Tk = T[k]
        Tk_norms = torch.norm(Tk, dim=1)
        loss += torch.sum((ones - Tk_norms) ** 2)

        for i in range(N - 1):
            for j in range(i + 1, N):
                loss += (abs(torch.dot(Tk[i], Tk[j].conj())) ** 2)
    
    # Unbiased loss
    for k in range(K - 1):
        for _k in range(k + 1, K):
            for i in range(N):
                for j in range(N):
                    loss += ((abs(torch.dot(T[k][i], T[_k][j].conj())) ** 2 - mutual) ** 2)
    
    return loss


def init_loss2(T, N, K):
    # T.shape: (K, N, N)

    ones = torch.ones(N)
    mutual = 1 / N

    loss = 0

    # Unitary loss
    for k in range(K):
        Tk = T[k]
        Tk_norms = torch.norm(Tk, dim=1)
        loss += torch.sum((ones - Tk_norms) ** 2)

        for i in range(N - 1):
            for j in range(i + 1, N):
                loss += (abs(torch.dot(Tk[i], Tk[j].conj())) ** 2)
    
    # Relax inner products to equality
    inner_list = []
    for k in range(K - 1):
        for _k in range(k + 1, K):
            for i in range(N):
                for j in range(N):
                    inner_list.append(abs(torch.dot(T[k][i], T[_k][j].conj())))
    num_pairs = len(inner_list)
    for idx1 in range(num_pairs - 1):
        for idx2 in range(idx1 + 1, num_pairs):
            loss += (inner_list[idx1] - inner_list[idx2]) ** 2
    
    return loss


def init_loss3(T, N, K, factor):
    # T.shape: (K, N, N)

    ones = torch.ones(N)
    mutual = factor / N

    loss = 0

    # Unitary loss
    for k in range(K):
        Tk = T[k]
        Tk_norms = torch.norm(Tk, dim=1)
        loss += torch.sum((ones - Tk_norms) ** 2)

        for i in range(N - 1):
            for j in range(i + 1, N):
                loss += (abs(torch.dot(Tk[i], Tk[j].conj())) ** 2)
    
    # Unbiased loss
    for k in range(K - 1):
        for _k in range(k + 1, K):
            for i in range(N):
                for j in range(N):
                    loss += ((abs(torch.dot(T[k][i], T[_k][j].conj())) ** 2 - mutual) ** 2)
    
    return loss


def random_bit(p):
    return 1 if random.random() < p else 0
