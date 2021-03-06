import numpy as np
import networkx as nx
import math

from scipy import sparse

def compute_dist(X):
    # computes all (squared) pairwise Euclidean distances between each data point in X
    # D_ij = <x_i - x_j, x_i - x_j>
    G = np.matmul(X, X.T)
    D = np.reshape(np.diag(G), (1, -1)) + np.reshape(np.diag(G), (-1, 1)) - 2 * G
    return D

def compute_kernel(D, eps, d):
    # computes kernel for approximating GL
    # D is matrix of pairwise distances
    K = np.exp(-D/eps) * np.power(eps, -d/2)
    return K

def compute_eigen(X, eps, K, d = 2,eps_quantile=0.5):
    # X is n x d matrix of data points
    n = X.shape[0]
    dists = compute_dist(X)
    if eps == "auto":
        triu_dists = np.triu(dists)
        eps = np.quantile(triu_dists[np.nonzero(triu_dists)], eps_quantile)
    W = compute_kernel(dists, eps, d)
    D = np.diag(np.sum(W, axis=1, keepdims=False))
    L = sparse.csr_matrix(D - W)
    S, U = sparse.linalg.eigsh(L, k = K, which='SM')
    S = np.reshape(S.real, (1, -1))/(eps * n)
    S[0,0] = 0 # manually enforce this
    # normalize eigenvectors in usual l2 norm
    U = np.divide(U.real, np.linalg.norm(U.real, axis=0, keepdims=True))
    return S, U, eps

def compute_wavelet_filter(eigenvec, eigenval, j):
    H = np.einsum('ik,jk->ij', eigenvec * h(eigenval, j), eigenvec)
    return H

def g(lam):
    return np.exp(-lam)

def h(lam,j):
    # lam is a numpy array
    # returns a numpy array
    return g(lam)**(2**(j-1)) - g(lam)**(2**j)


def calculate_wavelet(eigenval,eigenvec,J, eps):
    dilation = np.arange(1,J+1).tolist()
    wavelet = []
    N = eigenvec.shape[0]
    wavelet.append(np.identity(N) - np.einsum('ik,jk->ij',eigenvec * g(eigenval), eigenvec))
    for dil in dilation:
        wavelet.append(compute_wavelet_filter(eigenvec, eigenval, dil))
    return wavelet, np.einsum('ik,jk->ij', eigenvec * g(eigenval*2**J), eigenvec)

def weighted_wavelet_transform(wavelet, f, N):
    Wf = [(1/N) * np.matmul(psi, f) for psi in wavelet]
    return Wf

def zero_order_feature(Aj, f, N, norm_list):
    if norm_list == "none":
        F0 = (1/N) * np.matmul(Aj, f).reshape(-1, 1)
    else:
        this_F0 = np.abs(f).reshape(-1, 1)
        F0 = np.sum(np.power(this_F0, norm_list[0]),axis=0).reshape(-1, 1)
        for i in range(1, len(norm_list)):
            F0 = np.vstack((F0, np.sum(np.power(this_F0, norm_list[i]), axis=0).reshape(-1, 1)))
    return F0

def first_order_feature(psi, Wf, Aj, N, norm_list):
    F1 = [(1/N) * np.matmul(Aj, np.abs(ele)) for ele in Wf]
    if norm_list == "none":
        F1 = [(1/N) * np.matmul(Aj, np.abs(ele)) for ele in Wf]
    else:
        this_F1 = np.hstack([(1/N) * np.abs(ele) for ele in Wf])
        F1 = np.sum(np.power(this_F1, norm_list[0]),axis=0).reshape(-1, 1)
        for i in range(1, len(norm_list)):
            F1 = np.vstack((F1, np.sum(np.power(this_F1, norm_list[i]),axis=0).reshape(-1, 1)))
    return np.reshape(F1, (-1, 1))

def selected_second_order_feature(psi,Wf,Aj, N, norm_list):
    #only takes j2 > j1
    temp = np.abs(Wf[0:1])
    F2 = (1/N) * np.einsum('ij,aj->ai', psi[1], temp)
    for i in range(2,len(psi)):
        temp = np.abs(Wf[0:i])
        F2 = np.concatenate((F2,(1/N) * np.einsum('ij,aj ->ai',psi[i],temp)),0)
    F2 = np.abs(F2)
    if norm_list == "none":
        F2 = np.reshape(F2, (-1, 1))
    else:
        this_F2 = F2
        F2 = np.sum(np.power(this_F2, norm_list[0]), axis = 0).reshape(-1, 1)
        for i in range(1, len(norm_list)):
            F2 = np.vstack((F2, np.sum(np.power(this_F2, norm_list[i]), axis=0).reshape(-1, 1)))
    return F2.reshape(-1,1)


def generate_feature(psi,Wf,Aj,f, N, norm="none"):
    #with zero order, first order and second order features
    F0 = zero_order_feature(Aj, f, N, norm)
    F1 = first_order_feature(psi,Wf,Aj, N, norm)
    F2 = selected_second_order_feature(psi,Wf,Aj, N, norm)
    F = np.concatenate((F0,F1),axis=0)
    F = np.concatenate((F,F2),axis=0)
    return F

def compute_all_features(eigenval, eigenvec, training_signal, test_signal, training_Y, test_Y, eps, N, norm_list, J):
    training_feature = []
    N_train = len(training_signal)
    for i in range(N_train):
        if i % 1000 == 0:
            print(i)
        psi,Aj = calculate_wavelet(eigenval,eigenvec,J, eps)
        Wf = weighted_wavelet_transform(psi,training_signal[i], N)
        these_features = generate_feature(psi, Wf, Aj, training_signal[i], N, norm_list)
        training_feature.append(np.concatenate(these_features, axis=0))
    test_feature = []
    N_test = len(test_signal)
    for i in range(N_test):
        if i % 1000 == 0:
            print(i)
        psi,Aj = calculate_wavelet(eigenval,eigenvec,J, eps)
        Wf = weighted_wavelet_transform(psi,test_signal[i], N)
        these_features = generate_feature(psi,Wf,Aj,test_signal[i],N, norm_list)
        test_feature.append(np.concatenate(these_features,axis=0))
    return training_feature, test_feature
