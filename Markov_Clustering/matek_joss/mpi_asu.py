from mpi4py import MPI
import numpy as np
import pandas as pd
import networkx as nx
import time
import matplotlib.pyplot as plt

def normalize_columns(matrix):
    col_sums = matrix.sum(axis=0)
    return matrix / col_sums

def inflate(matrix, inflation_factor):
    matrix = np.power(matrix, inflation_factor)
    return normalize_columns(matrix)

def has_converged(matrix, prev_matrix, threshold=1e-5):
    return np.allclose(matrix, prev_matrix, atol=threshold)

def mpi_matrix_multiply(comm, A, B):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n = A.shape[0]

    rows_per_proc = n // size
    start = rank * rows_per_proc
    end = (rank + 1) * rows_per_proc if rank != size - 1 else n

    local_A = A[start:end, :]
    local_C = np.dot(local_A, B)

    # Gabungkan hasil ke proses 0
    if rank == 0:
        C = np.empty((n, n), dtype='d')
    else:
        C = None

    comm.Gather(local_C, C, root=0)

    # Broadcast hasil ke semua proses
    comm.Bcast(C, root=0)
    return C

def mcl_mpi(matrix, expansion=2, inflation=2, max_iter=100, tol=1e-7):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    matrix = normalize_columns(matrix)

    for i in range(max_iter):
        prev_matrix = matrix.copy()
        # Expansion step (paralel)
        matrix = mpi_matrix_multiply(comm, matrix, matrix)
        # Inflation step (lokal, tiap proses lakukan sama)
        matrix = inflate(matrix, inflation)

        if rank == 0 and has_converged(matrix, prev_matrix, tol):
            print(f"Converged at iteration {i}")
            break

    return matrix

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:.2f}" for val in row))

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
#=======================================================

    # Ukuran graph (harus dibagi rata oleh jumlah proses)
    N = 198
    
    if rank == 0:
        # Contoh adjacency matrix
        adj = np.array([
            [1,1,1,0,0,0,0,0],
            [1,1,1,0,0,0,0,0],
            [1,1,1,0,0,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,1,1,0,0,0],
            [0,0,0,0,0,1,1,1],
            [0,0,0,0,0,1,1,1],
            [0,0,0,0,0,1,1,1],
        ], dtype=float)
    else:
        adj = np.empty((N, N), dtype='d')
    # Broadcast matrix ke semua proses
    comm.Bcast(adj, root=0)
    
    '''
    df = pd.read_csv("F:\\Coding\\Komparawrrr\\Markov_Clustering\\matek_joss\\gugel_scholar.csv")

    G = nx.from_pandas_edgelist(df, source='Author', target='Co Author')

    adj_matek = nx.to_numpy_array(G, dtype=np.float32)

    # Broadcast matrix ke semua proses
    comm.Bcast(adj_matek, root=0)
    '''

#========================================================
    result = mcl_mpi(adj, expansion=2, inflation=2)
#========================================================
    if rank == 0:
        print("Hasil akhir matriks MCL:")
        print(np.round(result, 2))
