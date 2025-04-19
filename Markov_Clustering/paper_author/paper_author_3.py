import numpy as np
from mpi4py import MPI
import pandas as pd
import networkx as nx
import time
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ================================= FUnctions ====================================

def Normalize(matrix):
    matrix2 = matrix.copy()

    for j in range(len(matrix2[0])):
        temp = 0
        for i in range(len(matrix2)):
            temp += matrix2[i][j]
        for i in range(len(matrix2)):
            matrix2[i][j] = matrix2[i][j] / temp

    return matrix2

def Inflate(matrix, r):
    matrix = np.power(matrix, r)
    col_sums = np.sum(matrix, axis=0)
    matrix = matrix / col_sums
    return matrix

def matmul(A, B):
    """Multiply two matrices using MPI.
    
    Args:
        A: 2D numpy array (N x M)
        B: 2D numpy array (M x P)
        comm: MPI communicator
    
    Returns:
        C: Result matrix (N x P) on rank 0, None on other ranks.
    """
    
    N, M = A.shape
    M, P = B.shape
    
    # --- Step 1: Broadcast matrix B to all processes ---
    B = comm.bcast(B, root=0)
    
    # --- Step 2: Scatter rows of A across processes ---
    rows_per_rank = N // size
    local_A = np.zeros((rows_per_rank, M))
    comm.Scatter(A, local_A, root=0)
    
    # --- Step 3: Local computation ---
    local_C = np.dot(local_A, B)
    
    # --- Step 4: Gather results back to rank 0 ---
    if rank == 0:
        C = np.zeros((N, P))
    else:
        C = None
    comm.Gather(local_C, C, root=0)
    
    return C  # Only rank 0 gets full result

def powerMatrix(matrix, n):
    if n < 1:
        raise ValueError("Pangkat matrix harus >= 1")

    for i in range(n - 1):
        matrix = matmul(matrix, matrix)

    return matrix

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:.2f}" for val in row))

def MarkovCluster(matrix_a, e, r, iter):
    matrix = matrix_a

    # add self loops
    for i in range(len(matrix)):
        matrix[i][i] = 1

    matrix = Normalize(matrix)

    print("Normalized Matrix:\n", matrix)

    iterate = iter
    tol = 1e-7
    ptr = 0
    converged = False

    while ptr < iterate and not converged:
        prev_matrix = matrix.copy()

        matrix = powerMatrix(matrix, e)

        matrix = Inflate(matrix, r)

        # Cek Konvergensi
        diff = np.abs(matrix - prev_matrix)
        max_diff = np.max(diff)
        if max_diff < tol:
            converged = True

        ptr += 1

    return matrix

def get_clusters_from_square_matrix(mcl_matrix):
    """
    mcl_matrix: square numpy array (n_nodes x n_nodes), hasil dari MCL
                dengan 1s menunjukkan koneksi antar node dalam cluster
    return: list of clusters, each cluster is a list of node indices
    """
    # Buat graph dari adjacency matrix
    G = nx.from_numpy_array(mcl_matrix)

    # Ambil connected components (cluster)
    clusters = [list(component) for component in nx.connected_components(G)]

    return clusters

# ================================ Main Program =================================


df = pd.read_csv("CA-GrQc.csv")

G = nx.from_pandas_edgelist(df, source='FromNodeId', target='ToNodeId')

adj_lastfm = nx.to_numpy_array(G, dtype=np.float32)

print("Numpy Adjacency Matrix:\n", adj_lastfm)

e_1 = 2
r_1 = 2
start = time.time()
MarkovMatrix = MarkovCluster(matrix_a=adj_lastfm, e = e_1, r = r_1, iter = 10)
end = time.time()

clusters = get_clusters_from_square_matrix(MarkovMatrix)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")
