import numpy as np
from graph import Graph
from numba import cuda
import cupy as cp
import time

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
    matrix = cp.power(matrix, r)
    col_sums = cp.sum(matrix, axis=0)
    matrix = matrix / col_sums
    return matrix

@cuda.jit
def matmul(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

def powerMatrix(matrix, n, bpg, tpb):
    if n < 1:
        raise ValueError("Pangkat matrix harus >= 1")

    matrix1 = matrix
    matrix2 = cp.zeros_like(matrix)

    for i in range(n - 1):
        matmul[bpg, tpb](matrix1, matrix, matrix2)
        cuda.synchronize()  # penting untuk sinkronisasi kernel
        matrix1 = matrix2.copy()

    return matrix1

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:.2f}" for val in row))

def MarkovCluster(graph, e, r, iter):
    matrix, indexmap = graph.to_adjacency_matrix()

    # Add self-loops
    for i in range(len(matrix)):
        matrix[i][i] = 1

    matrix = Normalize(matrix)

    matrix = cp.asarray(matrix)

    # Setup CUDA kernel execution config
    threadsperblock = (32, 32)
    blockspergrid_x = int(np.ceil(matrix.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(matrix.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    iterate = iter
    tol = 1e-5
    ptr = 0
    converged = False

    while ptr < iterate and not converged:
        prev_matrix = matrix.copy()

        matrix = powerMatrix(matrix, e, bpg=blockspergrid, tpb=threadsperblock)
        matrix = Inflate(matrix, r)

        # Cek konvergensi
        diff = cp.abs(matrix - prev_matrix)
        max_diff = cp.max(diff)
        if max_diff < tol:
            converged = True

        ptr += 1

    return matrix

# =========================== Main Program ===========================

n_nodes = 1000
n_edge = 1200
e = 2
r = 2

graph = Graph(directed=False)
graph.generate_random_connected_graph(num_nodes=n_nodes, num_edges=n_edge)

start = time.time()
MarkovMatrix = MarkovCluster(graph, e, r, iter = 10)
end = time.time()

print("Adjacency Matrix of Last Iteration:")
# Uncomment to print
print_matrix(cp.asnumpy(MarkovMatrix))

print(f"Ukuran matriks: {n_nodes} x {n_nodes}, total edge: {n_edge}")
print(f"Parameter: e = {e}, r = {r}")
print(f"Waktu eksekusi: {end - start:.4f} detik")
