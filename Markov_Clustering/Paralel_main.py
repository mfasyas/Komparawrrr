import numpy as np
from graph import Graph
from numba import cuda
import cupy as cp


'''
This is the code to run the main program

    The algorithm given by:

    /* 1. Input is an un-directed graph, power parameter e, and inflation parameter r. */
    Done by Defining Class Object Graph and adding power parameter and inflation as input

    /* 2. Create the associated matrix */

    /* 3. Add self loops to each node (optional)*/

    /* 4. Normalize the Matrix */

    /* 5. Expand by taking the e-th power of the matrix */

    /* 6. Inflate by taking inflation of the resulting matrix with parameter r */

    /* 7. Repeat steps 5 and 6 until a steady state is reached (convergence) */

    /* (Optional) 8. Iterpret resulting matrix to discover clusters */

'''

# Functions
def Normalize(matrix):
    matrix2 = matrix

    for j in range(len(matrix2[0])):
        temp = 0
        for i in range(len(matrix2)):
            temp += matrix2[i][j]

        for i in range(len(matrix2)):
            matrix2[i][j] = matrix2[i][j] / temp
        
    return matrix2

def Inflate(matrix, r):
    matrix2 = matrix

    for j in range(len(matrix2[0])):
        sum = 0
        for i in range(len(matrix2[0])):
            matrix2[i][j] = matrix2[i][j] ** r
            sum += matrix2[i][j]

        for i in range(len(matrix2[0])):
            matrix2[i][j] = matrix2[i][j] / sum

    return matrix2

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)  
    if i < C.shape[0] and j < C.shape[1]:   # grid can extend beyond C
        tmp = 0.                            
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]        # multiply elements in row i of A and column j of B and add to temp
        C[i, j] = tmp

def powerMatrix(matrix, n, bpg, tpb):
    if n < 1:
        raise ValueError
        return None

    matrix1, matrix2 = matrix, matrix
    matrix3 = cp.zeros((len(matrix[0]), len(matrix[1])), dtype=np.float64)  

    for i in range(n-1):
        matmul[bpg, tpb](matrix1, matrix2, matrix3) # Sequential Part
        matrix1 = matrix3

    return matrix1

    # Note: Input of matrix is always square matrix, no power of zero


# Print Matrix
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))

def MarkovCluster(graph, e, r):
    matrix, indexmap = graph.to_adjacency_matrix()

    for i in range(len(matrix)):
        matrix[i][i] = 1

    matrix = Normalize(matrix)

    matrix = cp.asarray(matrix)

    iterate = 20
    ptr = 0

    print(cuda.detect())
    threadsperblock = (16, 16)  # each block will contain 16x16 threads, typically 128 - 512 threads/block
    blockspergrid_x = int(np.ceil(matrix.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(matrix.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)  # we calculate the gridsize (number of blocks) from array
    print(blockspergrid)
    print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")

    while ptr <= iterate:

        matrix = powerMatrix(matrix, e, bpg=blockspergrid, tpb=threadsperblock)
        print(f"Adjacency Matrix Powered steps: {ptr}")
        print_matrix(matrix)

        matrix = Inflate(matrix, r)

        print(f"Adjacency Matrix Inflated steps: {ptr}")
        print_matrix(matrix)

        ptr += 1

    return matrix


# Parameter
n_nodes = 5  # Number of nodes in graph
n_edge = 6   # Number of edge in graph

e = 2 # Power Parameter
r = 2 # Inflation Parameter

# 1. Input is an un-directed graph, power parameter e, and inflation parameter r.

graph = Graph(directed = False)

# graph.generate_random_connected_graph(num_nodes = n_nodes, num_edges = n_edge)
# Generate random connected graph based on parameter

g = Graph(directed=False)

graph = Graph(directed=False)

# Add all nodes (1 to 12)
for i in range(1, 13):
    graph.add_node(i)

# Add edges based on the adjacency list can be done manually
graph.add_edge(1, 1)
graph.add_edge(1, 2)
graph.add_edge(1, 6)
graph.add_edge(1, 7)
graph.add_edge(1, 10)

graph.add_edge(2, 2)
graph.add_edge(2, 3)
graph.add_edge(2, 5)

graph.add_edge(3, 3)
graph.add_edge(3, 4)
graph.add_edge(3, 5)

graph.add_edge(4, 4)
graph.add_edge(4, 8)
graph.add_edge(4, 9)
graph.add_edge(4, 11)

graph.add_edge(5, 5)
graph.add_edge(5, 7)
graph.add_edge(5, 8)

graph.add_edge(6, 6)
graph.add_edge(6, 10)

graph.add_edge(7, 7)
graph.add_edge(7, 10)

graph.add_edge(8, 8)
graph.add_edge(8, 9)
graph.add_edge(8, 11)

graph.add_edge(9, 9)
graph.add_edge(9, 11)
graph.add_edge(9, 12)

graph.add_edge(10, 10)

graph.add_edge(11, 11)
graph.add_edge(11, 12)

graph.add_edge(12, 12)





#==================================== Main Program ====================================

MarkovMatrix = MarkovCluster(graph, e, r)

print("Adjacency Matrix of Last Iteration")
print_matrix(MarkovMatrix)