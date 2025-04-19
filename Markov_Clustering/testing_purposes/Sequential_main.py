import numpy as np
from Markov_Clustering.testing_purposes.graph import Graph
import time 


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

# Sequential Matrix Multiplication
def multiply(matrix_a, matrix_b):
    # Initialize the result matrix with zeros, size of result matrix
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    # Perform matrix multiplication
    for i in range(len(matrix_a)):

        for j in range(len(matrix_b[0])):

            for k in range(len(matrix_b)):

                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

# Power of Matrix
def powerMatrix(matrix, n):
    if n < 1:
        raise ValueError
        return None

    matrix1, matrix2 = matrix, matrix

    for i in range(n-1):
        result = multiply(matrix1, matrix2) # Sequential Part
        matrix1 = result

    return result

    # Note: Input of matrix is always square matrix, no power of zero

# Matrix Normalization
def Normalize(matrix):
    matrix2 = matrix

    for j in range(len(matrix2[0])):
        temp = 0
        for i in range(len(matrix2)):
            temp += matrix2[i][j]

        for i in range(len(matrix2)):
            matrix2[i][j] = matrix2[i][j] / temp
        
    return matrix2
    
# Inflation of Matrix
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

# Markov Clustering
def MarkovCluster(graph, e, r):
    matrix, indexmap = graph.to_adjacency_matrix()

    for i in range(len(matrix)):
        matrix[i][i] = 1

    matrix = Normalize(matrix)

    iterate = 20
    ptr = 0

    while ptr <= iterate:

        matrix = powerMatrix(matrix, e) # Exponent
        print(f"Adjacency Matrix Powered steps: {ptr}")
        print_matrix(matrix)

        matrix = Inflate(matrix, r)

        print(f"Adjacency Matrix Inflated steps: {ptr}")
        print_matrix(matrix)

        ptr += 1

    return matrix

# Print Matrix
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))
#==================================== Main Program ====================================

# Parameter
n_nodes = 1000  # Number of nodes in graph
n_edge = 1500   # Number of edge in graph

e = 2 # Power Parameter
r = 2 # Inflation Parameter


# 1. Input is an un-directed graph, power parameter e, and inflation parameter r.
graph = Graph(directed = False)

graph.generate_random_connected_graph(num_nodes = n_nodes, num_edges = n_edge)

# Add all nodes (1 to 12)
for i in range(1, 13):
    graph.add_node(i)

'''
# Add edges based on the adjacency list
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
'''

'''
# 2. Create the associated matrix
matrix, indexmap = graph.to_adjacency_matrix()

# 3. Add self loops to each node (optional)

for i in range(n_nodes):
    matrix[i][i] = 1


# 4. Normalize the Matrix

for j in range(n_nodes):
    temp = 0
    for i in range(n_nodes):
        temp += matrix[i][j]

    for i in range(n_nodes):
        matrix[i][j] = matrix[i][j] / temp


# 5. Expand by taking the e-th power of the matrix 
new_matrix = powerMatrix(matrix, n = e)

# 6. Inflate by taking inflation of the resulting matrix with parameter r
for j in range(n_nodes):
    sum = 0
    for i in range(n_nodes):
        new_matrix[i][j] = (new_matrix[i][j] ** r)
        sum += new_matrix[i][j]

    for i in range(n_nodes):
        new_matrix[i][j] = new_matrix[i][j] / sum

# 7. Repeat until steady state is reached

'''


start = time.time()
MarkovMatrix = MarkovCluster(graph, e, r)
end = time.time()

print("Adjacency Matrix of Last Iteration")
print_matrix(MarkovMatrix)
print(f"Waktu eksekusi: {end - start:.4f} detik")