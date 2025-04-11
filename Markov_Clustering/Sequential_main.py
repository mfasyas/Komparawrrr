import numpy as np
from graph import Graph


'''
This is the code to run the main program

    /* 1. Input is an un-directed graph, power parameter e, and inflation parameter r.*/
    Done

    /* 2. Create the associated matrix*/

    /* 3. Add self loops to each node (optional)*/

    /* 4. Normalize the Matrix*/

    /* 5. Expand by taking the e-th power of the matrix*/

    /* 6. Inflate by taking inflation of the resulting matrix with parameter r*/

    /* 7. Repeat steps 5 and 6 until a steady state is reached (convergence)*/

    /* (Optional) 8. Iterpret resulting matrix to discover clusters*/

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




#==================================== Main Program ====================================
