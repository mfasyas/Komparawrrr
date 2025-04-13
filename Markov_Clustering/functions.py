import numpy as np
import cupy as cp
from numba import cuda, float32
from numba import jit

# Function to Multiply Matrices Paralel
@cuda.jit
def fast_matmul(A, B, C, TPB = 16):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return
    
    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.

    tmp = 0.

    for i in range(bpg):

        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until computing finished
        cuda.syncthreads()

    C[x, y] = tmp

# Power Matrix
def powerMatrix(matrix, n):
    if n < 1:
        raise ValueError
        return None

    matrix1, matrix2 = matrix, matrix

    for i in range(n-1):
        result = fast_matmul(matrix1, matrix2) # Sequential Part
        matrix1 = result

    return result

    # Note: Input of matrix is always square matrix, no power of zero
#================================================================================================

# Kernel to compute column sums (temp_j for each column)
@cuda.jit
def compute_column_sums(matrix, col_sums):
    row, col = cuda.grid(2)
    rows, cols = matrix.shape

    if row < rows and col < cols:
        val = matrix[row, col]
        if val == 1:
            cuda.atomic.add(col_sums, col, 1)

# Kernel to normalize matrix elements by column sums
@cuda.jit
def normalize_matrix(matrix, col_sums, result):
    row, col = cuda.grid(2)
    rows, cols = matrix.shape

    if row < rows and col < cols:
        sum_col = col_sums[col]
        if sum_col != 0:
            result[row, col] = matrix[row, col] / sum_col
        else:
            result[row, col] = 0  # Avoid division by zero

def Normalize(matrix):
    matrix_np = np.array(matrix, dtype=np.float32)
    rows, cols = matrix_np.shape

    # Allocate device arrays
    d_matrix = cuda.to_device(matrix_np)
    d_col_sums = cuda.device_array(cols, dtype=np.float32)
    d_result = cuda.device_array_like(matrix_np)

    # Initialize column sums to zero
    cuda.to_device(np.zeros(cols, dtype=np.float32), to=d_col_sums)

    threadsperblock = (16, 16)
    blockspergrid_x = (rows + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (cols + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Step 1: Compute column sums
    compute_column_sums[blockspergrid, threadsperblock](d_matrix, d_col_sums)

    # Step 2: Normalize the matrix
    normalize_matrix[blockspergrid, threadsperblock](d_matrix, d_col_sums, d_result)

    return d_result.copy_to_host()

#================================================================================================

# Kernel to raise each element to the r-th power
@cuda.jit
def power_kernel(matrix, r, powered_matrix):
    row, col = cuda.grid(2)
    rows, cols = matrix.shape
    if row < rows and col < cols:
        powered_matrix[row, col] = matrix[row, col] ** r

# Kernel to compute column sums after raising to power
@cuda.jit
def column_sum_kernel(matrix, col_sums):
    row, col = cuda.grid(2)
    rows, cols = matrix.shape
    if row < rows and col < cols:
        cuda.atomic.add(col_sums, col, matrix[row, col])

# Kernel to normalize powered matrix using column sums
@cuda.jit
def normalize_kernel(powered_matrix, col_sums, result):
    row, col = cuda.grid(2)
    rows, cols = powered_matrix.shape
    if row < rows and col < cols:
        col_sum = col_sums[col]
        if col_sum != 0:
            result[row, col] = powered_matrix[row, col] / col_sum
        else:
            result[row, col] = 0  # Handle divide-by-zero safely

def Inflate(matrix, r):
    matrix_np = np.array(matrix, dtype=np.float32)
    rows, cols = matrix_np.shape

    d_matrix = cuda.to_device(matrix_np)
    d_powered = cuda.device_array_like(matrix_np)
    d_col_sums = cuda.device_array(cols, dtype=np.float32)
    d_result = cuda.device_array_like(matrix_np)

    threadsperblock = (16, 16)
    blockspergrid_x = (rows + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (cols + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Step 1: Raise entries to power r
    power_kernel[blockspergrid, threadsperblock](d_matrix, r, d_powered)

    # Step 2: Zero column sums
    cuda.to_device(np.zeros(cols, dtype=np.float32), to=d_col_sums)

    # Step 3: Compute new column sums
    column_sum_kernel[blockspergrid, threadsperblock](d_powered, d_col_sums)

    # Step 4: Normalize each element by its column sum
    normalize_kernel[blockspergrid, threadsperblock](d_powered, d_col_sums, d_result)

    return d_result.copy_to_host()
