from mpi4py import MPI
import numpy as np

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Matrix dimensions (N x N)
    N = 1000  # Small size for demonstration; increase for benchmarking
    A = None
    B = None
    C = None

    # Root process (rank 0) initializes matrices
    if rank == 0:
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        C = np.zeros((N, N))
        print("Matrix A:\n", A)
        print("Matrix B:\n", B)

    # Broadcast matrix B to all processes
    B = comm.bcast(B, root=0)

    # Scatter rows of A to different processes
    local_rows = N // size
    local_A = np.zeros((local_rows, N))
    comm.Scatter(A, local_A, root=0)

    # Local computation (partial matrix multiplication)
    local_C = np.dot(local_A, B)

    # Gather results back to root
    comm.Gather(local_C, C, root=0)

    # Root prints the result
    if rank == 0:
        print("Result (A x B):\n", C)

if __name__ == "__main__":
    main()