#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h> // header for openmp

// Matrix parameters
#define SIZE 1024

// Fungsi membuat sembarang matriks berukuran besar
void fillMatrix(double **matrix)
{
    for (int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            matrix[i][j] = rand() % 100 + 1;
            //Random number between 1 and 100
        }
    }
}
// Fungsi perkalian matriks
void multiplyMatricesParalel(double **a, double **b, double **result)
{
    int i, j, k;
    /*start the parallelism*/
    #pragma omp parallel for private(i, j, k) shared(a, b, result)
    for(i = 0; i < SIZE; i++)
    {
        for(j = 0; j < SIZE; j++)
        {
            result[i][j] = 0;
            for(k = 0; k < SIZE; k++)
            {
                result[i][j] += a[j][k] * b[k][j];
            }
        }
    }
}
void multiplyMatricesParalel2(double **a, double **b, double **result)
{
    int i, j, k;
    /*start the parallelism*/
    for(i = 0; i < SIZE; i++)
    {
        #pragma omp parallel for private(j, k)
        for(j = 0; j < SIZE; j++)
        {
            result[i][j] = 0;
            for(k = 0; k < SIZE; k++)
            {
                result[i][j] += a[j][k] * b[k][j];
            }
        }
    }
}
void multiplyMatricesParalel3(double **a, double **b, double **result)
{
    int i, j, k;
    /*start the parallelism*/
    for(i = 0; i < SIZE; i++)
    {
        for(j = 0; j < SIZE; j++)
        {
            result[i][j] = 0;
            #pragma omp parallel for private(k) reduction(+:result[i][j])
            for(k = 0; k < SIZE; k++)
            {
                result[i][j] += a[j][k] * b[k][j];
            }
        }
    }
}
void multiplyMatricesSequential(double **a, double **b, double **result)
{
    int i, j, k;
    /*start the parallelism*/
    for(i = 0; i < SIZE; i++)
    {
        for(j = 0; j < SIZE; j++)
        {
            result[i][j] = 0;
            for(k = 0; k < SIZE; k++)
            {
                result[i][j] += a[j][k] * b[k][j];
            }
        }
    }
}
// Function to allocate memory for a matrix
double **allocateMatrix()
{
    double **matrix = (double **)malloc(SIZE *sizeof(double *) );
    if (matrix == NULL){
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < SIZE; i++)
    {
        matrix[i] = (double *)malloc(SIZE *sizeof(double));
        if (matrix == NULL){
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}
// Function to free memory of a matrix
void freeMatrix(double **matrix)
{
    for(int i = 0; i < SIZE; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}
int main()
{
    double **a, **b, **result, **result2, **result3, **result4;
    clock_t start, end;
    double cpu_time_used;

    srand(time(NULL));
    // Seed the random number generator

    // Allocate memory for the matrix

    a = allocateMatrix();
    b = allocateMatrix();
    result = allocateMatrix();
    result2 = allocateMatrix();
    result3 = allocateMatrix();
    result4 = allocateMatrix();

    // fill matrx with random numbers

    fillMatrix(a);
    fillMatrix(b);

    //start timer
    start = clock();
    //multiply matrixx a and b and store in the result matrix
    multiplyMatricesSequential(a, b, result);
    //end the timer
    end = clock();
    cpu_time_used = ((double (end - start)) / CLOCKS_PER_SEC);

    printf("Time needed for matrix multiplication S: %f seconds \n", cpu_time_used);
    //-----------------------------------------------------
    start = clock();
    //multiply matrixx a and b and store in the result matrix
    multiplyMatricesParalel(a, b, result2);
    //end the timer
    end = clock();
    cpu_time_used = ((double (end - start)) / CLOCKS_PER_SEC);

    printf("Time needed for matrix multiplication P1: %f seconds \n", cpu_time_used);
    //-----------------------------------------------------
    start = clock();
    //multiply matrixx a and b and store in the result matrix
    multiplyMatricesParalel(a, b, result3);
    //end the timer
    end = clock();
    cpu_time_used = ((double (end - start)) / CLOCKS_PER_SEC);

    printf("Time needed for matrix multiplication P2: %f seconds \n", cpu_time_used);
    //-----------------------------------------------------
    start = clock();
    //multiply matrixx a and b and store in the result matrix
    multiplyMatricesParalel(a, b, result4);
    //end the timer
    end = clock();
    cpu_time_used = ((double (end - start)) / CLOCKS_PER_SEC);

    printf("Time needed for matrix multiplication P3: %f seconds \n", cpu_time_used);

    //free the allocated memory

    freeMatrix(a);
    freeMatrix(b);
    freeMatrix(result);
    freeMatrix(result2);
    freeMatrix(result3);
    freeMatrix(result4);

    return 0;
}