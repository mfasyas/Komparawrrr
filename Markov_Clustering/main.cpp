// main.cpp
#include <iostream>
#include "functions.h"

int main() {
    const int rowsA = 2, colsA = 3;
    const int rowsB = 3, colsB = 2;
    int A[rowsA * colsA] = {1, 2, 3, 4, 5, 6};
    int B[rowsB * colsB] = {7, 8, 9, 10, 11, 12};
    int C[rowsA * colsB] = {};

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::cout << "Result matrix:\n";
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            std::cout << C[i * colsB + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
