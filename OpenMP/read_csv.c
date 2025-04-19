#include <stdio.h>
#include <stdlib.h>

#define MAX_NODES 5242  // Sesuaikan dengan jumlah maksimum node

int main() {
    FILE *file = fopen("F:/Coding/Komparawrrr/OpenMP/CA-GrQc.csv", "r");
    if (file == NULL) {
        printf("Tidak dapat membuka file.\n");
        return 1;
    }

    int adj_matrix[MAX_NODES][MAX_NODES] = {0};
    int from_node, to_node;

    // Lewati header
    fscanf(file, "%*[^\n]\n");

    // Baca edges dari file CSV dan isi adjacency matrix
    while (fscanf(file, "%d,%d", &from_node, &to_node) != EOF) {
        adj_matrix[from_node][to_node] = 1;
    }

    fclose(file);

    // Cetak adjacency matrix
    for (int i = 0; i < MAX_NODES; i++) {
        for (int j = 0; j < MAX_NODES; j++) {
            printf("%d ", adj_matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}
