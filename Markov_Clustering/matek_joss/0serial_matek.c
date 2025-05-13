#include <stdio.h>
#include <stdlib.h>

#define MAX_AUTHORS 1000  // Sesuaikan jika jumlah ID melebihi 1000

int main() {
    FILE *file = fopen("id_only.csv", "r");
    if (!file) {
        perror("Gagal membuka file");
        return 1;
    }

    int adjMatrix[MAX_AUTHORS][MAX_AUTHORS] = {0};
    int author_id, coauthor_id;
    char line[1024];

    // Lewati header
    fgets(line, sizeof(line), file);

    int max_id = 0;

    while (fgets(line, sizeof(line), file)) {
        // Baca dua ID dari baris CSV (indeks kolom ke-1 dan ke-2, abaikan kolom pertama)
        sscanf(line, "%*d,%d,%d", &author_id, &coauthor_id);

        // Tandai hubungan dua arah
        adjMatrix[author_id][coauthor_id] = 1;
        adjMatrix[coauthor_id][author_id] = 1;

        // Lacak ID terbesar (untuk mencetak matriks dengan ukuran tepat)
        if (author_id > max_id) max_id = author_id;
        if (coauthor_id > max_id) max_id = coauthor_id;
    }

    fclose(file);

    // Cetak adjacency matrix
    printf("Adjacency Matrix:\n");
    for (int i = 0; i <= max_id; i++) {
        for (int j = 0; j <= max_id; j++) {
            printf("%d ", adjMatrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}