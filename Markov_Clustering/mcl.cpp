#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

const int N = 1000;           // Ukuran graf 100x100
const int MAX_ITER = 50;     // Maksimum iterasi
const double EPSILON = 1e-10; // Kriteria konvergensi
const double INFLATION = 2.0; // Parameter inflasi
const double DENSITY = 0.3;  // Kepadatan graf (3% edge terisi)

// Fungsi untuk membuat matriks
vector<vector<double>> create_matrix(int size) {
    return vector<vector<double>>(size, vector<double>(size, 0.0));
}

// Fungsi untuk menyalin matriks
void copy_matrix(const vector<vector<double>>& src, vector<vector<double>>& dst) {
    for (size_t i = 0; i < src.size(); ++i) {
        for (size_t j = 0; j < src[i].size(); ++j) {
            dst[i][j] = src[i][j];
        }
    }
}

// Fungsi untuk normalisasi matriks (kolom)
void normalize(vector<vector<double>>& m) {
    for (size_t j = 0; j < m[0].size(); ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < m.size(); ++i) {
            sum += m[i][j];
        }
        if (sum > 0) {
            for (size_t i = 0; i < m.size(); ++i) {
                m[i][j] /= sum;
            }
        }
    }
}

// Fungsi untuk perkalian matriks (ekspansi)
void matrix_multiply(const vector<vector<double>>& a, 
                    const vector<vector<double>>& b,
                    vector<vector<double>>& result) {
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b[0].size(); ++j) {
            result[i][j] = 0.0;
            for (size_t k = 0; k < a[0].size(); ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

// Fungsi untuk inflasi
void apply_inflation(vector<vector<double>>& m, double inflation) {
    for (auto& row : m) {
        for (auto& val : row) {
            val = pow(val, inflation);
        }
    }
}

// Fungsi untuk mengecek konvergensi
bool has_converged(const vector<vector<double>>& a, 
                  const vector<vector<double>>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            if (fabs(a[i][j] - b[i][j]) > EPSILON) {
                return false;
            }
        }
    }
    return true;
}

// Fungsi untuk membangun graf acak
void build_random_graph(vector<vector<double>>& graph, double density) {
    srand(time(NULL));
    int edges = static_cast<int>(graph.size() * graph[0].size() * density);
    
    for (int e = 0; e < edges; ++e) {
        int i = rand() % graph.size();
        int j = rand() % graph[0].size();
        graph[i][j] = 1.0 + (rand() % 10) / 10.0; // Bobot acak 1.0-2.0
    }
}

// Algoritma MCL utama
void run_mcl(vector<vector<double>>& graph, int max_iter, double inflation) {
    vector<vector<double>> current = create_matrix(graph.size());
    vector<vector<double>> next = create_matrix(graph.size());
    
    copy_matrix(graph, current);
    normalize(current);
    
    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        matrix_multiply(current, current, next);   // Ekspansi
        apply_inflation(next, inflation);          // Inflasi
        normalize(next);                           // Normalisasi
        
        if (has_converged(current, next)) break;
        
        swap(current, next);  // Tukar matriks untuk iterasi berikutnya
    }
    
    cout << "MCL completed in " << iter << " iterations" << endl;
}

int main() {
    // Inisialisasi graf
    vector<vector<double>> graph = create_matrix(N);
    build_random_graph(graph, DENSITY);
    
    cout << "Running MCL on " << N << "x" << N 
         << " graph (density: " << DENSITY*100 << "%)" << endl;
    
    // Jalankan MCL dan ukur waktu
    clock_t start = clock();
    run_mcl(graph, MAX_ITER, INFLATION);
    clock_t end = clock();
    
    double time_spent = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "Execution time: " << time_spent << " seconds" << endl;
    
    return 0;
}