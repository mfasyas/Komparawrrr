#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <algorithm>

using namespace std;

class MCL {
private:
    vector<vector<double>> matrix;
    int size;
    bool add_self_loops;

    bool is_converged(const vector<vector<double>>& prev, const vector<vector<double>>& curr, double epsilon) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (fabs(prev[i][j] - curr[i][j]) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

    void normalize_matrix() {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int i = 0; i < size; ++i) {
                sum += matrix[i][j];
            }
            if (sum > 0) {
                for (int i = 0; i < size; ++i) {
                    matrix[i][j] /= sum;
                }
            }
        }
    }

    void expand(int e) {
        vector<vector<double>> result = matrix;
        vector<vector<double>> temp;

        for (int power = 1; power < e; ++power) {
            temp = vector<vector<double>>(size, vector<double>(size, 0.0));
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    for (int k = 0; k < size; ++k) {
                        temp[i][j] += result[i][k] * matrix[k][j];
                    }
                }
            }
            result = temp;
        }
        matrix = result;
    }

    void inflate(double r) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                matrix[i][j] = pow(matrix[i][j], r);
            }
        }
    }

public:
    MCL(int n, bool self_loops = true) : size(n), add_self_loops(self_loops) {
        matrix = vector<vector<double>>(size, vector<double>(size, 0.0));
    }

    void add_edge(int i, int j, double weight) {
        matrix[i][j] = weight;
        matrix[j][i] = weight;
    }

    void random_graph(double density) {
        srand(time(NULL));
        int possible_edges = size * (size - 1) / 2;
        int edges = static_cast<int>(possible_edges * density);

        for (int e = 0; e < edges; ++e) {
            int i = rand() % size;
            int j = rand() % size;
            if (i != j && matrix[i][j] == 0) { // Pastikan edge unik dan bukan self-loop
                double weight = 0.1 + (rand() % 90) / 100.0; // Bobot 0.10-1.00
                add_edge(i, j, weight);
            }
        }
    }

    void run(int e, double r, int max_iter, double epsilon = 1e-3) { // epsilon lebih longgar
        if (add_self_loops) {
            for (int i = 0; i < size; ++i) {
                matrix[i][i] = 0.5; // Self-loop dengan bobot kecil
            }
        }

        normalize_matrix();

        vector<vector<double>> prev_matrix;
        int iter;
        for (iter = 0; iter < max_iter; ++iter) {
            prev_matrix = matrix;

            expand(e);
            inflate(r);
            normalize_matrix();
            cout << "Iteration: " << iter << " is done" << "\n";

            if (is_converged(prev_matrix, matrix, epsilon)) {
                break;
            }
        }

        cout << "MCL converged after " << iter << " iterations\n";
    }

    void print_clusters(double threshold = 0.1) {
        vector<bool> assigned(size, false);
        int cluster_num = 1;

        for (int i = 0; i < size; ++i) {
            if (!assigned[i]) {
                vector<int> cluster;
                for (int j = 0; j < size; ++j) {
                    if (matrix[i][j] > threshold) {
                        cluster.push_back(j);
                        assigned[j] = true;
                    }
                }
                if (!cluster.empty()) {
                    cout << "Cluster " << cluster_num++ << ": ";
                    for (int node : cluster) cout << node << " ";
                    cout << "\n";
                }
            }
        }
    }
};

int main() {
    const int size = 1000;          // Ukuran lebih kecil untuk demonstrasi
    const double density = 0.111;   // Kepadatan lebih tinggi
    const int e = 2;              
    const double r = 2;         // Inflasi lebih rendah
    const int max_iter = 100;     

    MCL mcl(size);
    mcl.random_graph(density);

    cout << "Running MCL with parameters:\n";
    cout << "Graph size: " << size << "x" << size << "\n";
    cout << "Edge density: " << density*100 << "%\n";
    cout << "Expansion (e): " << e << "\n";
    cout << "Inflation (r): " << r << "\n\n";

    clock_t start = clock();
    mcl.run(e, r, max_iter);
    clock_t end = clock();

    cout << "\nDetected clusters:\n";
    mcl.print_clusters();

    double time_spent = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    cout << "\nExecution time: " << time_spent << " seconds\n";

    return 0;
}