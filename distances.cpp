#include <cmath>
#include <vector>
#include <memory.h>

using namespace std;


int similarity(int* sequence1, int* sequence2, int n_row) {
    int sum = 0;
    for(int i = 0; i < n_row; i++) {
        sum += abs(sequence1[i] - sequence2[i]);
    }

    return sum;
}

extern "C" {
    double** distance(
        double* sequence1, 
        double* sequence2, 
        int n_row,
        int n_col,
        int** feature_coding_matrix1, 
        int** feature_coding_matrix2,
        double alpha
        ) {
            int distance1;
            double distance2;
            double min_distance;

            // init memory
            double** distances = new double*[n_row];
            for(int i = 0; i < n_col; i++) {
                distances[i] = new double[n_row];
                memset(distances[i], 1, sizeof(double) * n_row);
            }

            // first value of warping path
            distance1 = similarity(feature_coding_matrix1[0], feature_coding_matrix2[0], n_row);
            distance2 = (sequence1[0] - sequence2[0]) * (sequence1[0] - sequence2[0]);
            distances[0][0] = alpha*distance1 + (1-alpha)*distance2;

            // first column of warping path
            for (int i = 0; i < n_row; i++) {
                distance1 = similarity(feature_coding_matrix1[i], feature_coding_matrix2[0], n_row);
                distance2 = (sequence1[i] - sequence2[0]) * (sequence1[i] - sequence2[0]);
                distances[i][0] = distances[i-1][0] + (alpha*distance1 + (1-alpha)*distance2);
            }
                
            // top row of warping path
            for (int j = 0; j < n_col; j++) {
                distance1 = similarity(feature_coding_matrix1[0], feature_coding_matrix2[j], n_row);
                distance2 = (sequence1[0] - sequence2[j]) * (sequence1[0] - sequence2[j]);
                distances[0][j] = distances[0][j-1] + (alpha*distance1 + (1-alpha)*distance2);
            }
                
            // warping path
            for (int i = 0; i < n_row; i++) {
                for (int j = 0; j < n_col; j++) {
                    distance1 = similarity(feature_coding_matrix1[i], feature_coding_matrix2[j], n_row);
                    distance2 = (sequence1[i] - sequence2[j]) * (sequence1[i] - sequence2[j]);

                    min_distance = distances[i-1][j-1];
                    if (min_distance > distances[i-1][j])
                        min_distance = distances[i-1][j];
                    if (min_distance > distances[i][j-1])
                         min_distance = distances[i][j-1];

                    distances[i][j] = min_distance + (alpha*distance1 + (1-alpha)*distance2);
                }
            }

            return distances;
        }
}
