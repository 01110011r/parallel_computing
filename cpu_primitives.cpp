//
// Created by rss on 4/1/26.
//

#include <iostream>
#include <omp.h>
#include <algorithm>
#include <ctime>
#include <tbb/parallel_for.h>
#include <vector>
#include <chrono>

#define ARR_SIZE 1000000
#define ARR_VAL 1
#define NUM_THREADS 7

using namespace std;

int main() {
// ########################## OpenMP (Open Multi Processing) ######################
// #pragma omp parallel
//     {
//         int id = omp_get_thread_num(); // thread reaqamini olish
//         int total = omp_get_num_threads(); // os dagi threads soni
//
//         std::cout << "Hello from thread " << id
//                   << " / " << total << std::endl;
//     }

//     cout << "########### Benchmark ############" << endl;
//     int *arr = new int[ARR_SIZE];
//     fill_n(arr, ARR_SIZE, ARR_VAL);
//
//     cout << "Serial" << endl;
//     double start_serial = omp_get_wtime();
//     for (int i = 0; i < ARR_SIZE; i++)
//     {
//         arr[i] = arr[i] * 2;
//     }
//     double end_serial = omp_get_wtime();
//     cout << "Serial time: " << end_serial - start_serial << endl;
//
//     cout << "Parallel. Threads num set to " << NUM_THREADS << endl;
//     omp_set_num_threads(NUM_THREADS); // 7
//     double start_parallel = omp_get_wtime();
// #pragma omp parallel for
//     for (int i = 0; i < ARR_SIZE; i++)
//     {
//         arr[i] = arr[i] / 2;
//     }
//     double end_parallel = omp_get_wtime();
//     cout << "Parallel time: " << end_parallel - start_parallel << endl;
//
//     // free
//     free(arr);

// ############################ TBB (Threading Building Blocks) #######################
    vector <int> arr(ARR_SIZE); // 1_000_000
    auto start = chrono::high_resolution_clock::now();
    tbb::parallel_for(0, ARR_SIZE, 1, [&](int i)
    {
        arr[i] = i * 2;
    });
    auto end = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double>(end - start).count();
    cout << "Size: " << arr.size() << "\t --> \t" << arr[0] << "..." << arr[arr.size() - 1] <<
        "\nTime: " << duration << endl;

    cout << "TBB da 2D massiv bilan ishlash" << endl;
    int mat_size = 1000;
    // vector<vector<int>> mat(ARR_SIZE, vector<int>(ARR_SIZE)); // fragmented memory
    vector<int> mat(mat_size * mat_size);
    tbb::parallel_for(0, mat_size, [&](int i)
    {
        for (int j = 0; j < mat_size; j++)
        // tbb::parallel_for(0, ARR_SIZE, [&](int j)
        {
            // rasmni blur qilishga misol
            // mat[i][j] = (mat[i][j] + mat[i+1][j] + mat[i][j+1])/3;

            // fill
            mat[i * mat_size + j] = i * mat_size + j;
        }//);
    });

}
