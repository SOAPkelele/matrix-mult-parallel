#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <iomanip>
#include <tuple>
#include <omp.h>
#include "array_types.hpp"
#include "linalg.hpp"
#include "benchmark.hpp"

template <class T>
matrix<T> matmul_aT_b(matrix<T> a, matrix<T> b)
{
    ptrdiff_t rowa = a.nrows();
    ptrdiff_t cola = a.ncols();
    ptrdiff_t rowb = b.nrows();
    ptrdiff_t colb = b.ncols();

    // тк матрица транспонированная то 
    if (rowa != rowb) {
        throw "Matrixes can\'t be multiplied\n";;
    }

    matrix<T> c(cola, colb);

    for (ptrdiff_t i = 0; i < cola; i++) {
        for (ptrdiff_t j = 0; j < colb; j++)
        {
            c(i, j) = 0;
        }
        for (ptrdiff_t k = 0; k < rowa; k++)
        {
            T aik = a(k, i);
            for (ptrdiff_t j = 0; j < colb; j++) {
                    c(i, j) += aik * b(k, j);                                   
                // std::cout << "C[" << i << ";" << j << "]" << " += " << aik << " * " << b(k, j) << "\n";
            }
        }
    }

    return c;
}

template <class T>
matrix<T> matmul_aT_b_complex(matrix<T> a, matrix<T> b)
{
    ptrdiff_t rowa = a.nrows();
    ptrdiff_t cola = a.ncols() / 2;
    ptrdiff_t rowb = b.nrows();
    ptrdiff_t colb = b.ncols() / 2;

    // тк матрица транспонированная то 
    if (rowa != rowb) {
        throw "Matrixes can\'t be multiplied\n";;
    }

    matrix<T> c(cola, colb*2);
    ptrdiff_t i, j, k;

#pragma omp parallel for private(i, j, k), shared(a, b, c)
    for (i = 0; i < cola; i++) {
        for (j = 0; j < colb; j++)
        {
            c(i, 2 * j) = 0;
            c(i, 2 * j + 1) = 0;
        }
        for (k = 0; k < rowa; k++)
        {
            T aik_re = a(k, 2 * i);
            T aik_im = (-1) * a(k, 2 * i + 1);
            for (j = 0; j < colb; j++) {
                c(i, 2 * j) += (aik_re * b(k, 2 * j) - aik_im * b(k, 2 * j + 1));
                c(i, 2 * j + 1) += (aik_re * b(k, 2 * j + 1) + aik_im * b(k, 2 * j));
            }
        }
    }

    return c;
}

template <typename T>
void print_matrix(matrix<T> m) {
    std::cout << m.nrows() << " " << m.ncols() << "\n";
    for (int i = 0; i < m.nrows(); i++) {
        for (int j = 0; j < m.ncols(); j++) {
            std::cout << std::setw(3) << m(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

std::tuple<matrix<double>, matrix<double>> get_matrixes_from_file(std::string filename) {
    std::ifstream file;
    file.open(filename);
    int a_rows, a_cols, b_rows, b_cols;

    file >> a_rows >> a_cols;
    matrix<double> a(a_rows, a_cols);
    for (int i = 0; i < a_rows; i++)
        for (int j = 0; j < a_cols; j++)
            file >> a(i, j);

    file >> b_rows >> b_cols;
    matrix<double> b(b_rows, b_cols);
    for (int i = 0; i < b_rows; i++)
        for (int j = 0; j < b_cols; j++)
            file >> b(i, j);
    file.close();
    std::cout << "file closed";
    return { a, b };
}

std::tuple<matrix<double>, matrix<double>> get_complex_matrixes_from_file(std::string filename) {
    std::ifstream file;
    file.open(filename);
    int a_rows, a_cols, b_rows, b_cols;

    file >> a_rows >> a_cols;
    a_cols *= 2;
    matrix<double> a(a_rows, a_cols);
    for (int i = 0; i < a_rows; i++)
        for (int j = 0; j < a_cols; j++)
            file >> a(i, j);
            
    file >> b_rows >> b_cols;
    b_cols *= 2;
    matrix<double> b(b_rows, b_cols);
    for (int i = 0; i < b_rows; i++)
        for (int j = 0; j < b_cols; j++)
            file >> b(i, j);

    file.close();
    std::cout << "file closed";
    return { a, b };
}

int main(int argc, char* argv[])
{   
    matrix<double> a;
    matrix<double> b;
    
    //считываем из файла  
    if (argv[1] == "<") {
        std::tuple<matrix<double>, matrix<double>> matrixes = get_complex_matrixes_from_file(argv[2]);
        a = std::get<0>(matrixes);
        b = std::get<1>(matrixes);
    }
    else { //считываем из stdin
        int a_rows, a_cols, b_rows, b_cols;
        //get matrix A
        std::cin >> a_rows >> a_cols;
        a_cols *= 2;
        a = matrix<double> (a_rows, a_cols);
        for (int i = 0; i < a_rows; i++) {
            for (unsigned j = 0; j < a_cols; j++) {
                std::cin >> a(i, j);
            }
        }

        //get matrix B
        std::cin >> b_rows >> b_cols;
        b_cols *= 2;
        b = matrix<double> (b_rows, b_cols);
        for (int i = 0; i < b_rows; i++) {
            for (unsigned j = 0; j < b_cols; j++) {
                std::cin >> b(i, j);
            }
        }
    }

    std::cout << "Matrix A\n";
    print_matrix(a);
    std::cout << "Matrix B\n";
    print_matrix(b);
    
    //глобал для указания количества потоков 
    omp_set_num_threads(4);

    std::function<matrix<double>()> test_mult = [=]() {return matmul_aT_b_complex(a, b); };
    auto benchresult_ijk = benchmark(test_mult, 1000);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    print_matrix(benchresult_ijk.result);

    //std::cout << "Matmul timing: " << benchresult_ijk.btime << " ms\n";

    return 0;
}

// complex matrix 20x20 
// 0.8235082 ms [1 thread]
// 0.4091862 ms [4 threads]
// 0.3691107 ms [8 threads]

/*template <class T>
matrix<T> matmul_aT_b_complex(matrix<T> a, matrix<T> b)
{
    ptrdiff_t rowa = a.nrows();
    ptrdiff_t cola = a.ncols();
    ptrdiff_t rowb = b.nrows();
    ptrdiff_t colb = b.ncols();

    // тк матрица транспонированная то 
    if (rowa != rowb) {
        throw "Matrixes can\'t be multiplied\n";;
    }

    matrix<T> c(cola, colb);
    ptrdiff_t i, j, k;

#pragma omp parallel for private(i, j, k), shared(a, b, c)
    for (i = 0; i < cola; i++) {
        for (j = 0; j < colb; j++)
        {
            c(i, j) = 0;
        }
        for (k = 0; k < rowa; k++)
        {
            T aik = a(k, i);
            for (j = 0; j < colb; j++) {
                if (j % 2 != 0) {
                    c(i, j) += aik * b(k, j);
                }
                else
                {
                    c(i, j) += aik * b(k, j);
                }
                // std::cout << "C[" << i << ";" << j << "]" << " += " << aik << " * " << b(k, j) << "\n";
            }
        }
    }

    return c;
}*/

/*for (ptrdiff_t i = 0; i < cola; i++) {
    for (ptrdiff_t j = 0; j < colb; j++) {
        c(i, j) = 0;
        for (ptrdiff_t k = 0; k < rowa; k++)
        {
            c(i, j) += a(k, i) * b(k, j);
            std::cout << "C["<<i<<";"<<j <<"]" << " += " << a(k, i) << " * " << b(k, j) << "\n";
        }
    }
}*/

/*int main(int argc, char* argv[])
{
    matrix<double> a;
    matrix<double> b;
    if (argv[1] == "<") {
        std::tuple<matrix<double>, matrix<double>> matrixes = get_matrixes_from_file(argv[2]);
        a = std::get<0>(matrixes);
        b = std::get<1>(matrixes);
    }
    else {
        int a_rows, a_cols, b_rows, b_cols;
        //get matrix A
        std::cout << "Matrix A: rows - columns\n";
        std::cin >> a_rows >> a_cols;
        a = matrix<double>(a_rows, a_cols);
        for (int i = 0; i < a_rows; i++) {
            for (unsigned j = 0; j < a_cols; j++) {
                std::cin >> a(i, j);
            }
        }

        //get matrix B
        std::cout << "Matrix B: rows - columns\n";
        std::cin >> b_rows >> b_cols;
        b = matrix<double>(b_rows, b_cols);
        for (int i = 0; i < b_rows; i++) {
            for (unsigned j = 0; j < b_cols; j++) {
                std::cin >> b(i, j);
            }
        }
    }

    std::cout << "Matrix A\n";
    print_matrix(a);
    std::cout << "Matrix B\n";
    print_matrix(b);

    std::function<matrix<double>()> test_mult = [=]() {return matmul_aT_b(a, b); };

    auto benchresult_ijk = benchmark(test_mult, 1000);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
        << "Matrix C:\n";
    print_matrix(benchresult_ijk.result);

    std::cout << "Matmul timing: " << benchresult_ijk.btime << " ms\n";

    return 0;
}*/