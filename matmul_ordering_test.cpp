#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <iomanip>
#include "array_types.hpp"
#include "linalg.hpp"
#include "benchmark.hpp"


template <class T>
matrix<T> matmul_ijk(matrix<T> a, matrix<T> b)
{
    ptrdiff_t rowa = a.nrows();
    ptrdiff_t cola = a.ncols();
    ptrdiff_t colb = b.ncols();

    T* a_raw = a.raw_ptr();
    T* b_raw = b.raw_ptr();

    matrix<T> c(rowa, colb);
    T* c_raw = c.raw_ptr();
    for (ptrdiff_t i = 0; i < rowa; i++) {
        for (ptrdiff_t j = 0; j < colb; j++)
        {
            c(i, j) = 0;
            for (ptrdiff_t k = 0; k < cola; k++)
            {
                c(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return c;
}

template <class T>
matrix<T> matmul_ikj(matrix<T> a, matrix<T> b)
{
    ptrdiff_t rowa = a.nrows();
    ptrdiff_t cola = a.ncols();
    ptrdiff_t colb = b.ncols();

    T* a_raw = a.raw_ptr();
    T* b_raw = b.raw_ptr();

    matrix<T> c(rowa, colb);
    T* c_raw = c.raw_ptr();
    for (ptrdiff_t i = 0; i < rowa; i++)
    {
        for (ptrdiff_t j = 0; j < colb; j++)
        {
            c(i, j) = 0;
        }
        for (ptrdiff_t k = 0; k < cola; k++)
        {
            T aik = a(i, k);
            for (ptrdiff_t j = 0; j < colb; j++)
            {
                c(i, j) += aik * b(k, j);
            }
        }
    }
    return c;
}

template <class T>
matrix<T> matmul_dot(matrix<T> a, matrix<T> b)
{
    ptrdiff_t rowa = a.nrows();
    ptrdiff_t cola = a.ncols();
    ptrdiff_t colb = b.ncols();

    T* a_raw = a.raw_ptr();
    T* b_raw = b.raw_ptr();

    matrix<T> c(rowa, colb);
    T* c_raw = c.raw_ptr();
    for (ptrdiff_t i = 0; i < rowa; i++)
        for (ptrdiff_t j = 0; j < colb; j++)
        {
            c(i, j) = dot(a.row(i), b.col(j));
        }
    return c;
}
template <typename T>
void print_matrix(matrix<T> m) {
    std::cout << "Matrix " << m.nrows() << " by " << m.ncols() << "\n";
    for (int i = 0; i < m.nrows(); i++) {
        for (int j = 0; j < m.ncols(); j++) {
            std::cout << std::setw(3) << m(i, j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
int main_1(int argc, char* argv[])
{
    //int n = std::stoi(argv[1]);
    int n = 10;
    matrix<double> a(n, n);
    matrix<double> b(n, n);

    for (long i = 0; i < a.length(); i++)
    {
        a(i) = i;
        b(b.length() - i - 1) = i;
    }
    print_matrix(a);
    print_matrix(b);
    std::function<double(int)> test_ijk = [=](int idx) {return matmul_ijk(a, b)(idx); };
    std::function<double(int)> test_ikj = [=](int idx) {return matmul_ikj(a, b)(idx); };
    std::function<double(int)> test_dot = [=](int idx) {return matmul_dot(a, b)(idx); };

    auto benchresult_ijk = benchmark(test_ijk, 0, 1000);
    auto benchresult_ikj = benchmark(test_ikj, 0, 1000);
    auto benchresult_dot = benchmark(test_dot, 0, 1000);

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
        << "Matmul timing (IJK): " << benchresult_ijk.btime << " ms\n"
        << "Answer(0,0) = " << benchresult_ijk.result << "\n"
        << "Matmul timing (IKJ): " << benchresult_ikj.btime << " ms\n"
        << "Answer(0,0) = " << benchresult_ikj.result << "\n"
        << "Matmul timing (DOT): " << benchresult_dot.btime << " ms\n"
        << "Answer(0,0) = " << benchresult_dot.result
        << std::endl;
    return 0;
}