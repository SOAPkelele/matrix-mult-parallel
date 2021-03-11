#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <functional>
#include <chrono>
#include <utility>

using size_t = std::size_t;

template <class T>
struct benchresult {
    T result;
    double btime;
};

template <class T>
auto benchmark(std::function<T()> fn, size_t nrepeat) {
    T result;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < nrepeat; i++) {
        result = fn();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_s = end - start;
    double ms_per_run = duration_s.count() * 1000 / nrepeat;
    return benchresult<T> {result, ms_per_run};
}

#endif