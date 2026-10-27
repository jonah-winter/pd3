#include <benchmark/benchmark.h>
#include "pd3.h"

static void BM_test_namesake(benchmark::State& state) {
    float d[] = {1,2,3,4,5,6,7,8,10};
    char e[256];
    pdmat3 a;
    pd3_make_mat(d, a);
    for(auto _ : state) {
        pdmat3 u;
        pdmat3 p;
        if(pd3_newton(a, u, p, 1e-6f, 500, e) != PD3_SUCC) { printf("%s\n", e); }
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(u);
        benchmark::DoNotOptimize(p);
    }
}

BENCHMARK(BM_test_namesake);

BENCHMARK_MAIN();