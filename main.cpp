#include <iostream>
#include <random>
#include <chrono>

#include "fast_flagstats.h"

void flag_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating flags: " << n << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    //uint16_t* vals = new uint16_t[n];
    uint16_t* vals;
    assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n*sizeof(uint16_t)));

    uint64_t times[8] = {0};
    uint64_t times_local[8];

    std::vector<uint32_t> ranges = {16, 64, 256, 512, 1024, 4096, 65536};
    for(int r = 0; r < ranges.size(); ++r) {
        std::uniform_int_distribution<uint16_t> distr(1, ranges[r]); // right inclusive

        for(int c = 0; c < cycles; ++c) {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            for(uint32_t i = 0; i < n; ++i) {
                vals[i] = distr(eng);
            }
            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            times[0] += time_span.count();
            times_local[0] = time_span.count();

            // start tests
            // scalar naive
            uint32_t flags[16];
            uint32_t time_naive = flag_stats_wrapper<&flag_stats_scalar_naive>(vals,n,&flags[0]);
            times[1] += time_naive;
            times_local[1] = time_naive;

            //uint32_t tt = flag_stats_wrapper<&flag_stats_scalar_naive, uint32_t>(vals,n,&flags[0]);
            //std::cerr << "tt=" << tt << std::endl;

            // scalar partition
            uint32_t flags2[16]; memset(flags2, 0, sizeof(uint32_t)*16);
            uint32_t time_partition = flag_stats_wrapper<&flag_stats_scalar_partition>(vals,n,&flags[0]);
            times[2] += time_partition;
            times_local[2] = time_partition;

            // avx2 aggl
            memset(flags2, 0, sizeof(uint32_t)*16);
            uint32_t avx2_timing = flag_stats_avx2(vals, n, &flags2[0]);
            times[3] += avx2_timing;
            times_local[3] = avx2_timing;

            // avx2 popcnt
            memset(flags2, 0, sizeof(uint32_t)*16);
            uint32_t popcnt_timing = flag_stats_avx2_popcnt(vals, n, &flags2[0]);
            times[4] += popcnt_timing;
            times_local[4] = popcnt_timing;

            memset(flags2, 0, sizeof(uint32_t)*16);
            uint32_t avx2_single_timing = flag_stats_avx2_single(vals, n, &flags2[0]);
            times[5] += avx2_single_timing;
            times_local[5] = avx2_single_timing;

            memset(flags2, 0, sizeof(uint32_t)*16);
            uint32_t avx2_timing_naive = flag_stats_avx2_naive_counter(vals, n, &flags2[0]);
            times[6] += avx2_timing_naive;
            times_local[6] = avx2_timing_naive;

            memset(flags2, 0, sizeof(uint32_t)*16);
            uint32_t sse_single_timing = flag_stats_sse_single(vals, n, &flags2[0]);
            times[7] += sse_single_timing;
            times_local[7] = sse_single_timing;

            std::cout << ranges[r] << "\t" << c << "\t" << times_local[0] << "\t" << times_local[1] << "\t" << times_local[2] << "\t" << times_local[3] << "\t" << times_local[4] << "\t" << times_local[5] << "\t" << times_local[6] << "\t" << times_local[7] << std::endl;
        }
#define AVG(pos) (cycles == 0 ? 0 : (double)times[pos]/cycles)
        std::cout << "average times\t" << AVG(0) << "\t" << AVG(1) << "\t" << AVG(2) << "\t" << AVG(3) << "\t" << AVG(4) << "\t" << AVG(5) << "\t" << AVG(6) << "\t" << AVG(7) << std::endl;
#undef AVG
#define INTS_SEC(cum) (cycles == 0 ? 0 : (double)(n/(times[cum]/cycles)*1e6))
        std::cout << "int16/s\t" << 0 << "\t" << INTS_SEC(1) << "\t" << INTS_SEC(2) << "\t" << INTS_SEC(3) << "\t" << INTS_SEC(4) << "\t" << INTS_SEC(5) << "\t" << INTS_SEC(6) << "\t" << INTS_SEC(7) << std::endl;
#undef INTS_SEC

        memset(times, 0, sizeof(uint64_t)*8);
    }

    delete[] vals;
}

int main(int argc, char **argv) {
    flag_test(100000000, 10);
    return(0);
}
