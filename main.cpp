#include <iostream>
#include <random>
#include <chrono>
#include <fstream>

#include "fast_flagstats.h"

bool assert_truth(uint32_t* __restrict__ vals, uint32_t* __restrict__ truth) {
    uint64_t n_all = 0;
    for(int i = 0; i < 16; ++i) n_all += vals[i];
    if(n_all == 0) return true;
    
    for(int i = 0; i < 16; ++i) {
        assert(vals[i] == truth[i]);
    }
    return true;
}

// Definition for time.
typedef std::chrono::high_resolution_clock::time_point clockdef;

template <uint32_t(f)(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags)>
uint32_t pospopcnt_u16_wrapper(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
     clockdef t1 = std::chrono::high_resolution_clock::now();
     (*f)(data, n, flags);

     clockdef t2 = std::chrono::high_resolution_clock::now();
     auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
     //std::cerr << "time_span=" << time_span.count() << std::endl;
     return time_span.count();
}

void flag_functions(uint16_t* __restrict__ vals, uint64_t* __restrict__ times, uint64_t* __restrict__ times_local, const uint32_t n) {
    uint32_t truth[16];
    uint32_t flags[16];

    // start tests
    // scalar naive
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t time_naive = pospopcnt_u16_wrapper<&pospopcnt_u16_scalar_naive>(vals,n,truth);
    times[1] += time_naive;
    times_local[1] = time_naive;

    // scalar partition
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t time_partition = pospopcnt_u16_wrapper<&pospopcnt_u16_scalar_partition>(vals,n,flags);
    times[2] += time_partition;
    times_local[2] = time_partition;
    assert_truth(flags, truth);

    // avx2 aggl
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2>(vals, n, flags);
    times[3] += avx2_timing;
    times_local[3] = avx2_timing;
    assert_truth(flags, truth);

    // avx2 popcnt
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t popcnt_timing = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_popcnt>(vals, n, flags);
    times[4] += popcnt_timing;
    times_local[4] = popcnt_timing;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_single_timing = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_single>(vals, n, flags);
    times[5] += avx2_single_timing;
    times_local[5] = avx2_single_timing;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_mula_remake = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_mula_unroll4>(vals, n, flags);
    times[16] += avx2_timing_mula_remake;
    times_local[16] = avx2_timing_mula_remake;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_naive = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_naive_counter>(vals, n, flags);
    times[6] += avx2_timing_naive;
    times_local[6] = avx2_timing_naive;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_lemire = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_lemire>(vals, n, flags);
    times[13] += avx2_timing_lemire;
    times_local[13] = avx2_timing_lemire;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_lemire2 = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_lemire2>(vals, n, flags);
    times[14] += avx2_timing_lemire2;
    times_local[14] = avx2_timing_lemire2;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_mula = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_mula>(vals, n, flags);
    times[15] += avx2_timing_mula;
    times_local[15] = avx2_timing_mula;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t sse_single_timing = pospopcnt_u16_wrapper<&pospopcnt_u16_sse_single>(vals, n, flags);
    times[7] += sse_single_timing;
    times_local[7] = sse_single_timing;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_mula_remake8 = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_mula_unroll8>(vals, n, flags);
    times[17] += avx2_timing_mula_remake8;
    times_local[17] = avx2_timing_mula_remake8;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t hist1x4_timing = pospopcnt_u16_wrapper<&pospopcnt_u16_hist1x4>(vals, n, flags);
    times[8] += hist1x4_timing;
    times_local[8] = hist1x4_timing;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_timings = pospopcnt_u16_wrapper<&pospopcnt_u16_avx512_popcnt>(vals, n, flags);
    times[9] += avx512_timings;
    times_local[9] = avx512_timings;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_timings32 = pospopcnt_u16_wrapper<&pospopcnt_u16_avx512_popcnt32_mask>(vals, n, flags);
    times[10] += avx512_timings32;
    times_local[10] = avx512_timings32;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_mula_remake16 = pospopcnt_u16_wrapper<&pospopcnt_u16_avx2_mula_unroll16>(vals, n, flags);
    times[18] += avx2_timing_mula_remake16;
    times_local[18] = avx2_timing_mula_remake16;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_agg_timings = pospopcnt_u16_wrapper<&pospopcnt_u16_avx512>(vals, n, flags);
    times[11] += avx512_agg_timings;
    times_local[11] = avx512_agg_timings;
    assert_truth(flags, truth);

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_timings64 = pospopcnt_u16_wrapper<&pospopcnt_u16_avx512_popcnt64_mask>(vals, n, &flags[0]);
    times[12] += avx512_timings64;
    times_local[12] = avx512_timings64;
    assert_truth(flags, truth);
}

void flag_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating flags: " << n << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    //uint16_t* vals = new uint16_t[n];
    uint16_t* vals;
    assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n*sizeof(uint16_t)));

    uint64_t times[20] = {0};
    uint64_t times_local[20] = {0};

    uint32_t flags[16] = {0};
    uint64_t flags_out[16] = {0};

    std::vector<uint32_t> ranges = {8, 16, 64, 256, 512, 1024, 4096, 65536};
    for(int r = 0; r < ranges.size(); ++r) {
        std::uniform_int_distribution<uint16_t> distr(1, ranges[r]); // right inclusive

        for(int c = 0; c < cycles; ++c) {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            for(uint32_t i = 0; i < n; ++i) {
                vals[i] = distr(eng); // draw random values
            }

            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            times[0] += time_span.count();
            times_local[0] = time_span.count();

            // start tests
            flag_functions(vals, times, times_local, n);

#define MHZ 2800000000.0
#define MBS(cum) (times_local[cum] == 0 ? 0 : ((n*sizeof(uint16_t)) / (1024*1024.0)) / (times_local[cum] / 1000000.0))
#define SPEED(cum) (times_local[cum] == 0 ? 0 : (MHZ * (times_local[cum] / 1000000.0) / n))
            std::cout << "MBS\t" << ranges[r] << "\t" << c;
            for(int i = 1; i < 19; ++i) std::cout << '\t' << MBS(i);
            std::cout << "\nCycles\t" << ranges[r] << "\t" << c;
            for(int i = 1; i < 19; ++i) std::cout << '\t' << SPEED(i);
            std::cout << std::endl;
#undef MBS
#undef SPEED
        }
#define AVG(pos) (times[pos] == 0 ? 0 : (double)times[pos]/cycles)
        std::cout << "average times\t" << AVG(1) << "\t" << AVG(2) << "\t" << AVG(3) << "\t" << AVG(4) << "\t" << AVG(5) << "\t" << AVG(6) << "\t" << AVG(7) << "\t" << AVG(8) << "\t" << AVG(9) << "\t" << AVG(10)<< "\t" << AVG(11) << "\t" << AVG(12) << "\t" << AVG(13) << "\t" << AVG(14) << "\t" << AVG(15) << "\t" << AVG(16) << "\t" << AVG(17) << "\t" << AVG(18) << std::endl;
#define INTS_SEC(cum) (times[cum] == 0 ? 0 : ((n*sizeof(uint16_t)) / (1024*1024.0)) / (AVG(cum) / 1000000.0))
#define AVG_CYCLES(pos) (times[pos] == 0 ? 0 : (MHZ * (AVG(pos) / 1000000.0) / n))
        std::cout << "MB/s\t" << INTS_SEC(1) << "\t" << INTS_SEC(2) << "\t" << INTS_SEC(3) << "\t" << INTS_SEC(4) << "\t" << INTS_SEC(5) << "\t" << INTS_SEC(6) << "\t" << INTS_SEC(7) << "\t" << INTS_SEC(8) << "\t" << INTS_SEC(9) << "\t" << INTS_SEC(10) << "\t" << INTS_SEC(11) << "\t" << INTS_SEC(12) << "\t" << INTS_SEC(13) << "\t" << INTS_SEC(14) << "\t" << INTS_SEC(15) << "\t" << INTS_SEC(16) << "\t" << INTS_SEC(17) << "\t" << INTS_SEC(18) << std::endl;
        std::cout << "Cycles/int\t" << AVG_CYCLES(1) << "\t" << AVG_CYCLES(2) << "\t" << AVG_CYCLES(3) << "\t" << AVG_CYCLES(4) << "\t" << AVG_CYCLES(5) << "\t" << AVG_CYCLES(6) << "\t" << AVG_CYCLES(7) << "\t" << AVG_CYCLES(8) << "\t" << AVG_CYCLES(9) << "\t" << AVG_CYCLES(10) << "\t" << AVG_CYCLES(11) << "\t" << AVG_CYCLES(12) << "\t" << AVG_CYCLES(13) << "\t" << AVG_CYCLES(14) << "\t" << AVG_CYCLES(15) << "\t" << AVG_CYCLES(16) << "\t" << AVG_CYCLES(17) << "\t" << AVG_CYCLES(18) << std::endl;
#undef AVG
#undef INTS_SEC

        memset(times, 0, sizeof(uint64_t)*20);
    }

    delete[] vals;
}

int main(int argc, char **argv) {
    std::cerr << "rdtsc=" << __rdtsc() << std::endl;
    if(argc == 1) flag_test(100000000, 10);
    else if(argc == 2) flag_test(std::atoi(argv[1]), 10);
    else if(argc == 3) flag_test(std::atoi(argv[1]), std::atoi(argv[2]));
    else return(1);
    return(0);
}
