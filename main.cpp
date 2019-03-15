#include <iostream>
#include <random>
#include <chrono>
#include <fstream>

#include "fast_flagstats.h"

void flag_functions(uint16_t* __restrict__ vals, uint64_t* __restrict__ times, uint64_t* __restrict__ times_local, const uint32_t n) {
    uint32_t flags[16];

    // start tests
    // scalar naive
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t time_naive = flag_stats_wrapper<&flag_stats_scalar_naive>(vals,n,flags);
    times[1] += time_naive;
    times_local[1] = time_naive;

    // scalar partition
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t time_partition = flag_stats_wrapper<&flag_stats_scalar_partition>(vals,n,flags);
    times[2] += time_partition;
    times_local[2] = time_partition;

    // avx2 aggl
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing = flag_stats_avx2(vals, n, flags);
    times[3] += avx2_timing;
    times_local[3] = avx2_timing;

    // avx2 popcnt
    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t popcnt_timing = flag_stats_avx2_popcnt(vals, n, flags);
    times[4] += popcnt_timing;
    times_local[4] = popcnt_timing;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_single_timing = flag_stats_avx2_single(vals, n, flags);
    times[5] += avx2_single_timing;
    times_local[5] = avx2_single_timing;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx2_timing_naive = flag_stats_avx2_naive_counter(vals, n, flags);
    times[6] += avx2_timing_naive;
    times_local[6] = avx2_timing_naive;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t sse_single_timing = flag_stats_sse_single(vals, n, flags);
    times[7] += sse_single_timing;
    times_local[7] = sse_single_timing;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t hist1x4_timing = flag_stats_hist1x4(vals, n, flags);
    times[8] += hist1x4_timing;
    times_local[8] = hist1x4_timing;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_timings = flag_stats_avx512_popcnt(vals, n, flags);
    times[9] += avx512_timings;
    times_local[9] = avx512_timings;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_timings32 = flag_stats_avx512_popcnt32_mask(vals, n, flags);
    times[10] += avx512_timings32;
    times_local[10] = avx512_timings32;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_agg_timings = flag_stats_avx512(vals, n, flags);
    times[11] += avx512_agg_timings;
    times_local[11] = avx512_agg_timings;

    memset(flags, 0, sizeof(uint32_t)*16);
    uint32_t avx512_timings64 = flag_stats_avx512_popcnt64_mask(vals, n, &flags[0]);
    times[12] += avx512_timings64;
    times_local[12] = avx512_timings64;
}

void flag_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating flags: " << n << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    //uint16_t* vals = new uint16_t[n];
    uint16_t* vals;
    assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n*sizeof(uint16_t)));

    uint64_t times[15] = {0};
    uint64_t times_local[15] = {0};

    uint32_t flags[16] = {0};
    uint64_t flags_out[16] = {0};

/*
    std::chrono::high_resolution_clock::time_point f1 = std::chrono::high_resolution_clock::now();
    std::ifstream f("/media/mdrk/NVMe/NA12886_S1_flags.bin", std::ios::binary | std::ios::in | std::ios::ate);
    if(f.is_open() == false) exit(1);
    uint64_t sz_file = f.tellg();
    f.seekg(0);
    uint64_t last_offset = 0;
    uint64_t n_read_tot = 0;
    
    while(true) {
        f.read(reinterpret_cast<char*>(vals), 1000000*sizeof(uint16_t));
        uint32_t n_read = f.tellg() - last_offset;
        if(f.tellg() == -1) n_read = sz_file - last_offset;
        n_read_tot += n_read;

        
        uint32_t avx2_timing_naive = flag_stats_avx2(vals, n_read, flags);
        for(int i = 0; i < 16; ++i) flags_out[i] += flags[i];

        last_offset = f.tellg();
        if(last_offset == sz_file || f.good() == false) break;
    }

    std::chrono::high_resolution_clock::time_point f2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(f2 - f1);
    std::cerr << "Time elapsed=" << time_span.count() << " -> " << ((n_read_tot*sizeof(uint16_t)) / (1024*1024.0)) / (time_span.count() / 1000000.0) << std::endl;
    std::cerr << "N: " << n_read_tot << " flags:";
    for(int i = 0; i < 16; ++i) std::cerr << " " << flags_out[i];
    std::cerr << std::endl;
*/

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
            // scalar naive
            flag_functions(vals, times, times_local, n);

            

#define MBS(cum) (times_local[cum] == 0 ? 0 : ((n*sizeof(uint16_t)) / (1024*1024.0)) / (times_local[cum] / 1000000.0))
            //std::cout << ranges[r] << "\t" << c << "\t" << times_local[0] << "\t" << times_local[1] << "\t" << times_local[2] << "\t" << times_local[3] << "\t" << times_local[4] << "\t" << times_local[5] << "\t" << times_local[6] << "\t" << times_local[7] << std::endl;
            std::cout << ranges[r] << "\t" << c << "\t" << MBS(0) << "\t" << MBS(1) << "\t" << MBS(2) << "\t" << MBS(3) << "\t" << MBS(4) << "\t" << MBS(5) << "\t" << MBS(6) << "\t" << MBS(7) << "\t" << MBS(8) << "\t" << MBS(9) << "\t" << MBS(10) << "\t" << MBS(11) << "\t" << MBS(12) << std::endl;
#undef MBS
        }
#define AVG(pos) (times[pos] == 0 ? 0 : (double)times[pos]/cycles)
        std::cout << "average times\t" << AVG(0) << "\t" << AVG(1) << "\t" << AVG(2) << "\t" << AVG(3) << "\t" << AVG(4) << "\t" << AVG(5) << "\t" << AVG(6) << "\t" << AVG(7) << "\t" << AVG(8) << "\t" << AVG(9) << "\t" << AVG(10)<< "\t" << AVG(11) << "\t" << AVG(12) << std::endl;
#define INTS_SEC(cum) (times[cum] == 0 ? 0 : ((n*sizeof(uint16_t)) / (1024*1024.0)) / (AVG(cum) / 1000000.0))
        std::cout << "MB/s\t" << 0 << "\t" << INTS_SEC(1) << "\t" << INTS_SEC(2) << "\t" << INTS_SEC(3) << "\t" << INTS_SEC(4) << "\t" << INTS_SEC(5) << "\t" << INTS_SEC(6) << "\t" << INTS_SEC(7) << "\t" << INTS_SEC(8) << "\t" << INTS_SEC(9) << "\t" << INTS_SEC(10) << "\t" << INTS_SEC(11) << "\t" << INTS_SEC(12) << std::endl;
#undef AVG
#undef INTS_SEC

        memset(times, 0, sizeof(uint64_t)*15);
    }

    delete[] vals;
}

int main(int argc, char **argv) {
    if(argc == 1) flag_test(100000000, 10);
    else if(argc == 2) flag_test(std::atoi(argv[1]), 10);
    else if(argc == 3) flag_test(std::atoi(argv[1]), std::atoi(argv[2]));
    else return(1);
    return(0);
}
