#include <iostream>//out streams
#include <random>//random generator (c++11)
#include <chrono>//time (c++11)

#include "fast_flagstats.h"

bool assert_truth(uint32_t* vals, uint32_t* truth) {
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

int pospopcnt_u16_wrapper(int(f)(const uint16_t* data, uint32_t n, uint32_t* flags), const uint16_t* data, uint32_t n, uint32_t* flags, uint64_t& times, uint64_t& times_local) {
    // Set counters to 0.
    memset(flags, 0, sizeof(uint32_t)*16);

    // Start timer.
    clockdef t1 = std::chrono::high_resolution_clock::now();

    // Call provided subroutine pointer.
    (*f)(data, n, flags);

    // End timer.
    clockdef t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    times += time_span.count();
    times_local = time_span.count();
    return 0;
}

void flag_functions(uint16_t* vals, uint64_t* times, uint64_t* times_local, const uint32_t n) {
    uint32_t truth[16];
    uint32_t flags[16];

    // Truth-set from naive scalar subroutine.
    pospopcnt_u16_wrapper(&pospopcnt_u16_scalar_naive,vals,n,truth,times[1],times_local[1]);
    
    pospopcnt_u16_wrapper(&pospopcnt_u16_scalar_partition,vals,n,flags,times[2],times_local[2]);
    assert_truth(flags, truth);
    
    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2,vals, n, flags,times[3],times_local[3]);
    assert_truth(flags, truth);
    
    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_popcnt,vals, n, flags,times[4],times_local[4]);
    assert_truth(flags, truth);
    
    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_single,vals, n, flags,times[5],times_local[5]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_mula_unroll4,vals, n, flags,times[16],times_local[16]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_naive_counter,vals, n, flags,times[6],times_local[6]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_lemire,vals, n, flags,times[13],times_local[13]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_lemire2,vals, n, flags,times[14],times_local[14]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_mula,vals, n, flags,times[15],times_local[15]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_sse_single,vals, n, flags,times[7],times_local[7]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_mula_unroll8,vals, n, flags,times[17],times_local[17]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_hist1x4,vals, n, flags,times[8],times_local[8]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx512_popcnt,vals, n, flags,times[9],times_local[9]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx512_popcnt32_mask,vals, n, flags,times[10],times_local[10]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx2_mula_unroll16,vals, n, flags,times[18],times_local[18]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx512,vals, n, flags,times[11],times_local[11]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_avx512_popcnt64_mask,vals, n, flags,times[12],times_local[12]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_sse_mula,vals, n, flags,times[19],times_local[19]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_sse_mula_unroll4,vals, n, flags,times[20],times_local[20]);
    assert_truth(flags, truth);

    pospopcnt_u16_wrapper(&pospopcnt_u16_sse_mula_unroll8,vals, n, flags,times[21],times_local[21]);
    assert_truth(flags, truth);
}

void flag_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating flags: " << n << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    //uint16_t* vals = new uint16_t[n];
    uint16_t* vals;
    // Memory align input data.
    assert(!posix_memalign((void**)&vals, SIMD_ALIGNMENT, n*sizeof(uint16_t)));

    uint64_t times[64] = {0};
    uint64_t times_local[64] = {0};

    const std::vector<uint32_t> ranges = {8, 16, 64, 256, 512, 1024, 4096, 65536};
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
            for(int i = 1; i < 22; ++i) std::cout << '\t' << MBS(i);
            std::cout << "\nCycles\t" << ranges[r] << "\t" << c;
            for(int i = 1; i < 22; ++i) std::cout << '\t' << SPEED(i);
            std::cout << std::endl;
#undef MBS
#undef SPEED
        }
#define AVG(pos) (times[pos] == 0 ? 0 : (double)times[pos]/cycles)
        std::cout << "Times\t" << AVG(1) << "\t" << AVG(2) << "\t" << AVG(3) << "\t" << AVG(4) << "\t" << AVG(5) << "\t" << AVG(6) << "\t" << AVG(7) << "\t" << AVG(8) << "\t" << AVG(9) << "\t" << AVG(10)<< "\t" << AVG(11) << "\t" << AVG(12) << "\t" << AVG(13) << "\t" << AVG(14) << "\t" << AVG(15) << "\t" << AVG(16) << "\t" << AVG(17) << "\t" << AVG(18) << std::endl;
#define INTS_SEC(cum) (times[cum] == 0 ? 0 : ((n*sizeof(uint16_t)) / (1024*1024.0)) / (AVG(cum) / 1000000.0))
#define AVG_CYCLES(pos) (times[pos] == 0 ? 0 : (MHZ * (AVG(pos) / 1000000.0) / n))
        std::cout << "MB/s\t" << INTS_SEC(1) << "\t" << INTS_SEC(2) << "\t" << INTS_SEC(3) << "\t" << INTS_SEC(4) << "\t" << INTS_SEC(5) << "\t" << INTS_SEC(6) << "\t" << INTS_SEC(7) << "\t" << INTS_SEC(8) << "\t" << INTS_SEC(9) << "\t" << INTS_SEC(10) << "\t" << INTS_SEC(11) << "\t" << INTS_SEC(12) << "\t" << INTS_SEC(13) << "\t" << INTS_SEC(14) << "\t" << INTS_SEC(15) << "\t" << INTS_SEC(16) << "\t" << INTS_SEC(17) << "\t" << INTS_SEC(18) << std::endl;
        std::cout << "Cycles/int\t" << AVG_CYCLES(1) << "\t" << AVG_CYCLES(2) << "\t" << AVG_CYCLES(3) << "\t" << AVG_CYCLES(4) << "\t" << AVG_CYCLES(5) << "\t" << AVG_CYCLES(6) << "\t" << AVG_CYCLES(7) << "\t" << AVG_CYCLES(8) << "\t" << AVG_CYCLES(9) << "\t" << AVG_CYCLES(10) << "\t" << AVG_CYCLES(11) << "\t" << AVG_CYCLES(12) << "\t" << AVG_CYCLES(13) << "\t" << AVG_CYCLES(14) << "\t" << AVG_CYCLES(15) << "\t" << AVG_CYCLES(16) << "\t" << AVG_CYCLES(17) << "\t" << AVG_CYCLES(18) << std::endl;
#undef AVG
#undef INTS_SEC

        memset(times, 0, sizeof(uint64_t)*64);
    }

    // Cleanup.
    delete[] vals;
}

int main(int argc, char **argv) {
    if(argc == 1) flag_test(100000000, 10);
    else if(argc == 2) flag_test(std::atoi(argv[1]), 10);
    else if(argc == 3) flag_test(std::atoi(argv[1]), std::atoi(argv[2]));
    else return(1);
    return(0);
}
