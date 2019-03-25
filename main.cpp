#include <iostream>//out streams
#include <random>//random generator (c++11)
#include <chrono>//time (c++11)
#include <cassert>//assert
#include <cstring>//memset

#include "fast_flagstats.h"

inline void* aligned_malloc(size_t size, size_t align) {
    void* result;
#ifdef _MSC_VER 
    result = _aligned_malloc(size, align);
#else 
     if(posix_memalign(&result, align, size)) result = 0;
#endif
    return result;
}

inline void aligned_free(void* ptr) {
#ifdef _MSC_VER 
      _aligned_free(ptr);
#else 
      free(ptr);
#endif
}

bool assert_truth(uint32_t* vals, uint32_t* truth) {
    uint64_t n_all = 0;
    for(int i = 0; i < 16; ++i) n_all += vals[i];
    if(n_all == 0) return true;
    
    for(int i = 0; i < 16; ++i) {
        assert(vals[i] == truth[i]);
    }
    return true;
}

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock::time_point clockdef;

int pospopcnt_u16_wrapper(pospopcnt_u16_method_type f, 
                          const uint16_t* data, uint32_t n, uint32_t* flags, 
                          uint64_t& times, uint64_t& times_local) 
{
    // Set counters to 0.
    memset(flags, 0, sizeof(uint32_t)*16);

    // Start timer.
    clockdef t1 = std::chrono::high_resolution_clock::now();

    // Call argument subroutine pointer.
    (*f)(data, n, flags);

    // End timer and update times.
    clockdef t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    times += time_span.count();
    times_local = time_span.count();
    return 0;
}

void benchmark(uint16_t* vals, uint64_t* times, uint64_t* times_local, const uint32_t n) {
    uint32_t truth[16];
    uint32_t flags[16];

    // Truth-set from naive scalar subroutine.
    pospopcnt_u16_wrapper(&pospopcnt_u16_scalar_naive,vals,n,truth,times[1],times_local[1]);
    
    for(int i = 2; i < 26; ++i) {
        pospopcnt_u16_wrapper(get_pospopcnt_u16_method(PPOPCNT_U16_METHODS(i)),vals,n,flags,times[i],times_local[i]);
        assert_truth(flags, truth);
    }
}

void flag_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating " << n << " flags." << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    // Memory align input data.
    uint16_t* vals = (uint16_t*)aligned_malloc(n*sizeof(uint16_t), SIMD_ALIGNMENT);

    uint64_t times[64] = {0};
    uint64_t times_local[64] = {0};

    const std::vector<uint32_t> ranges = {8, 16, 64, 256, 512, 1024, 4096, 65536};
    for(int r = 0; r < ranges.size(); ++r) {
        std::uniform_int_distribution<uint16_t> distr(1, ranges[r]); // right inclusive

        for(int c = 0; c < cycles; ++c) {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            // Generate random data every iteration.
            for(uint32_t i = 0; i < n; ++i) {
                vals[i] = distr(eng);
            }

            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            times[0] += time_span.count();
            times_local[0] = time_span.count();

            // Start benchmarking.
            benchmark(vals, times, times_local, n);

#define MHZ 2800000000.0
#define MBS(cum) (times_local[cum] == 0 ? 0 : ((n*sizeof(uint16_t)) / (1024*1024.0)) / (times_local[cum] / 1000000.0))
#define SPEED(cum) (times_local[cum] == 0 ? 0 : (MHZ * (times_local[cum] / 1000000.0) / n))
            std::cout << "MBS\t" << ranges[r] << "\t" << c;
            for(int i = 1; i < 26; ++i) std::cout << '\t' << MBS(i);
            std::cout << std::endl;
            std::cout << "Cycles\t" << ranges[r] << "\t" << c;
            for(int i = 1; i < 26; ++i) std::cout << '\t' << SPEED(i);
            std::cout << std::endl;
#undef MBS
#undef SPEED
        }
#define AVG(pos) (times[pos] == 0 ? 0 : (double)times[pos]/cycles)
        std::cout << "Times\t" << ranges[r] << "\t" << "F";
        for (int i = 1; i < 26; ++i) std::cout << '\t' << AVG(i);
        std::cout << std::endl;

#define INTS_SEC(cum) (times[cum] == 0 ? 0 : ((n*sizeof(uint16_t)) / (1024*1024.0)) / (AVG(cum) / 1000000.0))
#define AVG_CYCLES(pos) (times[pos] == 0 ? 0 : (MHZ * (AVG(pos) / 1000000.0) / n))
        std::cout << "MB/s\t" << ranges[r] << "\t" << "F";
        for (int i = 1; i < 26; ++i) std::cout << "\t" << INTS_SEC(i);
        std::cout << std::endl;
        std::cout << "Cycles/int\t" << ranges[r] << "\t" << "F";
        for (int i = 1; i < 26; ++i) std::cout << "\t" << AVG_CYCLES(i);
        std::cout << std::endl;
        
#undef AVG
#undef INTS_SEC
#undef MHZ

        memset(times, 0, sizeof(uint64_t)*64);
    }

    // Cleanup.
    aligned_free(vals);
}

int main(int argc, char **argv) {
    if(argc == 1)      flag_test(100000000, 10);
    else if(argc == 2) flag_test(std::atoi(argv[1]), 10);
    else if(argc == 3) flag_test(std::atoi(argv[1]), std::atoi(argv[2]));
    else return(1);
    return(0);
}
