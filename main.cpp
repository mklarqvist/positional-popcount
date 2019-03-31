#include <iostream>//out streams
#include <random>//random generator (c++11)
#include <chrono>//time (c++11)
#include <cassert>//assert
#include <cstring>//memset

#include "pospopcnt.h"

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

struct bench_unit {
    bench_unit() : valid(false), cycles(0), cycles_local(0), times(0), times_local(0){}

    bool valid;
    float cycles;
    float cycles_local;
    uint64_t times;
    uint64_t times_local;
};

uint64_t get_cpu_cycles() {
    uint64_t result;
    __asm__ volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"
                     (result)::"%rdx");
    return result;
};

bool assert_truth(uint32_t* vals, uint32_t* truth) {
    uint64_t n_all = 0;
    for(int i = 0; i < 16; ++i) n_all += vals[i];
    if(n_all == 0) return true;
    
    // temp
    bool fail = false;

    for(int i = 0; i < 16; ++i) {
        //assert(vals[i] == truth[i]);
        if (vals[i] != truth[i]) {
            fail = true;
        }
    }

    if (fail) {
        std::cerr << "FAILURE:" << std::endl;
        for (int i = 0; i < 16; ++i) {
            std::cerr << truth[i] << "\t" << vals[i] << std::endl;
        }
    }

    return true;
}

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock::time_point clockdef;

int pospopcnt_u16_wrapper(pospopcnt_u16_method_type f, 
                          const uint16_t* data, uint32_t n, uint32_t* flags, 
                          bench_unit& unit) 
{
    // Set counters to 0.
    memset(flags, 0, sizeof(uint32_t)*16);

    // Start timer.
    clockdef t1 = std::chrono::high_resolution_clock::now();
    const uint64_t cpu_cycles_before = get_cpu_cycles();

    // Call argument subroutine pointer.
    (*f)(data, n, flags);

    // End timer and update times.
    uint64_t cpu_cycles_after = get_cpu_cycles();
    clockdef t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    unit.times += time_span.count();
    unit.times_local = time_span.count();
    unit.cycles += (cpu_cycles_after - cpu_cycles_before);
    unit.cycles_local = (cpu_cycles_after - cpu_cycles_before);
    for (int i = 0; i < 16; ++i) unit.valid += flags[i];
    return 0;
}

void benchmark(uint16_t* vals, std::vector<bench_unit>& units, const uint32_t n) {
    uint32_t truth[16];
    uint32_t flags[16];

    // Truth-set from naive scalar subroutine.
    pospopcnt_u16_wrapper(&pospopcnt_u16_scalar_naive,vals,n,truth,units[1]);
    
    for(int i = 2; i < PPOPCNT_NUMBER_METHODS; ++i) {
        pospopcnt_u16_wrapper(get_pospopcnt_u16_method(PPOPCNT_U16_METHODS(i)),vals,n,flags,units[i]);
        assert_truth(flags, truth);
    }
}

void flag_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating " << n << " flags." << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    // Memory align input data.
    uint16_t* vals = (uint16_t*)aligned_malloc(n*sizeof(uint16_t), SIMD_ALIGNMENT);

    std::vector<bench_unit> units(64);

    std::cout << "Type\tRange\tIteration";
    for (int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) std::cout << "\t" << pospopcnt_u16_method_names[i];
    std::cout << std::endl;

    const std::vector<uint32_t> ranges = {8, 16, 64, 256, 512, 1024, 4096, 65536};
    for (int r = 0; r < ranges.size(); ++r) {
        std::uniform_int_distribution<uint16_t> distr(1, ranges[r]); // right inclusive

        for (int c = 0; c < cycles; ++c) {
            std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

            // Generate random data every iteration.
            for (int i = 0; i < n; ++i) {
                vals[i] = distr(eng);
            }

            std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
            auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            units[0].times += time_span.count();
            units[0].times_local = time_span.count();

            // Start benchmarking.
            benchmark(vals, units, n);

#define MBS(cum) ((n*sizeof(uint16_t)) / (1024*1024.0)) / (units[cum].times_local / 1000000.0)
            std::cout << "MBS\t" << ranges[r] << "\t" << c;
            for(int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) std::cout << '\t' << (units[i].valid ? MBS(i) : 0);
            std::cout << std::endl;
            std::cout << "Cycles\t" << ranges[r] << "\t" << c;
            for(int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) std::cout << '\t' << (units[i].valid ? units[i].cycles_local / float(n) : 0);
            std::cout << std::endl;
#undef MBS
        }
#define AVG(pos) (double)units[pos].times/cycles
        std::cout << "Times\t" << ranges[r] << "\t" << "F";
        for (int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) std::cout << '\t' << AVG(i);
        std::cout << std::endl;

#define INTS_SEC(cum) ((n*sizeof(uint16_t)) / (1024*1024.0)) / (AVG(cum) / 1000000.0)
        std::cout << "MB/s\t" << ranges[r] << "\t" << "F";
        for (int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) std::cout << "\t" << (units[i].valid ? INTS_SEC(i) : 0);
        std::cout << std::endl;
        std::cout << "Cycles/int\t" << ranges[r] << "\t" << "F";
        for (int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) std::cout << "\t" << (units[i].valid ? units[i].cycles / (cycles*float(n)) : 0);
        std::cout << std::endl;
#undef AVG

        units = std::vector<bench_unit>(64);
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
