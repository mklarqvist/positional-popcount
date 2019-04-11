#include <iostream>//out streams
#include <random>//random generator (c++11)
#include <chrono>//time (c++11)
#include <cassert>//assert
#include <cstring>//memset

#include "pospopcnt.h"

inline void* aligned_malloc(size_t size, size_t align) {
    void* result;
#if __STDC_VERSION__ >= 201112L
    result = aligned_alloc(align, size);
#elif _POSIX_VERSION >= 200112L
     if (posix_memalign(&result, align, size)) result = nullptr;
#else 
#ifdef _MSC_VER 
    result = _aligned_malloc(size, align);
#else
    result = new uint8_t*[size];
#endif

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

void generate_random_data(uint16_t* data, uint32_t n) {
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    std::uniform_int_distribution<uint16_t> distr(0, std::numeric_limits<uint16_t>::max()-1); // right inclusive

    for (int i = 0; i < n; ++i) {
        data[i] = distr(eng);
    }
}

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock::time_point clockdef;

int pospopcnt_u16_wrapper(pospopcnt_u16_method_type f, int id, int iterations,
                          uint16_t* data, uint32_t n, 
                          bench_unit& unit) 
{
    // Set counters to 0.
    uint32_t counters[16] = {0};
    uint32_t flags_truth[16] = {0};

    uint32_t cycles_low = 0, cycles_high = 0;
    uint32_t cycles_low1 = 0, cycles_high1 = 0;
    // Start timer.
    
    //const uint64_t cpu_cycles_before = get_cpu_cycles();

    std::vector<uint64_t> clocks;
    std::vector<uint32_t> times;

#ifndef _MSC_VER
// Intel guide:
// @see: https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/ia-32-ia-64-benchmark-code-execution-paper.pdf
asm   volatile ("CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx"); 
asm   volatile("RDTSCP\n\t"
               "mov %%edx, %0\n\t"
               "mov %%eax, %1\n\t"
               "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx"); 
asm   volatile ("CPUID\n\t"
                "RDTSC\n\t"
                "mov %%edx, %0\n\t"
                "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx"); 
asm   volatile("RDTSCP\n\t"
               "mov %%edx, %0\n\t"
               "mov %%eax, %1\n\t"
               "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx");
#endif

    for (int i = 0; i < iterations; ++i) {
        memset(counters, 0, sizeof(uint32_t)*16);
        memset(flags_truth, 0, sizeof(uint32_t)*16);
        generate_random_data(data, n);

        pospopcnt_u16_scalar_naive(data, n, flags_truth);

        clockdef t1 = std::chrono::high_resolution_clock::now();

#ifdef __linux__ 
    // unsigned long flags;
    // preempt_disable(); /*we disable preemption on our CPU*/
    // raw_local_irq_save(flags); /*we disable hard interrupts on our CPU*/  
    /*at this stage we exclusively own the CPU*/ 
#endif

#ifndef _MSC_VER 
    asm   volatile ("CPUID\n\t"
                    "RDTSC\n\t"
                    "mov %%edx, %0\n\t"
                    "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx");
#endif
    // Call argument subroutine pointer.
    (*f)(data, n, counters);

#ifndef _MSC_VER 
    asm   volatile("RDTSCP\n\t"
                   "mov %%edx, %0\n\t"
                   "mov %%eax, %1\n\t"
                   "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx");
#endif
#ifdef __linux__ 
        // raw_local_irq_restore(flags);/*we enable hard interrupts on our CPU*/
        // preempt_enable();/*we enable preemption*/
#endif

        clockdef t2 = std::chrono::high_resolution_clock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

        assert_truth(counters, flags_truth);
        // std::cerr << cycles_low <<"-" << cycles_high << " and " << cycles_low1 << "-" << cycles_high1 << std::endl;
        // std::cerr << "diff=" << (cycles_low1-cycles_low) << "->" << (cycles_low1-cycles_low)/(double)n << std::endl;
        uint64_t start = ( ((uint64_t)cycles_high  << 32) | cycles_low  );
        uint64_t end   = ( ((uint64_t)cycles_high1 << 32) | cycles_low1 );

        clocks.push_back(end - start);
        times.push_back(time_span.count());
    }

    uint64_t tot_cycles = 0, tot_time = 0;
    uint64_t min_c = std::numeric_limits<uint64_t>::max(), max_c = 0;
    for (int i = 0; i < clocks.size(); ++i) {
        tot_cycles += clocks[i];
        tot_time += times[i];
        min_c = std::min(min_c, clocks[i]);
        max_c = std::max(max_c, clocks[i]);
    }
    double mean_cycles = tot_cycles / (double)clocks.size();
    uint32_t mean_time = tot_time / (double)clocks.size();

    double variance = 0, stdDeviation = 0, mad = 0;
    for(int i = 0; i < clocks.size(); ++i) {
        variance += pow(clocks[i] - mean_cycles, 2);
        mad += std::abs(clocks[i] - mean_cycles);
    }
    mad /= clocks.size();
    variance /= clocks.size();
    stdDeviation = sqrt(variance);

    std::cout << pospopcnt_u16_method_names[id] << "\t" << n << "\t" << 
        mean_cycles << "\t" <<
        min_c << "(" << min_c/mean_cycles << ")" << "\t" << 
        max_c << "(" << max_c/mean_cycles << ")" << "\t" <<
        stdDeviation << "\t" << 
        mad << "\t" << 
        mean_time << "\t" << 
        mean_cycles / n << "\t" << 
        ((n*sizeof(uint16_t)) / (1024*1024.0)) / (mean_time / 1000000000.0) << std::endl;

    // End timer and update times.
    //uint64_t cpu_cycles_after = get_cpu_cycles();
    
    unit.times += mean_time;
    unit.times_local = mean_time;
    //unit.cycles += (cpu_cycles_after - cpu_cycles_before);
    //unit.cycles_local = (cpu_cycles_after - cpu_cycles_before);
    unit.cycles += mean_cycles;
    unit.cycles_local = mean_cycles;
    for (int i = 0; i < 16; ++i) unit.valid += counters[i];

    //std::cerr << cycles_low <<"-" << cycles_high << " and " << cycles_low1 << "-" << cycles_high1 << std::endl;
    //std::cerr << "diff=" << (cycles_low1-cycles_low) << "->" << (cycles_low1-cycles_low)/(double)n << std::endl;
    return 0;
}

void benchmark(uint16_t* vals, std::vector<bench_unit>& units, const uint32_t n, int iterations) {
    // Cycle over algorithms.
    for(int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) {
    // for(int i = 2; i < 24; ++i) {
        pospopcnt_u16_wrapper(get_pospopcnt_u16_method(PPOPCNT_U16_METHODS(i)),i,iterations,vals,n,units[i]);
    }
}

void flag_test(uint32_t n, uint32_t cycles = 1) {
    std::cerr << "Generating " << n << " flags. (" << n*sizeof(uint16_t) / 1024 << "kb) repeated " << cycles << " times." << std::endl;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    // Memory align input data.
    uint16_t* vals = (uint16_t*)aligned_malloc(n*sizeof(uint16_t), POSPOPCNT_SIMD_ALIGNMENT);
    std::vector<bench_unit> units(64);
    std::cout << "Algorithm\tNumIntegers\tMeanCycles\tMinCycles\tMaxCycles\tStdDeviationCycles\tMeanAbsDev\tMeanTime(nanos)\tMeanCyclesInt\tThroughput(MB/s)" << std::endl;
    benchmark(vals, units, n, cycles);
        
    // Cleanup.
    aligned_free(vals);
}

int main(int argc, char **argv) {
    if(argc == 1)      flag_test(1000000, 500);
    else if(argc == 2) flag_test(std::atoi(argv[1]), 500);
    else if(argc == 3) flag_test(std::atoi(argv[1]), std::atoi(argv[2]));
    else return(1);
    return(0);
}
