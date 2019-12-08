#include <iostream>//out streams
#include <random>//random generator (c++11)
#include <chrono>//time (c++11)
#include <cassert>//assert
#include <cstring>//memset
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <iomanip>

#ifdef _MSC_VER
# include <intrin.h>
#else
# include <x86intrin.h>
#endif

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

uint64_t get_cpu_cycles() {
    uint64_t result;
#ifndef _MSC_VER
    __asm__ volatile(".byte 15;.byte 49;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"
                     (result)::"%rdx");
#else
    result = __rdtsc();
#endif
    return result;
};

bool assert_truth(uint32_t* vals, uint32_t* truth) {
    uint64_t n_all = 0;
    for(int i = 0; i < 16; ++i) n_all += vals[i];
    if(n_all == 0) return true;
    
    // temp
    bool fail = false;

    for(int i = 0; i < 16; ++i) {
        if (vals[i] != truth[i]) {
            fail = true;
        }
    }

    if (fail) {
        std::cout << "FAILURE:" << std::endl;
        for (int i = 0; i < 16; ++i) {
            std::cout << truth[i] << "\t" << vals[i];
            if (truth[i] != vals[i])
                std::cout << " ***";

            std::cout << '\n';
        }
    }

    return true;
}

template <typename IntegerType>
void generate_random_data(IntegerType* data, size_t n) {
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator

    static_assert(sizeof(int32_t) >= sizeof(IntegerType), "please adjust uniform_int_distribution construction");

    std::uniform_int_distribution<uint32_t> distr(0, std::numeric_limits<IntegerType>::max()-1); // right inclusive

    for (int i = 0; i < n; ++i) {
        data[i] = distr(eng);
    }
}

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock::time_point clockdef;

struct Measurement {
    struct {
        uint64_t total;
        double   mean;
    } time;

    struct {
        uint64_t total;
        uint64_t min;
        uint64_t max;
        double   mean;
        double   mad;
        double   variance;
        double   stddev;
    } cycles;

    size_t count; // input items
    size_t size; // input size in bytes (items * sizeof(item))
    double throughput; // MB/s
};

class StatisticsBuilder {
    std::vector<uint32_t> times;
    std::vector<uint64_t> clocks;
public:
    StatisticsBuilder(const size_t estimated_size) {
        assert(estimated_size > 0);
        times.reserve(estimated_size);
        clocks.reserve(estimated_size);
    }

    void add_record(clockdef start, clockdef end, uint64_t rdtsc_start, uint64_t rdtsc_end) {
        const auto time_span = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        times.push_back(time_span.count());
        clocks.push_back(rdtsc_end - rdtsc_start);
    }

    Measurement calculate() const {
        Measurement meas;

        const double n = clocks.size();

        meas.time.total   = std::accumulate(times.begin(), times.end(), 0);
        meas.time.mean    = meas.time.mean / n;

        meas.cycles.total = std::accumulate(clocks.begin(), clocks.end(), 0);
        meas.cycles.min   = *std::min_element(clocks.begin(), clocks.end());
        meas.cycles.max   = *std::max_element(clocks.begin(), clocks.end());
        meas.cycles.mean  = meas.cycles.total / n;

        const double variance   = std::accumulate(clocks.begin(), clocks.end(), 0.0,
                                  [&meas](double sum, uint64_t clocks) {
                                      return sum + pow(clocks - meas.cycles.mean, 2.0);
                                  });
        const double mad        = std::accumulate(clocks.begin(), clocks.end(), 0.0,
                                  [&meas](double sum, uint64_t clocks) {
                                      return sum + std::abs(clocks - meas.cycles.mean);
                                  });

        meas.cycles.mad      = mad / n;
        meas.cycles.variance = variance / n;
        meas.cycles.stddev   = sqrt(meas.cycles.variance);

        return meas;
    }
};


template <typename pospopcnt_function_type, typename ItemType>
Measurement pospopcnt_wrapper(
    const char* method_name,
    pospopcnt_function_type measured_function,
    pospopcnt_function_type reference_function,
    int iterations,
    ItemType* data,
    size_t n)
{
    static_assert(std::is_unsigned<ItemType>::value, "ItemType must be an unsigned type");

    // Set counters to 0.
    uint32_t counters[16] = {0};
    uint32_t flags_truth[16] = {0};

    uint32_t cycles_low = 0, cycles_high = 0;
    uint32_t cycles_low1 = 0, cycles_high1 = 0;
    // Start timer.
    
    StatisticsBuilder stats(iterations);

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
        memset(counters, 0, sizeof(counters));
        memset(flags_truth, 0, sizeof(flags_truth));
        generate_random_data(data, n);

        reference_function(data, n, flags_truth);

        const clockdef t1 = std::chrono::high_resolution_clock::now();

#ifndef _MSC_VER 
    asm   volatile ("CPUID\n\t"
                    "RDTSC\n\t"
                    "mov %%edx, %0\n\t"
                    "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "%rax", "%rbx", "%rcx", "%rdx");
#endif
    // Call argument subroutine pointer.
    measured_function(data, n, counters);

#ifndef _MSC_VER 
    asm   volatile("RDTSCP\n\t"
                   "mov %%edx, %0\n\t"
                   "mov %%eax, %1\n\t"
                   "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "%rax", "%rbx", "%rcx", "%rdx");
#endif

        const clockdef t2 = std::chrono::high_resolution_clock::now();

        assert_truth(counters, flags_truth);

#define RDTSC_u64(high, low) (((uint64_t)(high) << 32)|(low))
        stats.add_record(t1, t2, RDTSC_u64(cycles_high, cycles_low), RDTSC_u64(cycles_high1, cycles_low1));
#undef RDTSC_u64
    }

    auto meas       = stats.calculate();
    meas.count      = n;
    meas.size       = meas.count * sizeof(ItemType);
    meas.throughput = (meas.size / (1024*1024.0)) / (meas.time.mean / 1000000000.0);

    return meas;
}

struct Parameters {
    size_t items_count;
    size_t iterations;
    std::string filter;
};

class MeasurementsPrinter {
    bool header_printed = false;
    bool only_time;
    std::ostream& out;
    
public:
    MeasurementsPrinter(std::ostream& out, bool only_time)
        : out(out)
        , only_time(only_time) {}

    void print(const char* method_name, const Measurement& meas) {
        if (only_time)
            print_short(method_name, meas);
        else {
            print_header();
            print_all(method_name, meas);
        }
    }

private:
    void print_all(const char* method_name, const Measurement& meas) {
        out << method_name << '\t'
            << meas.count << '\t'
            << meas.cycles.mean << '\t'
            << meas.cycles.min << "(" << meas.cycles.min / meas.cycles.mean << ")" << '\t'
            << meas.cycles.max << "(" << meas.cycles.max / meas.cycles.mean << ")" << '\t'
            << meas.cycles.stddev << '\t'
            << meas.cycles.mad << '\t'
            << meas.time.mean << '\t'
            << meas.cycles.mean / meas.count << '\t'
            << meas.throughput << '\n';
    }

    void print_short(const char* method_name, const Measurement& meas) {
        out.width(50);
        out << std::left << method_name << ' ';
        out.width(10);
        out.setf(std::ios::fixed, std::ios::floatfield);
        out << std::right << std::setprecision(2) << meas.cycles.mean << '\n';
    }

    void print_header() {
        if (header_printed)
            return;

        std::cout << "Algorithm\tNumIntegers\tMeanCycles\tMinCycles\tMaxCycles\tStdDeviationCycles\tMeanAbsDev\tMeanTime(nanos)\tMeanCyclesInt\tThroughput(MB/s)" << std::endl;
        header_printed = true;
    }
};

void benchmark(uint16_t* vals, const Parameters& params) {
    // Cycle over algorithms.

    MeasurementsPrinter printer(std::cout, true);
    for(int i = 1; i < PPOPCNT_NUMBER_METHODS; ++i) {
        const char* name = pospopcnt_u16_method_names[i];
        if (std::string(name).find(params.filter) == std::string::npos)
            continue;

        auto method = get_pospopcnt_u16_method(PPOPCNT_U16_METHODS(i));
        auto reference = pospopcnt_u16_scalar_naive;
        const auto meas = pospopcnt_wrapper<pospopcnt_u16_method_type, uint16_t>(
            name, method, reference, params.iterations, vals, params.items_count);

        printer.print(name, meas);
    }
    for(int i = 0; i < PPOPCNT_U8_NUMBER_METHODS; ++i) {
        const char* name = pospopcnt_u8_method_names[i];
        if (std::string(name).find(params.filter) == std::string::npos)
            continue;

        auto method = get_pospopcnt_u8_method(PPOPCNT_U8_METHODS(i));
        auto reference = pospopcnt_u8_scalar_naive;
        const auto meas = pospopcnt_wrapper<pospopcnt_u8_method_type, uint8_t>(
            name, method, reference, params.iterations, (uint8_t*)vals, params.items_count);

        printer.print(name, meas);
    }
}

void flag_test(const Parameters& params) {
    std::cerr << "Will test " << params.items_count << " flags. (" << params.items_count*sizeof(uint16_t) / 1024 << "kB) repeated " << params.iterations << " times." << std::endl;

    // Memory align input data.
    uint16_t* vals = (uint16_t*)aligned_malloc(params.items_count*sizeof(uint16_t), POSPOPCNT_SIMD_ALIGNMENT);
        
    benchmark(vals, params);
    // Cleanup.
    aligned_free(vals);
}

bool parse_args(int argc, char* argv[], Parameters& params);

int main(int argc, char **argv) {
    Parameters params;

    if (!parse_args(argc, argv, params)) {
        std::cout << "Usage: " << argv[0] << "[input-size [iterations-count [function-filter]]]" << '\n';
        return EXIT_FAILURE;
    }

    flag_test(params);
    return EXIT_SUCCESS;
}

bool parse_args(int argc, char* argv[], Parameters& params) {
    params.items_count = 1000000;
    params.iterations  = 500;
    params.filter      = "";

    if (argc > 1)
        params.items_count = std::atoi(argv[1]);

    if (argc > 2)
        params.iterations = std::atoi(argv[2]);

    if (argc > 3)
        params.filter = argv[3];

    return (argc <= 4);
}
