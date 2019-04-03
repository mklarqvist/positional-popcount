#ifdef __linux__

/* ****************************
*  Definitions
******************************/
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <libgen.h>
#include <random>
#include <string>
#include <vector>

#include "pospopcnt.h"
#include "linux-perf-events.h"
#include "popcnt.h"

pospopcnt_u16_method_type pospopcnt_u16_methods[] = {
    pospopcnt_u16, // higher-level heuristic
    pospopcnt_u16_scalar_naive,
    pospopcnt_u16_scalar_naive_nosimd,
    pospopcnt_u16_scalar_partition,
    pospopcnt_u16_scalar_hist1x4,
    pospopcnt_u16_sse_single,
    pospopcnt_u16_sse_mula,
    pospopcnt_u16_sse_mula_unroll4,
    pospopcnt_u16_sse_mula_unroll8,
    pospopcnt_u16_sse_mula_unroll16,
    pospopcnt_u16_sse_sad,
    pospopcnt_u16_sse_csa,
    pospopcnt_u16_avx2_popcnt,
    pospopcnt_u16_avx2,
    pospopcnt_u16_avx2_naive_counter,
    pospopcnt_u16_avx2_single,
    pospopcnt_u16_avx2_lemire,
    pospopcnt_u16_avx2_lemire2,
    pospopcnt_u16_avx2_mula,
    pospopcnt_u16_avx2_mula_unroll4,
    pospopcnt_u16_avx2_mula_unroll8,
    pospopcnt_u16_avx2_mula_unroll16,
    pospopcnt_u16_avx2_mula3,
    pospopcnt_u16_avx2_csa,
    pospopcnt_u16_avx512,
    pospopcnt_u16_avx512_popcnt32_mask,
    pospopcnt_u16_avx512_popcnt64_mask,
    pospopcnt_u16_avx512_popcnt,
    pospopcnt_u16_avx512_mula,
    pospopcnt_u16_avx512_mula_unroll4,
    pospopcnt_u16_avx512_mula_unroll8,
    pospopcnt_u16_avx512_mula2,
    pospopcnt_u16_avx512_mula3,
    pospopcnt_u16_avx512_csa};

void print16(uint32_t *flags) {
    for (int k = 0; k < 16; k++)
        printf(" %8u ", flags[k]);
    printf("\n");
}

std::vector<unsigned long long>
compute_mins(std::vector< std::vector<unsigned long long> > allresults) {
    if (allresults.size() == 0)
        return std::vector<unsigned long long>();
    
    std::vector<unsigned long long> answer = allresults[0];
    
    for (size_t k = 1; k < allresults.size(); k++) {
        assert(allresults[k].size() == answer.size());
        for (size_t z = 0; z < answer.size(); z++) {
            if (allresults[k][z] < answer[z])
                answer[z] = allresults[k][z];
        }
    }
    return answer;
}

std::vector<double>
compute_averages(std::vector< std::vector<unsigned long long> > allresults) {
    if (allresults.size() == 0)
        return std::vector<double>();
    
    std::vector<double> answer(allresults[0].size());
    
    for (size_t k = 0; k < allresults.size(); k++) {
        assert(allresults[k].size() == answer.size());
        for (size_t z = 0; z < answer.size(); z++) {
            answer[z] += allresults[k][z];
        }
    }

    for (size_t z = 0; z < answer.size(); z++) {
        answer[z] /= allresults.size();
    }
    return answer;
}

/**
 * @brief 
 * 
 * @param n          Number of integers.
 * @param iterations Number of iterations.
 * @param fn         Target function pointer.
 * @param verbose    Flag enabling verbose output.
 * @return           Returns true if the results are correct. Returns false if the results
 *                   are either incorrect or the target function is not supported.
 */
bool benchmark(uint32_t n, uint32_t iterations, pospopcnt_u16_method_type fn, bool verbose, bool test) {
    std::vector<int> evts;
    std::vector<uint16_t> vdata(n);
    evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
    evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
    evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
    evts.push_back(PERF_COUNT_HW_CACHE_REFERENCES);
    evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
    LinuxEvents<PERF_TYPE_HARDWARE> unified(evts);
    std::vector<unsigned long long> results; // tmp buffer
    std::vector< std::vector<unsigned long long> > allresults;
    results.resize(evts.size());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);

    bool isok = true;
    for (uint32_t i = 0; i < iterations; i++) {
        for (size_t k = 0; k < vdata.size(); k++) {
            vdata[k] = dis(gen); // random init.
        }
        uint32_t correctflags[16] = {0};
        pospopcnt_u16_scalar_naive(vdata.data(), vdata.size(), correctflags); // this is our gold standard
        uint32_t flags[16] = {0};
        
        unified.start();
        fn(vdata.data(), vdata.size(), flags);
        unified.end(results);

        uint64_t tot_obs = 0;
        for (size_t k = 0; k < 16; ++k) tot_obs += flags[k];
        if (tot_obs == 0) { // when a method is not supported it returns all zero
            return false;
        }

        for (size_t k = 0; k < 16; k++) {
            if (correctflags[k] != flags[k]) {
                if (test) {
                    printf("bug:\n");
                    printf("expected : ");
                    print16(correctflags);
                    printf("got      : ");
                    print16(flags);
                    return false;
                } else {
                    isok = false;
                }
            }
        }
        allresults.push_back(results);
    }

    std::vector<unsigned long long> mins = compute_mins(allresults);
    std::vector<double> avg = compute_averages(allresults);
    
    if (verbose) {
        printf("instructions per cycle %4.2f, cycles per 16-bit word:  %4.3f, "
               "instructions per 16-bit word %4.3f \n",
                double(mins[1]) / mins[0], double(mins[0]) / n, double(mins[1]) / n);
        // first we display mins
        printf("min: %8llu cycles, %8llu instructions, \t%8llu branch mis., %8llu "
               "cache ref., %8llu cache mis.\n",
                mins[0], mins[1], mins[2], mins[3], mins[4]);
        printf("avg: %8.1f cycles, %8.1f instructions, \t%8.1f branch mis., %8.1f "
               "cache ref., %8.1f cache mis.\n",
                avg[0], avg[1], avg[2], avg[3], avg[4]);
    } else {
        printf("cycles per 16-bit word:  %4.3f \n", double(mins[0]) / n);
        // printf("%4.3f \n", double(mins[0]) / n);
    }

    return isok;
}

void measurepopcnt(uint32_t n, uint32_t iterations, bool verbose) {
    std::vector<int> evts;
    std::vector<uint16_t> vdata(n);
    evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
    evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
    evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
    evts.push_back(PERF_COUNT_HW_CACHE_REFERENCES);
    evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
    LinuxEvents<PERF_TYPE_HARDWARE> unified(evts);
    std::vector<unsigned long long> results; // tmp buffer
    std::vector< std::vector<unsigned long long> > allresults;
    results.resize(evts.size());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 0xFFFF);

#if POSPOPCNT_SIMD_VERSION >= 6    
    n = vdata.size() / (512 / 16) * (512 / 16);
#elif POSPOPCNT_SIMD_VERSION >= 5
    n = vdata.size() / (256 / 16) * (256 / 16);
#endif
    for (uint32_t i = 0; i < iterations; i++) {
        for (size_t k = 0; k < vdata.size(); k++) {
            vdata[k] = dis(gen); // random init.
        }
#if POSPOPCNT_SIMD_VERSION >= 6        
        uint64_t expected = popcnt_harley_seal((const __m512i*) vdata.data(), vdata.size() / (512 / 16));       
        unified.start();
        uint64_t measured = popcnt_harley_seal((const __m512i*) vdata.data(), vdata.size() / (512 / 16));
        unified.end(results);
#elif POSPOPCNT_SIMD_VERSION >= 5
        uint64_t expected = popcnt_avx2((const __m256i*) vdata.data(), vdata.size() / (256 / 16));
        unified.start();
        uint64_t measured = popcnt_avx2((const __m256i*) vdata.data(), vdata.size() / (256 / 16));
        unified.end(results);
#endif
        assert(measured == expected);
        allresults.push_back(results);
    }

    std::vector<unsigned long long> mins = compute_mins(allresults);
    std::vector<double> avg = compute_averages(allresults);
#if POSPOPCNT_SIMD_VERSION >= 6 
    printf("%-40s\t","avx512popcnt");    
#elif POSPOPCNT_SIMD_VERSION >= 5
    printf("%-40s\t","avx256popcnt");  
#endif
    if (verbose) {
        printf("instructions per cycle %4.2f, cycles per 16-bit word:  %4.3f, "
                "instructions per 16-bit word %4.3f \n",
                double(mins[1]) / mins[0], double(mins[0]) / n / 4, double(mins[1]) / n / 4);
        // first we display mins
        printf("min: %8llu cycles, %8llu instructions, \t%8llu branch mis., %8llu "
                "cache ref., %8llu cache mis.\n",
                mins[0], mins[1], mins[2], mins[3], mins[4]);
        printf("avg: %8.1f cycles, %8.1f instructions, \t%8.1f branch mis., %8.1f "
                "cache ref., %8.1f cache mis.\n",
                avg[0], avg[1], avg[2], avg[3], avg[4]);
    } else {
        printf("cycles per 16-bit word:  %4.3f \n", double(mins[0]) / n / 4);
    }     
}

static void print_usage(char *command) {
    printf(" Try %s -n 100000 -i 15 -v \n", command);
    printf("-n is the number of 16-bit words \n");
    printf("-i is the number of tests or iterations \n");
    printf("-v makes things verbose\n");
}

int main(int argc, char **argv) {
    size_t n = 10000000;
    size_t iterations = 0; 
    bool verbose = false;
    int c;

    while ((c = getopt(argc, argv, "vhn:i:")) != -1) {
        switch (c) {
        case 'n':
            n = atoll(optarg);
            break;
        case 'v':
            verbose = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        case 'i':
            iterations = atoi(optarg);
            break;
        default:
            abort();
        }
    }

    if(n > UINT32_MAX) {
       printf("setting n to %u \n", UINT32_MAX);
       n = UINT32_MAX;
    }

    if(iterations > UINT32_MAX) {
       printf("setting iterations to %u \n", UINT32_MAX);
       iterations = UINT32_MAX;
    }

    if(iterations == 0) {
      if(n < 1000000) iterations = 100;
      else iterations = 10;
    }
    printf("n = %zu \n", n);
    printf("iterations = %zu \n", iterations);
    if(n == 0) {
       printf("n cannot be zero.\n");
       return EXIT_FAILURE;
    }

    measurepopcnt(n, iterations, verbose);
    
    for (size_t k = 0; k < PPOPCNT_NUMBER_METHODS; k++) {
        printf("%-40s\t", pospopcnt_u16_method_names[k]);
        fflush(NULL);
        bool isok = benchmark(n, iterations, pospopcnt_u16_methods[k], verbose, true);
        if (isok == false) {
            printf("Problem detected with %s.\n", pospopcnt_u16_method_names[k]);
        }
        if (verbose)
            printf("\n");
    }

    if (!verbose)
        printf("Try -v to get more details.\n");

    return EXIT_SUCCESS;
}
#else //  __linux__

#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("This is a linux-specific benchmark\n");
    return EXIT_SUCCESS;
}

#endif
