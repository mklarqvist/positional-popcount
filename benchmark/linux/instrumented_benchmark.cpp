#ifdef __linux__

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

#include "fast_flagstats.h"
#include "linux-perf-events.h"

typedef int (*pospopcnt16)(const uint16_t *, uint32_t, uint32_t *);

static inline void CSA(__m256i *h, __m256i *l, __m256i a, __m256i b,
                       __m256i c) {
  const __m256i u = _mm256_xor_si256(a, b);
  *h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
  *l = _mm256_xor_si256(u, c);
}

int purecircuit(const uint16_t *array, uint32_t len, uint32_t *flags) {
  for (size_t i = 0; i < 16; i++)
    flags[i] = 0;
  for (uint32_t i = len - (len % (16 * 16)); i < len; ++i) {
    for (int j = 0; j < 16; ++j) {
      flags[j] += ((array[i] & (1 << j)) >> j);
    }
  }
  const __m256i *data = (const __m256i *)array;
  size_t size = len / 16;
  __m256i ones = _mm256_setzero_si256();
  __m256i twos = _mm256_setzero_si256();
  __m256i fours = _mm256_setzero_si256();
  __m256i eights = _mm256_setzero_si256();
  __m256i sixteens = _mm256_setzero_si256();
  __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  uint16_t buffer[16];

  // uint16_t x = 0;
  while (i < limit) {
    __m256i counter[16];
    for (size_t i = 0; i < 16; i++) {
      counter[i] = _mm256_setzero_si256();
    }
    size_t thislimit = limit;
    if (thislimit - i >= (1 << 16))
      thislimit = i + (1 << 16) - 1;
    for (; i < thislimit; i += 16) {
      CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i),
          _mm256_lddqu_si256(data + i + 1));
      CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 2),
          _mm256_lddqu_si256(data + i + 3));
      CSA(&foursA, &twos, twos, twosA, twosB);
      CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i + 4),
          _mm256_lddqu_si256(data + i + 5));
      CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 6),
          _mm256_lddqu_si256(data + i + 7));
      CSA(&foursB, &twos, twos, twosA, twosB);
      CSA(&eightsA, &fours, fours, foursA, foursB);
      CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i + 8),
          _mm256_lddqu_si256(data + i + 9));
      CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 10),
          _mm256_lddqu_si256(data + i + 11));
      CSA(&foursA, &twos, twos, twosA, twosB);
      CSA(&twosA, &ones, ones, _mm256_lddqu_si256(data + i + 12),
          _mm256_lddqu_si256(data + i + 13));
      CSA(&twosB, &ones, ones, _mm256_lddqu_si256(data + i + 14),
          _mm256_lddqu_si256(data + i + 15));
      CSA(&foursB, &twos, twos, twosA, twosB);
      CSA(&eightsB, &fours, fours, foursA, foursB);
      CSA(&sixteens, &eights, eights, eightsA, eightsB);
      for (size_t i = 0; i < 16; i++) {
        counter[i] = _mm256_add_epi16(
            counter[i], _mm256_and_si256(sixteens, _mm256_set1_epi16(1)));
        sixteens = _mm256_srli_epi16(sixteens, 1);
      }
    }
    for (size_t i = 0; i < 16; i++) {
      _mm256_storeu_si256((__m256i *)buffer, counter[i]);
      for (size_t z = 0; z < 16; z++) {
        flags[i] += buffer[z] * 16;
      }
    }
  }

  _mm256_storeu_si256((__m256i *)buffer, ones);
  for (size_t i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      flags[j] += ((buffer[i] & (1 << j)) >> j);
    }
  }

  _mm256_storeu_si256((__m256i *)buffer, twos);
  for (size_t i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      flags[j] += 2 * ((buffer[i] & (1 << j)) >> j);
    }
  }
  _mm256_storeu_si256((__m256i *)buffer, fours);
  for (size_t i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      flags[j] += 4 * ((buffer[i] & (1 << j)) >> j);
    }
  }
  _mm256_storeu_si256((__m256i *)buffer, eights);
  for (size_t i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      flags[j] += 8 * ((buffer[i] & (1 << j)) >> j);
    }
  }
  return 0;
}



#define NUMBEROFFNC 29
pospopcnt16 ourfunctions[NUMBEROFFNC] = {
  pospopcnt_u16_scalar_naive_nosimd,  pospopcnt_u16_scalar_naive,
  pospopcnt_u16_scalar_partition,     pospopcnt_u16_hist1x4,
  pospopcnt_u16_sse_single,           pospopcnt_u16_sse_mula,
  pospopcnt_u16_sse_mula_unroll4,     pospopcnt_u16_sse_mula_unroll8,
  pospopcnt_u16_sse_mula_unroll16,    pospopcnt_u16_avx2_popcnt,
  pospopcnt_u16_avx2,                 pospopcnt_u16_avx2_naive_counter,
  pospopcnt_u16_avx2_single,          pospopcnt_u16_avx2_lemire,
  pospopcnt_u16_avx2_lemire2,         pospopcnt_u16_avx2_mula,
  pospopcnt_u16_avx2_mula2,           pospopcnt_u16_avx2_mula3,
  pospopcnt_u16_avx2_mula_unroll4,    pospopcnt_u16_avx2_mula_unroll8,
  pospopcnt_u16_avx2_mula_unroll16,   pospopcnt_u16_avx512,
  pospopcnt_u16_avx512_popcnt32_mask, pospopcnt_u16_avx512_popcnt64_mask,
  pospopcnt_u16_avx512_popcnt,        pospopcnt_u16_avx512_mula,
  pospopcnt_u16_avx512_mula_unroll4,  pospopcnt_u16_avx512_mula_unroll8, purecircuit
};

std::string ourfunctionsnames[NUMBEROFFNC] = {
  "pospopcnt_u16_scalar_naive_nosimd",  "pospopcnt_u16_scalar_naive",
  "pospopcnt_u16_scalar_partition",     "pospopcnt_u16_hist1x4",
  "pospopcnt_u16_sse_single",           "pospopcnt_u16_sse_mula",
  "pospopcnt_u16_sse_mula_unroll4",     "pospopcnt_u16_sse_mula_unroll8",
  "pospopcnt_u16_sse_mula_unroll16",    "pospopcnt_u16_avx2_popcnt",
  "pospopcnt_u16_avx2",                 "pospopcnt_u16_avx2_naive_counter",
  "pospopcnt_u16_avx2_single",          "pospopcnt_u16_avx2_lemire",
  "pospopcnt_u16_avx2_lemire2",         "pospopcnt_u16_avx2_mula",
  "pospopcnt_u16_avx2_mula2",           "pospopcnt_u16_avx2_mula3",
  "pospopcnt_u16_avx2_mula_unroll4",    "pospopcnt_u16_avx2_mula_unroll8",
  "pospopcnt_u16_avx2_mula_unroll16",   "pospopcnt_u16_avx512",
  "pospopcnt_u16_avx512_popcnt32_mask", "pospopcnt_u16_avx512_popcnt64_mask",
  "pospopcnt_u16_avx512_popcnt",        "pospopcnt_u16_avx512_mula",
  "pospopcnt_u16_avx512_mula_unroll4",  "pospopcnt_u16_avx512_mula_unroll8", "purecircuit"
};


static __m256i avx2_popcount(const __m256i vec) {

    const __m256i lookup = _mm256_setr_epi8(
        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,

        /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
        /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
        /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
        /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
    );

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo  = _mm256_and_si256(vec, low_mask);
    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);

    return _mm256_add_epi8(popcnt1, popcnt2);
}


static __m256i popcount(const __m512i v)
{
    const __m256i lo = _mm512_extracti64x4_epi64(v, 0);
    const __m256i hi = _mm512_extracti64x4_epi64(v, 1);
    const __m256i s  = _mm256_add_epi8(avx2_popcount(lo), avx2_popcount(hi));

    return _mm256_sad_epu8(s, _mm256_setzero_si256());
}


static uint64_t avx2_sum_epu64(const __m256i v) {
    
    return _mm256_extract_epi64(v, 0)
         + _mm256_extract_epi64(v, 1)
         + _mm256_extract_epi64(v, 2)
         + _mm256_extract_epi64(v, 3);
}


static void CSA(__m512i* h, __m512i* l, __m512i a, __m512i b, __m512i c) {

  *l = _mm512_ternarylogic_epi32(c, b, a, 0x96);
  *h = _mm512_ternarylogic_epi32(c, b, a, 0xe8);
}


static uint64_t popcnt_harley_seal(const __m512i* data, const uint64_t size)
{
  __m256i total     = _mm256_setzero_si256();
  __m512i ones      = _mm512_setzero_si512();
  __m512i twos      = _mm512_setzero_si512();
  __m512i fours     = _mm512_setzero_si512();
  __m512i eights    = _mm512_setzero_si512();
  __m512i sixteens  = _mm512_setzero_si512();
  __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

  const uint64_t limit = size - size % 16;
  uint64_t i = 0;

  for(; i < limit; i += 16)
  {
    CSA(&twosA, &ones, ones, _mm512_loadu_si512(data+i+0), _mm512_loadu_si512(data+i+1));
    CSA(&twosB, &ones, ones, _mm512_loadu_si512(data+i+2), _mm512_loadu_si512(data+i+3));
    CSA(&foursA, &twos, twos, twosA, twosB);
    CSA(&twosA, &ones, ones, _mm512_loadu_si512(data+i+4), _mm512_loadu_si512(data+i+5));
    CSA(&twosB, &ones, ones, _mm512_loadu_si512(data+i+6), _mm512_loadu_si512(data+i+7));
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsA,&fours, fours, foursA, foursB);
    CSA(&twosA, &ones, ones, _mm512_loadu_si512(data+i+8), _mm512_loadu_si512(data+i+9));
    CSA(&twosB, &ones, ones, _mm512_loadu_si512(data+i+10), _mm512_loadu_si512(data+i+11));
    CSA(&foursA, &twos, twos, twosA, twosB);
    CSA(&twosA, &ones, ones, _mm512_loadu_si512(data+i+12), _mm512_loadu_si512(data+i+13));
    CSA(&twosB, &ones, ones, _mm512_loadu_si512(data+i+14), _mm512_loadu_si512(data+i+15));
    CSA(&foursB, &twos, twos, twosA, twosB);
    CSA(&eightsB, &fours, fours, foursA, foursB);
    CSA(&sixteens, &eights, eights, eightsA, eightsB);

    total = _mm256_add_epi64(total, popcount(sixteens));
  }

  total = _mm256_slli_epi64(total, 4);     // * 16
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(eights), 3)); // += 8 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(fours),  2)); // += 4 * ...
  total = _mm256_add_epi64(total, _mm256_slli_epi64(popcount(twos),   1)); // += 2 * ...
  total = _mm256_add_epi64(total, popcount(ones));

  for(; i < size; i++) {
    total = _mm256_add_epi64(total, popcount(_mm512_loadu_si512(data+i)));
  }


  return avx2_sum_epu64(total);
}

void print16(uint32_t *flags) {
    for (int k = 0; k < 16; k++)
        printf(" %8u ", flags[k]);
    printf("\n");
}

std::vector<unsigned long long>
computemins(std::vector< std::vector<unsigned long long> > allresults) {
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
computeavgs(std::vector< std::vector<unsigned long long> > allresults) {
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

void measurepopcnt(uint16_t n, uint32_t iterations, bool verbose) {
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

    n = vdata.size() / (512 / 16) * (512/16);
    for (uint32_t i = 0; i < iterations; i++) {
        for (size_t k = 0; k < vdata.size(); k++) {
            vdata[k] = dis(gen); // random init.
        }
        uint64_t expected = popcnt_harley_seal((const __m512i*) vdata.data(), vdata.size() / (512 / 16));       
        unified.start();
        uint64_t measured = popcnt_harley_seal((const __m512i*) vdata.data(), vdata.size() / (512 / 16));
        unified.end(results);
        assert(measured == expected);
        allresults.push_back(results);
    }

    std::vector<unsigned long long> mins = computemins(allresults);
    std::vector<double> avg = computeavgs(allresults);
    printf("%-40s\t","avx512popcnt");    
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
    }

     
}


/**
 * @brief 
 * 
 * @param n          Number of integers.
 * @param iterations Number of iterations.
 * @param fn         Target function pointer.
 * @param verbose    Flag enabling verbose output.
 * @return Returns true if the results are correct. returns false if the results
 *         are either incorrect or the target function is not supported.
 */
bool benchmark(uint16_t n, uint32_t iterations, pospopcnt16 fn, bool verbose, bool test) {
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

    std::vector<unsigned long long> mins = computemins(allresults);
    std::vector<double> avg = computeavgs(allresults);
    
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
    }

    return isok;
}

static void print_usage(char *command) {
    printf(" Try %s -n 100000 -i 15 -v \n", command);
    printf("-n is the number of 16-bit words \n");
    printf("-i is the number of tests or iterations \n");
    printf("-v makes things verbose\n");
}

int main(int argc, char **argv) {
    size_t n = 10000000;
    size_t iterations = 100;
    bool verbose = false;
    int c;

    while ((c = getopt(argc, argv, "vhn:i:")) != -1) {
        switch (c) {
        case 'n':
            n = atol(optarg);
            break;
        case 'v':
            verbose = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        case 'i':
            iterations = atol(optarg);
            break;
        default:
            abort();
        }
    }
    printf("n = %zu \n", n);
    
    for (size_t k = 0; k < NUMBEROFFNC; k++) {
        printf("%-40s\t", ourfunctionsnames[k].c_str());
        fflush(NULL);
        bool isok = benchmark(n, iterations, ourfunctions[k], verbose, true);
        if (isok == false) {
            printf("Problem detected with %s.\n", ourfunctionsnames[k].c_str());
            printf("%-40s\t", ourfunctionsnames[k].c_str());
            benchmark(n, iterations, ourfunctions[k], verbose, false);
        }
        if (verbose)
            printf("\n");
    }
    measurepopcnt(n, iterations, verbose);
    if (!verbose)
        printf("Try -v to get more details.\n");

    return EXIT_SUCCESS;
}

///////////// THE END /////////////////
#else //  __linux__

#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("This is a linux-specific benchmark\n");
    return EXIT_SUCCESS;
}

#endif
