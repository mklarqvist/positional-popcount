/*
* Copyright (c) 2019 Marcus D. R. Klarqvist
* Author(s): Marcus D. R. Klarqvist
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
#include "fast_flagstats.h"

#if SIMD_VERSION >= 5
uint32_t flag_stats_avx2_popcnt(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    // 1 load data
    // 2 x | (((data[x] & mask[i]) >> i) << j)
    // 3 popcount
    __m256i masks[16];
    __m256i stubs[16];
    for(int i = 0; i < 16; ++i) {
        masks[i] = _mm256_set1_epi16(1 << i);
        stubs[i] = _mm256_set1_epi16(0);
    }

    uint32_t out_counters[16];
    memset(out_counters, 0, sizeof(uint32_t)*16);

    const __m256i* data_vectors = reinterpret_cast<const __m256i*>(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_cycles_updates = n_cycles / 16;

#define UPDATE(idx, shift) stubs[idx] = _mm256_or_si256(stubs[idx], _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(data_vectors[pos], masks[idx]),  idx), shift));
#define ITERATION(idx) {                                               \
        UPDATE(idx,0);  UPDATE(idx,1);  UPDATE(idx,2);  UPDATE(idx,3); \
        UPDATE(idx,4);  UPDATE(idx,5);  UPDATE(idx,6);  UPDATE(idx,7); \
        UPDATE(idx,8);  UPDATE(idx,9);  UPDATE(idx,10); UPDATE(idx,11);\
        UPDATE(idx,12); UPDATE(idx,13); UPDATE(idx,14); UPDATE(idx,15);\
        ++pos;                                                         \
}
#define BLOCK {                                                    \
        ITERATION(0);  ITERATION(1);  ITERATION(2);  ITERATION(3); \
        ITERATION(4);  ITERATION(5);  ITERATION(6);  ITERATION(7); \
        ITERATION(8);  ITERATION(9);  ITERATION(10); ITERATION(11);\
        ITERATION(12); ITERATION(13); ITERATION(14); ITERATION(15);\
}

    uint32_t pos = 0;
    for(int i = 0; i < n_cycles_updates; ++i) {
        BLOCK // unrolled

        /*
        // Not unrolled
        for(int c = 0; c < 16; ++c, ++pos) { // 16 iterations per register
            for(int j = 0; j < 16; ++j) { // each 1-hot per register
                UPDATE(j,c)
            }
        }
        */

        for(int j = 0; j < 16; ++j) {
            PIL_POPCOUNT_AVX2(out_counters[j], stubs[j])
            stubs[j] = _mm256_set1_epi16(0);
        }
    }

    // residual
    for(int i = pos*16; i < n; ++i) {
        for(int j = 0; j < 16; ++j) {
            out_counters[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    for(int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    //std::cerr << "popcnt=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << out_counters[i];
    //std::cerr << std::endl;

#undef BLOCK
#undef ITERATION
#undef UPDATE

    return(time_span.count());
}

uint32_t flag_stats_avx2(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    __m256i masks[16];
    __m256i counters[16];
    for(int i = 0; i < 16; ++i) {
        masks[i]    = _mm256_set1_epi16(1 << i);
        counters[i] = _mm256_set1_epi16(0);
    }
    uint32_t out_counters[16];
    memset(out_counters, 0, sizeof(uint32_t)*16);

    const __m256i hi_mask = _mm256_set1_epi32(0xFFFF0000);
    const __m256i lo_mask = _mm256_set1_epi32(0x0000FFFF);
    const __m256i* data_vectors = reinterpret_cast<const __m256i*>(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_update_cycles = std::floor((double)n_cycles / 65536);

#define UPDATE(idx) counters[idx]  = _mm256_add_epi16(counters[idx],  _mm256_srli_epi16(_mm256_and_si256(data_vectors[pos], masks[idx]),  idx))
#define ITERATION  {                                   \
        UPDATE(0);  UPDATE(1);  UPDATE(2);  UPDATE(3); \
        UPDATE(4);  UPDATE(5);  UPDATE(6);  UPDATE(7); \
        UPDATE(8);  UPDATE(9);  UPDATE(10); UPDATE(11);\
        UPDATE(12); UPDATE(13); UPDATE(14); UPDATE(15);\
        ++pos; ++k;                                    \
}

    uint32_t pos = 0;
    for(int i = 0; i < n_update_cycles; ++i) { // each block of 2^16 values
        for(int k = 0; k < 65536; ) // max sum of each 16-bit value in a register
            ITERATION // unrolled

        // Compute vector sum
        for(int k = 0; k < 16; ++k) { // each flag register
            // Accumulator
            // ((16-bit high & 16 high) >> 16) + (16-bit low & 16-low)
            __m256i x = _mm256_add_epi32(
                           _mm256_srli_epi32(_mm256_and_si256(counters[k], hi_mask), 16),
                           _mm256_and_si256(counters[k], lo_mask));
            __m256i t1 = _mm256_hadd_epi32(x,x);
            __m256i t2 = _mm256_hadd_epi32(t1,t1);
            __m128i t4 = _mm_add_epi32(_mm256_castsi256_si128(t2),_mm256_extractf128_si256(t2,1));
            out_counters[k] += _mm_cvtsi128_si32(t4);

            /*
            // Naive counter
            uint16_t* d = reinterpret_cast<uint16_t*>(&counters[k]);
            for(int j = 0; j < 16; ++j) { // each uint16_t in the register
                out_counters[k] += d[j];
            }
            */

            counters[k] = _mm256_set1_epi16(0);
        }
    }

    // residual
    for(int i = pos*16; i < n; ++i) {
        for(int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    for(int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    //std::cerr << "simd=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << out_counters[i];
    //std::cerr << std::endl;

#undef ITERATION
#undef UPDATE

    return(time_span.count());

}

uint32_t flag_stats_avx2_naive_counter(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    __m256i masks[16];
    __m256i counters[16];
    for(int i = 0; i < 16; ++i) {
        masks[i]    = _mm256_set1_epi16(1 << i);
        counters[i] = _mm256_set1_epi16(0);
    }
    uint32_t out_counters[16];
    memset(out_counters, 0, sizeof(uint32_t)*16);

    const __m256i* data_vectors = reinterpret_cast<const __m256i*>(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_update_cycles = std::floor((double)n_cycles / 65536);
    //std::cerr << n << " values and " << n_cycles << " cycles " << n_residual << " residual cycles" << std::endl;

#define UPDATE(idx) counters[idx]  = _mm256_add_epi16(counters[idx],  _mm256_srli_epi16(_mm256_and_si256(data_vectors[pos], masks[idx]),  idx))

    uint32_t pos = 0;
    for(int i = 0; i < n_update_cycles; ++i) { // each block of 2^16 values
        for(int k = 0; k < 65536; ++pos,++k) { // max sum of each 16-bit value in a register
            for(int p = 0; p < 16; ++p) // Not unrolled
                UPDATE(p);
        }

        // Compute vector sum
        for(int k = 0; k < 16; ++k) { // each flag register
            // Naive counter
            uint16_t* d = reinterpret_cast<uint16_t*>(&counters[k]);
            for(int j = 0; j < 16; ++j) // each uint16_t in the register
                out_counters[k] += d[j];

            counters[k] = _mm256_set1_epi16(0);
        }
    }

    // residual
    for(int i = pos*16; i < n; ++i) {
        for(int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    for(int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    //std::cerr << "simd=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << out_counters[i];
    //std::cerr << std::endl;

#undef UPDATE

    return(time_span.count());
}

uint32_t flag_stats_avx2_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    __m256i counter = _mm256_set1_epi16(0);
    const __m256i one_mask =  _mm256_set1_epi16(1);
    // set_epi is parameterized backwards (15->0)
    const __m256i masks = _mm256_set_epi16(1 << 15, 1 << 14, 1 << 13, 1 << 12,
                                           1 << 11, 1 << 10, 1 << 9,  1 << 8,
                                           1 << 7,  1 << 6,  1 << 5,  1 << 4,
                                           1 << 3,  1 << 2,  1 << 1,  1 << 0);
    uint32_t out_counters[16] = {0};
    const __m256i* data_vectors = reinterpret_cast<const __m256i*>(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_update_cycles = std::floor((double)n_cycles / 4096);

#define UPDATE(idx) counter = _mm256_add_epi16(counter, _mm256_and_si256(_mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(_mm256_extract_epi16(data_vectors[pos], idx)), masks), masks), one_mask));
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

    uint32_t pos = 0;
    for(int i = 0; i < n_update_cycles; ++i) { // each block of 65536 values
        for(int k = 0; k < 4096; ++k, ++pos) { // max sum of each 16-bit value in a register (65536/16)
            BLOCK
        }

        // Compute vector sum
        for(int k = 0; k < 16; ++k) // each flag register
            out_counters[k] += _mm256_extract_epi16(counter, k);

        counter = _mm256_set1_epi16(0);
    }

#undef UPDATE
#undef BLOCK

    // residual
    for(int i = pos*16; i < n; ++i) {
        for(int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    for(int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    //std::cerr << "simd=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << out_counters[i];
    //std::cerr << std::endl;

    return(time_span.count());

}
#else
uint32_t flag_stats_avx2_popcnt(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) { return(0); }
uint32_t flag_stats_avx2(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) { return(0); }
uint32_t flag_stats_avx2_naive_counter(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) { return(0); }
uint32_t flag_stats_avx2_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) { return(0); }
#endif

#if SIMD_VERSION >= 3
uint32_t flag_stats_sse_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    __m128i counterLo = _mm_set1_epi16(0);
    __m128i counterHi = _mm_set1_epi16(0);
    const __m128i one_mask =  _mm_set1_epi16(1);
    // set_epi is parameterized backwards (15->0)
    const __m128i masksLo = _mm_set_epi16(1 << 15, 1 << 14, 1 << 13, 1 << 12,
                                          1 << 11, 1 << 10, 1 << 9,  1 << 8);
    const __m128i masksHi = _mm_set_epi16(1 << 7,  1 << 6,  1 << 5,  1 << 4,
                                          1 << 3,  1 << 2,  1 << 1,  1 << 0);

    uint32_t out_counters[16] = {0};
    const __m128i* data_vectors = reinterpret_cast<const __m128i*>(data);
    const uint32_t n_cycles = n / 8;
    const uint32_t n_update_cycles = std::floor((double)n_cycles / 4096);

#define UPDATE_LO(idx) counterLo = _mm_add_epi16(counterLo, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(_mm_extract_epi16(data_vectors[pos], idx)), masksLo), masksLo), one_mask));
#define UPDATE_HI(idx) counterHi = _mm_add_epi16(counterHi, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(_mm_extract_epi16(data_vectors[pos], idx)), masksHi), masksHi), one_mask));
#define BLOCK {                                         \
    UPDATE_LO(0) UPDATE_LO(1) UPDATE_LO(2) UPDATE_LO(3) \
    UPDATE_LO(4) UPDATE_LO(5) UPDATE_LO(6) UPDATE_LO(7) \
    UPDATE_HI(0) UPDATE_HI(1) UPDATE_HI(2) UPDATE_HI(3) \
    UPDATE_HI(4) UPDATE_HI(5) UPDATE_HI(6) UPDATE_HI(7) \
}
#define UH(idx) out_counters[idx] += _mm_extract_epi16(counterLo, idx - 8);
#define UL(idx) out_counters[idx] += _mm_extract_epi16(counterHi, idx);

    uint32_t pos = 0;
    for(int i = 0; i < n_update_cycles; ++i) { // each block of 65536 values
        for(int k = 0; k < 4096; ++k, ++pos) { // max sum of each 16-bit value in a register (65536/16)
            BLOCK
        }

        // Compute vector sum (unroll to prevent possible compiler errors
        // regarding constness of parameter N in _mm_extract_epi16).
        UL(0)  UL(1)  UL(2)  UL(3)
        UL(4)  UL(5)  UL(6)  UL(7)
        UH(8)  UH(9)  UH(10) UH(11)
        UH(12) UH(13) UH(14) UH(15)
        counterLo = _mm_set1_epi16(0);
        counterHi = _mm_set1_epi16(0);
    }

#undef UL
#undef UH
#undef BLOCK
#undef UPDATE_HI
#undef UPDATE_LO

    // residual
    for(int i = pos*8; i < n; ++i) {
        for(int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    for(int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    //std::cerr << "simd=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << out_counters[i];
    //std::cerr << std::endl;

    return(time_span.count());

}
#else
uint32_t flag_stats_sse_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) { return(0); }
#endif

uint32_t flag_stats_scalar_naive(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    memset(flags, 0, 16*sizeof(uint32_t));

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    std::cerr << "truth=";
    for(int i = 0; i < 16; ++i) std::cerr << " " << flags[i];
    std::cerr << std::endl;

    return(time_span.count());
}

uint32_t flag_stats_scalar_partition(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    uint32_t low[256] = {0}, high[256] = {0};
    memset(flags, 0, 16*sizeof(uint32_t));

    for(int i = 0; i < n; ++i) {
        ++low[data[i] & 255];
        ++high[(data[i] >> 8) & 255];
    }

    for(int i = 0; i < 256; ++i) {
        for(int k = 0; k < 8; ++k) {
            flags[k] += ((i & (1 << k)) >> k) * low[i];
        }
    }

    for(int i = 0; i < 256; ++i) {
        for(int k = 0; k < 8; ++k) {
            flags[k+8] += ((i & (1 << k)) >> k) * high[i];
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    //std::cerr << "truth=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << flags[i];
    //std::cerr << std::endl;

    return(time_span.count());
}

uint32_t flag_stats_hist1x4(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
     std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

     uint32_t low[256] = {0}, high[256] = {0};
     memset(flags, 0, 16*sizeof(uint32_t));
     
     int i = 0;
     for (i = 0; i < (n & ~3); i+=4) {
          ++low[data[i+0] & 255];
          ++high[(data[i+0] >> 8) & 255];
          ++low[data[i+1] & 255];
          ++high[(data[i+1] >> 8) & 255];
          ++low[data[i+2] & 255];
          ++high[(data[i+2] >> 8) & 255];
          ++low[data[i+3] & 255];
          ++high[(data[i+3] >> 8) & 255];
     }
     while (i < n) {
          ++low[data[i] & 255];
          ++high[(data[i++] >> 8) & 255];
     }

     for(int i = 0; i < 256; ++i) {
        for(int k = 0; k < 8; ++k) {
            flags[k] += ((i & (1 << k)) >> k) * low[i];
        }
    }

    for(int i = 0; i < 256; ++i) {
        for(int k = 0; k < 8; ++k) {
            flags[k+8] += ((i & (1 << k)) >> k) * high[i];
        }
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    //std::cerr << "truth=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << flags[i];
    //std::cerr << std::endl;

    return(time_span.count());
}

#if SIMD_VERSION >= 6
uint32_t flag_stats_avx512_popcnt32(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    __m512i masks[16];
    for(int i = 0; i < 16; ++i) {
        masks[i] = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
    }
    uint32_t out_counters[16];
    memset(out_counters, 0, sizeof(uint32_t)*16);

    const __m512i* data_vectors = reinterpret_cast<const __m512i*>(data);
    const uint32_t n_cycles = n / 32;

#define UPDATE(pos) out_counters[pos] += PIL_POPCOUNT((uint64_t)_mm512_cmpeq_epu16_mask(_mm512_and_epi32(data_vectors[i], masks[pos]), masks[pos]));
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

    uint32_t pos = 0;
    for(int i = 0; i < n_cycles; ++i) { // each block of 2^16 values
        BLOCK
    }

    // residual
    for(int i = pos*32; i < n; ++i) {
        for(int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    for(int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    //std::cerr << "simd=";
    //for(int i = 0; i < 16; ++i) std::cerr << " " << out_counters[i];
    //std::cerr << std::endl;

#undef BLOCK
#undef UPDATE

    return(time_span.count());
}

uint32_t flag_stats_avx512_popcnt(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    __m512i masks[16];
    __m512i counters[16];
    for(int i = 0; i < 16; ++i) {
        masks[i] = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
        counters[i] = _mm512_set1_epi32(0);
    }
    uint32_t out_counters[16];
    memset(out_counters, 0, sizeof(uint32_t)*16);

    const __m512i* data_vectors = reinterpret_cast<const __m512i*>(data);
    const uint32_t n_cycles = n / 32;

#define UPDATE(pos) counters[pos] = _mm512_add_epi32(counters[pos], avx512_popcount(_mm512_and_epi32(data_vectors[i], masks[pos])));
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

    //uint32_t pos = 0;
    for(int i = 0; i < n_cycles; ++i) { // each block of 2^16 values
        BLOCK
    }

    // residual
    for(int i = pos*32; i < n; ++i) {
        for(int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for(int i = 0; i < 16; ++i) {
        uint32_t* v = reinterpret_cast<uint32_t*>(&counters[i]);
        for(int j = 0; j < 16; ++j)
            flags[i] += v[j];
    }
    for(int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    

    std::cerr << "simd=";
    for(int i = 0; i < 16; ++i) std::cerr << " " << out_counters[i];
    std::cerr << std::endl;

    return(time_span.count());
}
#else
uint32_t flag_stats_avx512_popcnt32(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) { return(0); }
uint32_t flag_stats_avx512_popcnt(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) { return(0); }
#endif

uint32_t compute_flag_stats(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    #if SIMD_VERSION >= 6
    return(flag_stats_avx2_naive_counter(data, n, flags)); // still fastest
    #elif SIMD_VERSION >= 5
    return(flag_stats_avx2_naive_counter(data, n, flags));
    #elif SIMD_VERSION >= 3
    return(flag_stats_sse_single(data, n, flags));
    #else
    return(flag_stats_hist1x4(data, n, flags));
    #endif
}