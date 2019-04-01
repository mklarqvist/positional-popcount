/*
* Copyright (c) 2019
* Author(s): Marcus D. R. Klarqvist, Wojciech Muła, and Daniel Lemire
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
#include <string.h> //memset

#include "pospopcnt.h"

int pospopcnt_u16(const uint16_t* data, uint32_t n, uint32_t* flags) {
#if SIMD_VERSION >= 6
    return(pospopcnt_u16_avx512_csa(data, n, flags));
#elif SIMD_VERSION >= 5
    return(pospopcnt_u16_avx2_csa(data, n, flags));
#elif SIMD_VERSION >= 3
    return(pospopcnt_u16_sse_sad(data, n, flags));
#else
    return(pospopcnt_u16_scalar_naive(data, n, flags));
#endif
}

int pospopcnt_u16_method(PPOPCNT_U16_METHODS method, const uint16_t* data, uint32_t n, uint32_t* flags) {
    switch(method) {
    case(PPOPCNT_AUTO): return pospopcnt_u16(data, n, flags);
    case(PPOPCNT_SCALAR): return pospopcnt_u16_scalar_naive(data, n, flags);
    case(PPOPCNT_SCALAR_NOSIMD): return pospopcnt_u16_scalar_naive_nosimd(data, n, flags);
    case(PPOPCNT_SCALAR_PARTITION): return pospopcnt_u16_scalar_partition(data, n, flags);
    case(PPOPCNT_SCALAR_HIST1X4): return pospopcnt_u16_hist1x4(data, n, flags);
    case(PPOPCNT_SSE_SINGLE): return pospopcnt_u16_sse_single(data, n, flags);
    case(PPOPCNT_SSE_MULA): return pospopcnt_u16_sse_mula(data, n, flags);
    case(PPOPCNT_SSE_MULA_UR4): return pospopcnt_u16_sse_mula_unroll4(data, n, flags);
    case(PPOPCNT_SSE_MULA_UR8): return pospopcnt_u16_sse_mula_unroll8(data, n, flags);
    case(PPOPCNT_SSE_MULA_UR16): return pospopcnt_u16_sse_mula_unroll16(data, n, flags);
    case(PPOPCNT_SSE_SAD): return pospopcnt_u16_sse_sad(data, n, flags);
    case(PPOPCNT_AVX2_POPCNT): return pospopcnt_u16_avx2_popcnt(data, n, flags);
    case(PPOPCNT_AVX2): return pospopcnt_u16_avx2(data, n, flags);
    case(PPOPCNT_AVX2_POPCNT_NAIVE): return pospopcnt_u16_avx2_naive_counter(data, n, flags);
    case(PPOPCNT_AVX2_SINGLE): return pospopcnt_u16_avx2_single(data, n, flags);
    case(PPOPCNT_AVX2_LEMIRE1): return pospopcnt_u16_avx2_lemire(data, n, flags);
    case(PPOPCNT_AVX2_LEMIRE2): return pospopcnt_u16_avx2_lemire2(data, n, flags);
    case(PPOPCNT_AVX2_MULA): return pospopcnt_u16_avx2_mula(data, n, flags);
    case(PPOPCNT_AVX2_MULA_UR4): return pospopcnt_u16_avx2_mula_unroll4(data, n, flags);
    case(PPOPCNT_AVX2_MULA_UR8): return pospopcnt_u16_avx2_mula_unroll8(data, n, flags);
    case(PPOPCNT_AVX2_MULA_UR16): return pospopcnt_u16_avx2_mula_unroll16(data, n, flags);
    case(PPOPCNT_AVX2_MULA3): return pospopcnt_u16_avx2_mula3(data, n, flags);
    case(PPOPCNT_AVX2_CSA): return pospopcnt_u16_avx2_csa(data, n, flags);
    case(PPOPCNT_AVX512): return pospopcnt_u16_avx512(data, n, flags);
    case(PPOPCNT_AVX512_MASK32): return pospopcnt_u16_avx512_popcnt32_mask(data, n, flags);
    case(PPOPCNT_AVX512_MASK64): return pospopcnt_u16_avx512_popcnt64_mask(data, n, flags);
    case(PPOPCNT_AVX512_POPCNT): return pospopcnt_u16_avx512_popcnt(data, n, flags);
    case(PPOPCNT_AVX512_MULA): return pospopcnt_u16_avx512_mula(data, n, flags);
    case(PPOPCNT_AVX512_MULA_UR4): return pospopcnt_u16_avx512_mula_unroll4(data, n, flags);
    case(PPOPCNT_AVX512_MULA_UR8): return pospopcnt_u16_avx512_mula_unroll8(data, n, flags);
    case(PPOPCNT_AVX512_MULA3): return pospopcnt_u16_avx512_mula3(data, n, flags);
    case(PPOPCNT_AVX512_CSA): return pospopcnt_u16_avx512_csa(data, n, flags);
    case(PPOPCNT_AVX512_MULA2): return pospopcnt_u16_avx512_mula2(data, n, flags);
    }
    return 0;
}

pospopcnt_u16_method_type get_pospopcnt_u16_method(PPOPCNT_U16_METHODS method) {
    switch(method) {
    case(PPOPCNT_AUTO): return &pospopcnt_u16;
    case(PPOPCNT_SCALAR): return &pospopcnt_u16_scalar_naive;
    case(PPOPCNT_SCALAR_NOSIMD): return &pospopcnt_u16_scalar_naive_nosimd;
    case(PPOPCNT_SCALAR_PARTITION): return &pospopcnt_u16_scalar_partition;
    case(PPOPCNT_SCALAR_HIST1X4): return &pospopcnt_u16_hist1x4;
    case(PPOPCNT_SSE_SINGLE): return &pospopcnt_u16_sse_single;
    case(PPOPCNT_SSE_MULA): return &pospopcnt_u16_sse_mula;
    case(PPOPCNT_SSE_MULA_UR4): return &pospopcnt_u16_sse_mula_unroll4;
    case(PPOPCNT_SSE_MULA_UR8): return &pospopcnt_u16_sse_mula_unroll8;
    case(PPOPCNT_SSE_MULA_UR16): return &pospopcnt_u16_sse_mula_unroll16;
    case(PPOPCNT_SSE_SAD): return &pospopcnt_u16_sse_sad;
    case(PPOPCNT_AVX2_POPCNT): return &pospopcnt_u16_avx2_popcnt;
    case(PPOPCNT_AVX2): return &pospopcnt_u16_avx2;
    case(PPOPCNT_AVX2_POPCNT_NAIVE): return &pospopcnt_u16_avx2_naive_counter;
    case(PPOPCNT_AVX2_SINGLE): return &pospopcnt_u16_avx2_single;
    case(PPOPCNT_AVX2_LEMIRE1): return &pospopcnt_u16_avx2_lemire;
    case(PPOPCNT_AVX2_LEMIRE2): return &pospopcnt_u16_avx2_lemire2;
    case(PPOPCNT_AVX2_MULA): return &pospopcnt_u16_avx2_mula;
    case(PPOPCNT_AVX2_MULA_UR4): return &pospopcnt_u16_avx2_mula_unroll4;
    case(PPOPCNT_AVX2_MULA_UR8): return &pospopcnt_u16_avx2_mula_unroll8;
    case(PPOPCNT_AVX2_MULA_UR16): return &pospopcnt_u16_avx2_mula_unroll16;
    case(PPOPCNT_AVX2_MULA3): return &pospopcnt_u16_avx2_mula3;
    case(PPOPCNT_AVX2_CSA): return pospopcnt_u16_avx2_csa;
    case(PPOPCNT_AVX512): return &pospopcnt_u16_avx512;
    case(PPOPCNT_AVX512_MASK32): return &pospopcnt_u16_avx512_popcnt32_mask;
    case(PPOPCNT_AVX512_MASK64): return &pospopcnt_u16_avx512_popcnt64_mask;
    case(PPOPCNT_AVX512_POPCNT): return &pospopcnt_u16_avx512_popcnt;
    case(PPOPCNT_AVX512_MULA): return &pospopcnt_u16_avx512_mula;
    case(PPOPCNT_AVX512_MULA_UR4): return &pospopcnt_u16_avx512_mula_unroll4;
    case(PPOPCNT_AVX512_MULA_UR8): return &pospopcnt_u16_avx512_mula_unroll8;
    case(PPOPCNT_AVX512_MULA3): return &pospopcnt_u16_avx512_mula3;
    case(PPOPCNT_AVX512_CSA): return &pospopcnt_u16_avx512_csa;
    case(PPOPCNT_AVX512_MULA2): return &pospopcnt_u16_avx512_mula2;
    }
    return 0;
}

#if SIMD_VERSION >= 5
int pospopcnt_u16_avx2_popcnt(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m256i masks[16];
    __m256i stubs[16];
    for (int i = 0; i < 16; ++i) {
        masks[i] = _mm256_set1_epi16(1 << i);
        stubs[i] = _mm256_set1_epi16(0);
    }

    uint32_t out_counters[16] = {0};

    const __m256i* data_vectors = (const __m256i*)(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_cycles_updates = n_cycles / 16;

#define UPDATE(idx, shift) stubs[idx] = _mm256_or_si256(stubs[idx], _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(_mm256_loadu_si256(data_vectors+pos), masks[idx]),  idx), shift));
#define ITERATION(idx) {                                           \
    UPDATE(0,idx);  UPDATE(1,idx);  UPDATE(2,idx);  UPDATE(3,idx); \
    UPDATE(4,idx);  UPDATE(5,idx);  UPDATE(6,idx);  UPDATE(7,idx); \
    UPDATE(8,idx);  UPDATE(9,idx);  UPDATE(10,idx); UPDATE(11,idx);\
    UPDATE(12,idx); UPDATE(13,idx); UPDATE(14,idx); UPDATE(15,idx);\
    ++pos;                                                         \
}
#define BLOCK {                                                \
    ITERATION(0);  ITERATION(1);  ITERATION(2);  ITERATION(3); \
    ITERATION(4);  ITERATION(5);  ITERATION(6);  ITERATION(7); \
    ITERATION(8);  ITERATION(9);  ITERATION(10); ITERATION(11);\
    ITERATION(12); ITERATION(13); ITERATION(14); ITERATION(15);\
}

    uint32_t pos = 0;
    for (int i = 0; i < n_cycles_updates; ++i) {
        BLOCK // unrolled
        
        // Not unrolled
        /*
        for (int c = 0; c < 16; ++c) { // 16 iterations per register
            ITERATION(c)
            // for (int j = 0; j < 16; ++j) { // each 1-hot per register
            //     UPDATE(j,c)
            // }
        }
        */

        for (int j = 0; j < 16; ++j) {
            PIL_POPCOUNT_AVX2(out_counters[j], stubs[j])
            stubs[j] = _mm256_set1_epi16(0);
        }
    }

    // residual
    for (int i = pos*16; i < n; ++i) {
        for (int j = 0; j < 16; ++j) {
            out_counters[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

#undef BLOCK
#undef ITERATION
#undef UPDATE

    return 0;
}

int pospopcnt_u16_avx2(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m256i masks[16];
    __m256i counters[16];
    for (int i = 0; i < 16; ++i) {
        masks[i]    = _mm256_set1_epi16(1 << i);
        counters[i] = _mm256_set1_epi16(0);
    }
    uint32_t out_counters[16] = {0};

    const __m256i hi_mask = _mm256_set1_epi32(0xFFFF0000);
    const __m256i lo_mask = _mm256_set1_epi32(0x0000FFFF);
    const __m256i* data_vectors = (const __m256i*)(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_update_cycles = n_cycles / 65536;

#define UPDATE(idx) counters[idx]  = _mm256_add_epi16(counters[idx],  _mm256_srli_epi16(_mm256_and_si256(_mm256_loadu_si256(data_vectors+pos), masks[idx]),  idx))
#define ITERATION  {                               \
    UPDATE(0);  UPDATE(1);  UPDATE(2);  UPDATE(3); \
    UPDATE(4);  UPDATE(5);  UPDATE(6);  UPDATE(7); \
    UPDATE(8);  UPDATE(9);  UPDATE(10); UPDATE(11);\
    UPDATE(12); UPDATE(13); UPDATE(14); UPDATE(15);\
    ++pos;                                         \
}
    uint32_t pos = 0;
    for (int i = 0; i < n_update_cycles; ++i) { // each block of 2^16 values
        for (int k = 0; k < 65536; ++k) // max sum of each 16-bit value in a register
            ITERATION // unrolled

        // Compute vector sum
        for (int k = 0; k < 16; ++k) { // each flag register
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
            uint16_t* d = (uint16_t*)(&counters[k]);
            for (int j = 0; j < 16; ++j) { // each uint16_t in the register
                out_counters[k] += d[j];
            }
            */

            counters[k] = _mm256_set1_epi16(0);
        }
    }

    // residual
    for (int i = pos*16; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

#undef ITERATION
#undef UPDATE

    return 0;
}

int pospopcnt_u16_avx2_naive_counter(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m256i masks[16];
    __m256i counters[16];
    for (int i = 0; i < 16; ++i) {
        masks[i]    = _mm256_set1_epi16(1 << i);
        counters[i] = _mm256_set1_epi16(0);
    }
    uint32_t out_counters[16] = {0};

    const __m256i* data_vectors = (const __m256i*)(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_update_cycles = n_cycles / 65536;
    //std::cerr << n << " values and " << n_cycles << " cycles " << n_residual << " residual cycles" << std::endl;

#define UPDATE(idx) counters[idx]  = _mm256_add_epi16(counters[idx],  _mm256_srli_epi16(_mm256_and_si256(_mm256_loadu_si256(data_vectors+pos), masks[idx]),  idx))

    uint32_t pos = 0;
    for (int i = 0; i < n_update_cycles; ++i) { // each block of 2^16 values
        for (int k = 0; k < 65536; ++pos,++k) { // max sum of each 16-bit value in a register
            for (int p = 0; p < 16; ++p) // Not unrolled
                UPDATE(p);
        }

        // Compute vector sum
        for (int k = 0; k < 16; ++k) { // each flag register
            // Naive counter
            uint16_t* d = (uint16_t*)(&counters[k]);
            for (int j = 0; j < 16; ++j) // each uint16_t in the register
                out_counters[k] += d[j];

            counters[k] = _mm256_set1_epi16(0);
        }
    }

    // residual
    for (int i = pos*16; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

#undef UPDATE

    return 0;
}

int pospopcnt_u16_avx2_single(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m256i counter = _mm256_set1_epi16(0);
    const __m256i one_mask =  _mm256_set1_epi16(1);
    // set_epi is parameterized backwards (15->0)
    const __m256i masks = _mm256_set_epi16(1 << 15, 1 << 14, 1 << 13, 1 << 12,
                                           1 << 11, 1 << 10, 1 << 9,  1 << 8,
                                           1 << 7,  1 << 6,  1 << 5,  1 << 4,
                                           1 << 3,  1 << 2,  1 << 1,  1 << 0);
    uint32_t out_counters[16] = {0};
    const __m256i* data_vectors = (const __m256i*)(data);
    const uint32_t n_cycles = n / 16;
    const uint32_t n_update_cycles = n_cycles / 4096;

#define UPDATE(idx) counter = _mm256_add_epi16(counter, _mm256_and_si256(_mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(_mm256_extract_epi16(_mm256_loadu_si256(data_vectors+pos), idx)), masks), masks), one_mask));
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

    uint32_t pos = 0;
    for (int i = 0; i < n_update_cycles; ++i) { // each block of 65536 values
        for (int k = 0; k < 4096; ++k, ++pos) { // max sum of each 16-bit value in a register (65536/16)
            BLOCK
        }

        // Compute vector sum
        // Unroll to prevent clang (LLVM) from throwing constant integer error such as:
        // error: argument to '__builtin_ia32_vec_ext_v16hi' must be a constant integer
        out_counters[0]  += _mm256_extract_epi16(counter,  0);
        out_counters[1]  += _mm256_extract_epi16(counter,  1);
        out_counters[2]  += _mm256_extract_epi16(counter,  2);
        out_counters[3]  += _mm256_extract_epi16(counter,  3);
        out_counters[4]  += _mm256_extract_epi16(counter,  4);
        out_counters[5]  += _mm256_extract_epi16(counter,  5);
        out_counters[6]  += _mm256_extract_epi16(counter,  6);
        out_counters[7]  += _mm256_extract_epi16(counter,  7);
        out_counters[8]  += _mm256_extract_epi16(counter,  8);
        out_counters[9]  += _mm256_extract_epi16(counter,  9);
        out_counters[10] += _mm256_extract_epi16(counter, 10);
        out_counters[11] += _mm256_extract_epi16(counter, 11);
        out_counters[12] += _mm256_extract_epi16(counter, 12);
        out_counters[13] += _mm256_extract_epi16(counter, 13);
        out_counters[14] += _mm256_extract_epi16(counter, 14);
        out_counters[15] += _mm256_extract_epi16(counter, 15);

        counter = _mm256_set1_epi16(0);
    }

#undef UPDATE
#undef BLOCK

    // residual
    for (int i = pos*16; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    return 0;
}
#else
int pospopcnt_u16_avx2_popcnt(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_naive_counter(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_single(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
#endif

#if SIMD_VERSION >= 3
int pospopcnt_u16_sse_single(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m128i counterLo = _mm_set1_epi16(0);
    __m128i counterHi = _mm_set1_epi16(0);
    const __m128i one_mask =  _mm_set1_epi16(1);
    // set_epi is parameterized backwards (15->0)
    const __m128i masksLo = _mm_set_epi16(1 << 15, 1 << 14, 1 << 13, 1 << 12,
                                          1 << 11, 1 << 10, 1 << 9,  1 << 8);
    const __m128i masksHi = _mm_set_epi16(1 << 7,  1 << 6,  1 << 5,  1 << 4,
                                          1 << 3,  1 << 2,  1 << 1,  1 << 0);

    uint32_t out_counters[16] = {0};
    const __m128i* data_vectors = (const __m128i*)(data);
    const uint32_t n_cycles = n / 8;
    const uint32_t n_update_cycles = n_cycles / 4096;

#define UPDATE_LO(idx) counterLo = _mm_add_epi16(counterLo, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(_mm_extract_epi16(_mm_loadu_si128(data_vectors+pos), idx)), masksLo), masksLo), one_mask));
#define UPDATE_HI(idx) counterHi = _mm_add_epi16(counterHi, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(_mm_extract_epi16(_mm_loadu_si128(data_vectors+pos), idx)), masksHi), masksHi), one_mask));
#define BLOCK {                                         \
    UPDATE_LO(0) UPDATE_LO(1) UPDATE_LO(2) UPDATE_LO(3) \
    UPDATE_LO(4) UPDATE_LO(5) UPDATE_LO(6) UPDATE_LO(7) \
    UPDATE_HI(0) UPDATE_HI(1) UPDATE_HI(2) UPDATE_HI(3) \
    UPDATE_HI(4) UPDATE_HI(5) UPDATE_HI(6) UPDATE_HI(7) \
}
#define UH(idx) out_counters[idx] += _mm_extract_epi16(counterLo, idx - 8);
#define UL(idx) out_counters[idx] += _mm_extract_epi16(counterHi, idx);

    uint32_t pos = 0;
    for (int i = 0; i < n_update_cycles; ++i) { // each block of 65536 values
        for (int k = 0; k < 4096; ++k, ++pos) { // max sum of each 16-bit value in a register (65536/16)
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
    for (int i = pos*8; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    return 0;
}

// By @aqrit (https://github.com/aqrit)
// @see: https://gist.github.com/aqrit/cb52b2ac5b7d0dfe9319c09d27237bf3
int pospopcnt_u16_sse_sad(const uint16_t* data, uint32_t n, uint32_t* flag_counts) {
    const __m128i zero = _mm_setzero_si128();
    const __m128i mask_lo_byte = _mm_srli_epi16(_mm_cmpeq_epi8(zero, zero), 8);
    const __m128i mask_lo_cnt  = _mm_srli_epi16(mask_lo_byte, 2);
    const __m128i mask_bits_a  = _mm_set1_epi8(0x41);
    const __m128i mask_bits_b  = _mm_add_epi8(mask_bits_a, mask_bits_a);

    __m128i counterA = zero;
    __m128i counterB = zero;
    __m128i counterC = zero;
    __m128i counterD = zero;

    for (const uint16_t* end = &data[(n & ~31)]; data != end; data += 32) {
        __m128i r0 = _mm_loadu_si128((__m128i*)&data[0]);
        __m128i r1 = _mm_loadu_si128((__m128i*)&data[8]);
        __m128i r2 = _mm_loadu_si128((__m128i*)&data[16]);
        __m128i r3 = _mm_loadu_si128((__m128i*)&data[24]);
        __m128i r4, r5, r6, r7;

        // seperate LOBYTE and HIBYTE of each WORD
        // (emulate PSHUFB F,D,B,9,7,5,3,1, E,C,A,8,6,4,2,0)
        r4 = _mm_and_si128(mask_lo_byte, r0);
        r5 = _mm_and_si128(mask_lo_byte, r1);
        r6 = _mm_and_si128(mask_lo_byte, r2);
        r7 = _mm_and_si128(mask_lo_byte, r3);
        r0 = _mm_srli_epi16(r0, 8);
        r1 = _mm_srli_epi16(r1, 8);
        r2 = _mm_srli_epi16(r2, 8);
        r3 = _mm_srli_epi16(r3, 8);
        r0 = _mm_packus_epi16(r0, r4);
        r1 = _mm_packus_epi16(r1, r5);
        r2 = _mm_packus_epi16(r2, r6);
        r3 = _mm_packus_epi16(r3, r7);

        // isolate bits to count
        r4 = _mm_and_si128(mask_bits_a, r0);
        r5 = _mm_and_si128(mask_bits_a, r1);
        r6 = _mm_and_si128(mask_bits_a, r2);
        r7 = _mm_and_si128(mask_bits_a, r3);

        // horizontal sum of qwords
        r4 = _mm_sad_epu8(r4, zero);
        r5 = _mm_sad_epu8(r5, zero);
        r6 = _mm_sad_epu8(r6, zero);
        r7 = _mm_sad_epu8(r7, zero);

        // sum 6-bit counts
        r4 = _mm_add_epi16(r4,r5);
        r4 = _mm_add_epi16(r4,r6);
        r4 = _mm_add_epi16(r4,r7);

        // unpack 6-bit counts to 32-bits
        r5 = _mm_and_si128(mask_lo_cnt, r4);
        r4 = _mm_srli_epi16(r4, 6);
        r4 = _mm_packs_epi32(r4, r5);

        // accumulate
        counterA = _mm_add_epi32(counterA, r4);

        // do it again...
        r4 = _mm_and_si128(mask_bits_b, r0);
        r5 = _mm_and_si128(mask_bits_b, r1);
        r6 = _mm_and_si128(mask_bits_b, r2);
        r7 = _mm_and_si128(mask_bits_b, r3);

        r4 = _mm_sad_epu8(r4, zero);
        r5 = _mm_sad_epu8(r5, zero);
        r6 = _mm_sad_epu8(r6, zero);
        r7 = _mm_sad_epu8(r7, zero);

        r4 = _mm_add_epi16(r4,r5);
        r4 = _mm_add_epi16(r4,r6);
        r4 = _mm_add_epi16(r4,r7);

        r5 = _mm_avg_epu8(zero, r4); // shift right 1
        r5 = _mm_and_si128(r5, mask_lo_cnt);
        r4 = _mm_srli_epi16(r4, 7);
        r4 = _mm_packs_epi32(r4, r5);

        counterB = _mm_add_epi32(counterB, r4); // accumulate

        // rotate right 4
        r4 = _mm_slli_epi16(r0, 12);
        r5 = _mm_slli_epi16(r1, 12);
        r6 = _mm_slli_epi16(r2, 12);
        r7 = _mm_slli_epi16(r3, 12);
        r0 = _mm_srli_epi16(r0, 4);
        r1 = _mm_srli_epi16(r1, 4);
        r2 = _mm_srli_epi16(r2, 4);
        r3 = _mm_srli_epi16(r3, 4);
        r0 = _mm_or_si128(r0, r4);
        r1 = _mm_or_si128(r1, r5);
        r2 = _mm_or_si128(r2, r6);
        r3 = _mm_or_si128(r3, r7);

        // do it again...
        r4 = _mm_and_si128(mask_bits_a, r0);
        r5 = _mm_and_si128(mask_bits_a, r1);
        r6 = _mm_and_si128(mask_bits_a, r2);
        r7 = _mm_and_si128(mask_bits_a, r3);

        r4 = _mm_sad_epu8(r4, zero);
        r5 = _mm_sad_epu8(r5, zero);
        r6 = _mm_sad_epu8(r6, zero);
        r7 = _mm_sad_epu8(r7, zero);

        r4 = _mm_add_epi16(r4,r5);
        r4 = _mm_add_epi16(r4,r6);
        r4 = _mm_add_epi16(r4,r7);

        r5 = _mm_and_si128(mask_lo_cnt, r4);
        r4 = _mm_srli_epi16(r4, 6);
        r4 = _mm_packs_epi32(r4, r5);

        counterC = _mm_add_epi32(counterC, r4); // accumulate

        // do it again...
        r0 = _mm_and_si128(r0, mask_bits_b);
        r1 = _mm_and_si128(r1, mask_bits_b);
        r2 = _mm_and_si128(r2, mask_bits_b);
        r3 = _mm_and_si128(r3, mask_bits_b);

        r0 = _mm_sad_epu8(r0, zero);
        r1 = _mm_sad_epu8(r1, zero);
        r2 = _mm_sad_epu8(r2, zero);
        r3 = _mm_sad_epu8(r3, zero);

        r0 = _mm_add_epi16(r0,r1);
        r0 = _mm_add_epi16(r0,r2);
        r0 = _mm_add_epi16(r0,r3);

        r1 = _mm_avg_epu8(zero, r0);
        r1 = _mm_and_si128(r1, mask_lo_cnt);
        r0 = _mm_srli_epi16(r0, 7);
        r0 = _mm_packs_epi32(r0, r1);

        counterD = _mm_add_epi32(counterD, r0); // accumulate
    }

    // transpose then store counters
    __m128i counter_1098 = _mm_unpackhi_epi32(counterA, counterB);
    __m128i counter_76FE = _mm_unpacklo_epi32(counterA, counterB);
    __m128i counter_32BA = _mm_unpacklo_epi32(counterC, counterD);
    __m128i counter_54DC = _mm_unpackhi_epi32(counterC, counterD);
    __m128i counter_7654 = _mm_unpackhi_epi64(counter_54DC, counter_76FE);
    __m128i counter_FEDC = _mm_unpacklo_epi64(counter_54DC, counter_76FE);
    __m128i counter_3210 = _mm_unpackhi_epi64(counter_1098, counter_32BA);
    __m128i counter_BA98 = _mm_unpacklo_epi64(counter_1098, counter_32BA);
    _mm_storeu_si128((__m128i*)&flag_counts[0], counter_3210);
    _mm_storeu_si128((__m128i*)&flag_counts[4], counter_7654);
    _mm_storeu_si128((__m128i*)&flag_counts[8], counter_BA98);
    _mm_storeu_si128((__m128i*)&flag_counts[12], counter_FEDC);

    // scalar tail loop
    int tail = n & 31;
    if (tail != 0) {
        uint64_t countsA = 0;
        uint64_t countsB = 0;
        do {
            // zero-extend a bit to 8-bits then accumulate
            // (emulate pdep)
            const uint64_t mask_01 = UINT64_C(0x0101010101010101);
            const uint64_t magic = UINT64_C(0x0000040010004001); // 1+(1<<14)+(x<<28)+(x<<42)
            uint64_t x = *data++;
            countsA += ((x & 0x5555) * magic) & mask_01;
            countsB += (((x >> 1) & 0x5555) * magic) & mask_01;
        } while(--tail);

        // transpose then store counters
        flag_counts[0]  += countsA & 0xFF;
        flag_counts[8]  += (countsA >> 8) & 0xFF;
        flag_counts[2]  += (countsA >> 16) & 0xFF;
        flag_counts[10] += (countsA >> 24) & 0xFF;
        flag_counts[4]  += (countsA >> 32) & 0xFF;
        flag_counts[12] += (countsA >> 40) & 0xFF;
        flag_counts[6]  += (countsA >> 48) & 0xFF;
        flag_counts[14] += (countsA >> 56) & 0xFF;
        flag_counts[1]  += countsB & 0xFF;
        flag_counts[9]  += (countsB >> 8) & 0xFF;
        flag_counts[3]  += (countsB >> 16) & 0xFF;
        flag_counts[11] += (countsB >> 24) & 0xFF;
        flag_counts[5]  += (countsB >> 32) & 0xFF;
        flag_counts[13] += (countsB >> 40) & 0xFF;
        flag_counts[7]  += (countsB >> 48) & 0xFF;
        flag_counts[15] += (countsB >> 56) & 0xFF;
    }

    return 0;
}
#else
int pospopcnt_u16_sse_single(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_sse2_sad(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
#endif

 __attribute__((optimize("no-tree-vectorize")))
int pospopcnt_u16_scalar_naive_nosimd(const uint16_t* data, uint32_t n, uint32_t* flags) {
    memset(flags, 0, 16*sizeof(uint32_t));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    return 0;
}

int pospopcnt_u16_scalar_naive(const uint16_t* data, uint32_t n, uint32_t* flags) {
    memset(flags, 0, 16*sizeof(uint32_t));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    return 0;
}

int pospopcnt_u16_scalar_partition(const uint16_t* data, uint32_t n, uint32_t* flags) {
    uint32_t low[256] = {0}, high[256] = {0};
    memset(flags, 0, 16*sizeof(uint32_t));

    for (int i = 0; i < n; ++i) {
        ++low[data[i] & 255];
        ++high[(data[i] >> 8) & 255];
    }

    for (int i = 0; i < 256; ++i) {
        for (int k = 0; k < 8; ++k) {
            flags[k] += ((i & (1 << k)) >> k) * low[i];
        }
    }

    for (int i = 0; i < 256; ++i) {
        for (int k = 0; k < 8; ++k) {
            flags[k+8] += ((i & (1 << k)) >> k) * high[i];
        }
    }

    return 0;
}

int pospopcnt_u16_hist1x4(const uint16_t* data, uint32_t n, uint32_t* flags) {
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

     for (int i = 0; i < 256; ++i) {
        for (int k = 0; k < 8; ++k) {
            flags[k] += ((i & (1 << k)) >> k) * low[i];
        }
    }

    for (int i = 0; i < 256; ++i) {
        for (int k = 0; k < 8; ++k) {
            flags[k+8] += ((i & (1 << k)) >> k) * high[i];
        }
    }

    return 0;
}

#if SIMD_VERSION >= 6
int pospopcnt_u16_avx512_popcnt32_mask(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m512i masks[16];
    for (int i = 0; i < 16; ++i) {
        masks[i] = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
    }
    uint32_t out_counters[16] = {0};

    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = n / 32;

#define UPDATE(pos) out_counters[pos] += PIL_POPCOUNT((uint64_t)_mm512_cmpeq_epu16_mask(_mm512_and_epi32(_mm512_loadu_si512(data_vectors+i), masks[pos]), masks[pos]));
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

    for (int i = 0; i < n_cycles; ++i) {
        BLOCK
    }

    // residual
    for (int i = n_cycles*32; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

#undef BLOCK
#undef UPDATE

    return 0;
}

int pospopcnt_u16_avx512_popcnt64_mask(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m512i masks[16];
    for (int i = 0; i < 16; ++i) {
        masks[i] = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
    }
    uint32_t out_counters[16] = {0};

    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = n / 32;

#define UPDATE(pos,add) (uint64_t)_mm512_cmpeq_epu16_mask(_mm512_and_epi32(_mm512_loadu_si512(data_vectors+i+add), masks[pos]), masks[pos])
#define UP(pos) out_counters[pos] += PIL_POPCOUNT((UPDATE(pos,0) << 32) | UPDATE(pos,1));
#define BLOCK {                 \
    UP(0)  UP(1)  UP(2)  UP(3)  \
    UP(4)  UP(5)  UP(6)  UP(7)  \
    UP(8)  UP(9)  UP(10) UP(11) \
    UP(12) UP(13) UP(14) UP(15) \
}
    
    int i = 0;
    for (/**/; i + 2 < n_cycles; i += 2) {
        BLOCK
    }

    // residual
    i *= 32;
    for (/**/; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

#undef BLOCK
#undef UP
#undef UPDATE

    return 0;
}

int pospopcnt_u16_avx512_popcnt(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m512i masks[16];
    __m512i counters[16];
    __m512i stubs[16];
    for (int i = 0; i < 16; ++i) {
        masks[i]    = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
        counters[i] = _mm512_set1_epi32(0);
        stubs[i] = _mm512_set1_epi32(0);
    }
    uint32_t out_counters[16] = {0};

    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = n / 32;

#define UPDATE(pos,shift) stubs[pos] = _mm512_or_si512(stubs[pos], _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(_mm512_loadu_si512(data_vectors+i+j), masks[pos]), pos), shift));
#define BLOCK(shift) {                                                  \
    UPDATE(0,shift)  UPDATE(1,shift)  UPDATE(2,shift)  UPDATE(3,shift)  \
    UPDATE(4,shift)  UPDATE(5,shift)  UPDATE(6,shift)  UPDATE(7,shift)  \
    UPDATE(8,shift)  UPDATE(9,shift)  UPDATE(10,shift) UPDATE(11,shift) \
    UPDATE(12,shift) UPDATE(13,shift) UPDATE(14,shift) UPDATE(15,shift) \
}
#define UC(pos) {                         \
    counters[pos] = _mm512_add_epi32(counters[pos], avx512_popcount(stubs[pos])); \
    stubs[pos]    = _mm512_set1_epi32(0); \
}
#define UC_BLOCK {              \
    UC(0)  UC(1)  UC(2)  UC(3)  \
    UC(4)  UC(5)  UC(6)  UC(7)  \
    UC(8)  UC(9)  UC(10) UC(11) \
    UC(12) UC(13) UC(14) UC(15) \
}

    int i = 0;
    for (/**/; i + 16 < n_cycles; i += 16) {
        for (int j = 0; j < 16; ++j) {
            BLOCK(j)
        }

        UC_BLOCK
    }

#undef UC_BLOCK
#undef UC
#undef BLOCK
#undef UPDATE

    // residual
    i *= 32;
    for (/**/; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) {
        uint32_t* v = (uint32_t*)(&counters[i]);
        for (int j = 0; j < 16; ++j)
            out_counters[i] += v[j];
    }
    
    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    return 0;
}

int pospopcnt_u16_avx512(const uint16_t* data, uint32_t n, uint32_t* flags) {
    __m512i masks[16];
    __m512i counters[16];
    const __m512i one_mask = _mm512_set1_epi32(1);
    for (int i = 0; i < 16; ++i) {
        masks[i]    = _mm512_set1_epi16(1 << i);
        counters[i] = _mm512_set1_epi32(0);
    }
    uint32_t out_counters[16] = {0};

    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = n / 32;

#define UPDATE(pos) {                                            \
    __m512i a   = _mm512_and_epi32(_mm512_loadu_si512(data_vectors+i), masks[pos]); \
    __m512i d   = _mm512_add_epi32(_mm512_and_epi32(_mm512_srli_epi32(a, pos), one_mask), _mm512_srli_epi32(a, pos+16)); \
    counters[pos] = _mm512_add_epi32(counters[pos], d);          \
}
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

    for (int i = 0; i < n_cycles; ++i) {
        BLOCK
    }

#undef BLOCK
#undef UPDATE

    // residual
    for (int i = n_cycles*32; i < n; ++i) {
        for (int j = 0; j < 16; ++j)
            out_counters[j] += ((data[i] & (1 << j)) >> j);
    }

    for (int i = 0; i < 16; ++i) {
        uint32_t* v = (uint32_t*)(&counters[i]);
        for (int j = 0; j < 16; ++j)
            out_counters[i] += v[j];
    }
 
    for (int i = 0; i < 16; ++i) flags[i] = out_counters[i];

    return 0;
}

#else
int pospopcnt_u16_avx512_popcnt32_mask(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512_popcnt(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512_popcnt64_mask(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
#endif

#if SIMD_VERSION >= 5

// by D. Lemire
int pospopcnt_u16_avx2_lemire(const uint16_t *array, uint32_t len, uint32_t *flags) {
    uint16_t buffer[16];
    __m256i bits = _mm256_set_epi16(-32768, 16384, 8192, 4096, 2048, 1024, 512, 256,
                                       128,    64,   32,   16,    8,    4,   2,   1);
    // we do the first part
    if (len < 64) {
        // don't bother with handcrafted SIMD
        pospopcnt_u16_scalar_naive(array, len, flags);
        return 0;
    }
    // handle the start (naively)
    __m256i count16 = _mm256_setzero_si256();

    {
        uint16_t startbuffer[32];
        memset(startbuffer, 0, 32 * 2);
        memcpy(startbuffer + 16, array, 16 * 2);
        for (size_t i = 1; i < 16; i++) {
            __m256i input = _mm256_loadu_si256((__m256i*)(startbuffer + i));
            __m256i m = _mm256_and_si256(input, bits);
            __m256i eq = _mm256_cmpeq_epi16(bits, m);
            count16 = _mm256_sub_epi16(count16, eq);
        }
    }

    {
        uint16_t startbuffer[32];
        memset(startbuffer, 0, 32 * 2);
        memcpy(startbuffer, array + len - 16, 16 * 2);
        for (size_t i = 1; i < 16; i++) {
            __m256i input = _mm256_loadu_si256((__m256i*)(startbuffer + i));
            __m256i m = _mm256_and_si256(input, bits);
            __m256i eq = _mm256_cmpeq_epi16(bits, m);
            count16 = _mm256_sub_epi16(count16, eq);
        }
    }
    
    _mm256_storeu_si256((__m256i*)buffer, count16);
    for (size_t k = 0; k < 16; k++) {
        flags[k] += buffer[k];
    }

    // main loop starts here
    for (size_t i = 0; i + 16 <= len;) {
        count16 = _mm256_setzero_si256();
        size_t j = 0;
        size_t maxj = 65535;

        if (maxj + i + 16 >= len)
            maxj = len - i - 15;

        for (; j < maxj; j++) {
            __m256i input = _mm256_loadu_si256((__m256i*)(array + i + j));
            __m256i m = _mm256_and_si256(input, bits);
            __m256i eq = _mm256_cmpeq_epi16(bits, m);
            count16 = _mm256_sub_epi16(count16, eq);
        }
        i += j;
        _mm256_storeu_si256((__m256i*)buffer, count16);
        for (size_t k = 0; k < 16; k++) {
            flags[k] += buffer[k];
        }
    }
    return 0;
}

// By D. Lemire
int pospopcnt_u16_avx2_lemire2(const uint16_t* array, uint32_t len, uint32_t* flags) {
    uint16_t buffer[16];
    __m256i bits = _mm256_set_epi16(-32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 
                                       128,    64,   32,   16,    8,    4,   2,   1);
    // we do the first part
    if (len < 64) {
        // don't bother with handcrafted SIMD
        pospopcnt_u16_scalar_naive(array, len, flags);
        return 0;
    }
    // handle the start (naively)
    __m256i count16 = _mm256_setzero_si256();

    {
        uint16_t startbuffer[32];
        memset(startbuffer, 0, 32 * 2);
        memcpy(startbuffer + 16, array, 16 * 2);
        for (size_t i = 1; i < 16; i++) {
            __m256i input = _mm256_loadu_si256((__m256i*)(startbuffer + i));
            __m256i m = _mm256_and_si256(input, bits);
            __m256i eq = _mm256_cmpeq_epi16(bits, m);
            count16 = _mm256_sub_epi16(count16, eq);
        }
    }

    {
        uint16_t startbuffer[32];
        memset(startbuffer, 0, 32 * 2);
        memcpy(startbuffer, array + len - 16, 16 * 2);
        for (size_t i = 1; i < 16; i++) {
            __m256i input = _mm256_loadu_si256((__m256i*)(startbuffer + i));
            __m256i m = _mm256_and_si256(input, bits);
            __m256i eq = _mm256_cmpeq_epi16(bits, m);
            count16 = _mm256_sub_epi16(count16, eq);
        }
    }

    _mm256_storeu_si256((__m256i*)buffer, count16);
    for (size_t k = 0; k < 16; k++) {
        flags[k] += buffer[k];
    }

    // main loop starts here
    for (size_t i = 0; i + 16 <= len;) {
        count16 = _mm256_setzero_si256();
        size_t j = 0;
        size_t maxj = 65535;

        if (maxj + i + 16 >= len)
            maxj = len - i - 15;

        if (maxj > 8) {
            for (; j < maxj - 7; j += 8) {
                __m256i input1 = _mm256_loadu_si256((__m256i*)(array + i + j));
                __m256i m1 = _mm256_and_si256(input1, bits);
                __m256i eq1 = _mm256_cmpeq_epi16(bits, m1);
                count16 = _mm256_sub_epi16(count16, eq1);
                __m256i input2 = _mm256_loadu_si256((__m256i*)(array + i + j + 1));
                __m256i m2 = _mm256_and_si256(input2, bits);
                __m256i eq2 = _mm256_cmpeq_epi16(bits, m2);
                count16 = _mm256_sub_epi16(count16, eq2);
                __m256i input3 = _mm256_loadu_si256((__m256i*)(array + i + j + 2));
                __m256i m3 = _mm256_and_si256(input3, bits);
                __m256i eq3 = _mm256_cmpeq_epi16(bits, m3);
                count16 = _mm256_sub_epi16(count16, eq3);
                __m256i input4 = _mm256_loadu_si256((__m256i*)(array + i + j + 3));
                __m256i m4 = _mm256_and_si256(input4, bits);
                __m256i eq4 = _mm256_cmpeq_epi16(bits, m4);
                count16 = _mm256_sub_epi16(count16, eq4);
                __m256i input5 = _mm256_loadu_si256((__m256i*)(array + i + j + 4));
                __m256i m5 = _mm256_and_si256(input5, bits);
                __m256i eq5 = _mm256_cmpeq_epi16(bits, m5);
                count16 = _mm256_sub_epi16(count16, eq5);
                __m256i input6 = _mm256_loadu_si256((__m256i*)(array + i + j + 5));
                __m256i m6 = _mm256_and_si256(input6, bits);
                __m256i eq6 = _mm256_cmpeq_epi16(bits, m6);
                count16 = _mm256_sub_epi16(count16, eq6);
                __m256i input7 = _mm256_loadu_si256((__m256i*)(array + i + j + 6));
                __m256i m7 = _mm256_and_si256(input7, bits);
                __m256i eq7 = _mm256_cmpeq_epi16(bits, m7);
                count16 = _mm256_sub_epi16(count16, eq7);
                __m256i input8 = _mm256_loadu_si256((__m256i*)(array + i + j + 7));
                __m256i m8 = _mm256_and_si256(input8, bits);
                __m256i eq8 = _mm256_cmpeq_epi16(bits, m8);
                count16 = _mm256_sub_epi16(count16, eq8);
            }
        }

        if (maxj > 4) {
            for (; j + 3 < maxj; j += 4) {
                __m256i input1 = _mm256_loadu_si256((__m256i*)(array + i + j));
                __m256i m1 = _mm256_and_si256(input1, bits);
                __m256i eq1 = _mm256_cmpeq_epi16(bits, m1);
                count16 = _mm256_sub_epi16(count16, eq1);
                __m256i input2 = _mm256_loadu_si256((__m256i*)(array + i + j + 1));
                __m256i m2 = _mm256_and_si256(input2, bits);
                __m256i eq2 = _mm256_cmpeq_epi16(bits, m2);
                count16 = _mm256_sub_epi16(count16, eq2);
                __m256i input3 = _mm256_loadu_si256((__m256i*)(array + i + j + 2));
                __m256i m3 = _mm256_and_si256(input3, bits);
                __m256i eq3 = _mm256_cmpeq_epi16(bits, m3);
                count16 = _mm256_sub_epi16(count16, eq3);
                __m256i input4 = _mm256_loadu_si256((__m256i*)(array + i + j + 3));
                __m256i m4 = _mm256_and_si256(input4, bits);
                __m256i eq4 = _mm256_cmpeq_epi16(bits, m4);
                count16 = _mm256_sub_epi16(count16, eq4);
            }
        }
        for (; j < maxj; j++) {
            __m256i input = _mm256_loadu_si256((__m256i*)(array + i + j));
            __m256i m = _mm256_and_si256(input, bits);
            __m256i eq = _mm256_cmpeq_epi16(bits, m);
            count16 = _mm256_sub_epi16(count16, eq);
        }
        i += j;

        _mm256_storeu_si256((__m256i*)buffer, count16);
        for (size_t k = 0; k < 16; k++) {
            flags[k] += buffer[k];
        }
    }
    return 0;
}

int pospopcnt_u16_avx2_mula2(const uint16_t* array, uint32_t len, uint32_t* flags) {
    __m256i counters[16];

    for (size_t i = 0; i < 16; i++) {
        counters[i] = _mm256_setzero_si256();
    }

    size_t i = 0;
    for (/**/; i + 16 <= len; i += 16) {
        __m256i input = _mm256_loadu_si256((__m256i*)(array + i));

        for (int j = 0; j < 16; j++) {
            __m256i bit = _mm256_and_si256(input, _mm256_set1_epi16(1));
            counters[j] = _mm256_add_epi16(counters[j], bit);
            input = _mm256_srli_epi16(input, 1);
        }
    }

     // residual
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

    uint16_t tmp[16];
    for (size_t i = 0; i < 16; i++) {
        _mm256_storeu_si256((__m256i*)tmp, counters[i]);
        for (int j=0; j < 16; j++)
            flags[i] += tmp[j];
    }
    return 0;
}

// By Daniel Lemire
int pospopcnt_u16_avx2_mula(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m256i* data_vectors = (const __m256i*)(array);
    const uint32_t n_cycles = len / 16;

    size_t i = 0;
    for (/**/; i + 2 <= n_cycles; i += 2) {
        __m256i v0 = _mm256_loadu_si256(data_vectors+i+0); // don't assume alignment
        __m256i v1 = _mm256_loadu_si256(data_vectors+i+1); // don't assume alignment
        
        // Steps:
        // Select LSB of V0 OR with V1 MSB
        // Select MSB of V0 OR with V1 LSB
        __m256i input0 = _mm256_or_si256(_mm256_and_si256(v0, _mm256_set1_epi16(0x00FF)), _mm256_slli_epi16(v1, 8));
        __m256i input1 = _mm256_or_si256(_mm256_and_si256(v0, _mm256_set1_epi16(0xFF00)), _mm256_srli_epi16(v1, 8));
        
        for (int i = 0; i < 8; i++) {
            flags[ 7 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input0));
            flags[15 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input1));
            input0 = _mm256_add_epi8(input0, input0);
            input1 = _mm256_add_epi8(input1, input1);
        }
    }

    i *= 16;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }
   
    return 0;
}

int pospopcnt_u16_avx2_mula3(const uint16_t* array, uint32_t len, uint32_t* flags) {
    __m256i counters[16];

    for (size_t i = 0; i < 16; i++) {
        counters[i] = _mm256_setzero_si256();
    }

    const __m256i mask1bit = _mm256_set1_epi16(0x5555); // 0101010101010101
    const __m256i mask2bit = _mm256_set1_epi16(0x3333); // 0011001100110011
    const __m256i mask4bit = _mm256_set1_epi16(0x0f0f); // 1111000011110000
    const __m256i mask8bit = _mm256_set1_epi16(0x00ff); // 0000000011111111
    
    const uint32_t n_cycles = len / (4096 * (16*16));
    const uint32_t n_total  = len / (16*16);
    uint16_t tmp[16];

/*------ Macros --------*/
#define LL(i,p,k)  const __m256i sum##p##k##_##i##bit_even = _mm256_add_epi8(input##p & mask##i##bit, input##k & mask##i##bit);
#define LO(i,p,k)  const __m256i sum##p##k##_##i##bit_odd  = _mm256_add_epi8(_mm256_srli_epi16(input##p, i) & mask##i##bit, _mm256_srli_epi16(input##k, i) & mask##i##bit);

#define LBLOCK(i)                                 \
    LL(i,0,1) LL(i,2,3)   LL(i,4,5)   LL(i,6,7)   \
    LL(i,8,9) LL(i,10,11) LL(i,12,13) LL(i,14,15) \
    LO(i,0,1) LO(i,2,3)   LO(i,4,5)   LO(i,6,7)   \
    LO(i,8,9) LO(i,10,11) LO(i,12,13) LO(i,14,15)

#define EVEN(b,i,k,p) input##i = sum##k##p##_##b##bit_even;
#define ODD(b,i,k,p)  input##i = sum##k##p##_##b##bit_odd;

#define UPDATE(i)                                                  \
    EVEN(i,0,0,1) EVEN(i,1,2,3)   EVEN(i,2,4,5)   EVEN(i,3,6,7)    \
    EVEN(i,4,8,9) EVEN(i,5,10,11) EVEN(i,6,12,13) EVEN(i,7,14,15)  \
     ODD(i,8,0,1)  ODD(i,9,2,3)    ODD(i,10,4,5)   ODD(i,11,6,7)   \
     ODD(i,12,8,9) ODD(i,13,10,11) ODD(i,14,12,13) ODD(i,15,14,15) \

#define UE(i,p,k) counters[i] = _mm256_add_epi16(counters[i], sum##p##k##_8bit_even);
#define UO(i,p,k) counters[i] = _mm256_add_epi16(counters[i], sum##p##k##_8bit_odd);

/*------ Start --------*/
#define L(p) __m256i input##p = _mm256_loadu_si256((__m256i*)(array + i*4096*256 + j*256 + p*16));
    size_t i = 0;
    for (/**/; i < n_cycles; ++i) {
        for (int j = 0; j < 4096; ++j) {
            // Load 16 registers.
            L(0)  L(1)  L(2)  L(3)  
            L(4)  L(5)  L(6)  L(7) 
            L(8)  L(9)  L(10) L(11) 
            L(12) L(13) L(14) L(15)

            // Perform updates for bits {1,2,4,8}.
            LBLOCK(1) UPDATE(1)
            LBLOCK(2) UPDATE(2)
            LBLOCK(4) UPDATE(4)
            LBLOCK(8) UPDATE(8)

            // Update accumulators.
            UE( 0,0,1) UE( 1, 2, 3) UE( 2, 4, 5) UE( 3, 6, 7)  
            UE( 4,8,9) UE( 5,10,11) UE( 6,12,13) UE( 7,14,15) 
            UO( 8,0,1) UO( 9, 2, 3) UO(10, 4, 5) UO(11, 6, 7) 
            UO(12,8,9) UO(13,10,11) UO(14,12,13) UO(15,14,15)
        }

        // Update.
        for (size_t i = 0; i < 16; i++) {
            _mm256_storeu_si256((__m256i*)tmp, counters[i]);
            for (int j = 0; j < 16; j++)
                flags[i] += tmp[j];
        }
        // Reset.
        for (size_t i = 0; i < 16; i++) {
            counters[i] = _mm256_setzero_si256();
        }
    }
#undef L
#define L(p) __m256i input##p = _mm256_loadu_si256((__m256i*)(array + i*256 + p*16));
    i *= 4096;
    for (/**/; i < n_total; ++i) {
        // Load 16 registers.
        L(0)  L(1)  L(2)  L(3)  
        L(4)  L(5)  L(6)  L(7) 
        L(8)  L(9)  L(10) L(11) 
        L(12) L(13) L(14) L(15)

        // Perform updates for bits {1,2,4,8}.
        LBLOCK(1) UPDATE(1)
        LBLOCK(2) UPDATE(2)
        LBLOCK(4) UPDATE(4)
        LBLOCK(8) UPDATE(8)

        // Update accumulators.
        UE( 0,0,1) UE( 1, 2, 3) UE( 2, 4, 5) UE( 3, 6, 7)  
        UE( 4,8,9) UE( 5,10,11) UE( 6,12,13) UE( 7,14,15) 
        UO( 8,0,1) UO( 9, 2, 3) UO(10, 4, 5) UO(11, 6, 7) 
        UO(12,8,9) UO(13,10,11) UO(14,12,13) UO(15,14,15)
    }

    i *= 256;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef UPDATE
#undef ODD
#undef EVEN
#undef LBLOCK
#undef LL
#undef LO
#undef UO
#undef UE

    for (size_t i = 0; i < 16; i++) {
        _mm256_storeu_si256((__m256i*)tmp, counters[i]);
        for (int j = 0; j < 16; j++)
            flags[i] += tmp[j];
    }
    return 0;
}

int pospopcnt_u16_avx2_csa(const uint16_t* array, uint32_t len, uint32_t* flags) {
    for (uint32_t i = len - (len % (16 * 16)); i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

    const __m256i*data = (const __m256i*)array;
    size_t size = len / 16;
    __m256i v1  = _mm256_setzero_si256();
    __m256i v2  = _mm256_setzero_si256();
    __m256i v4  = _mm256_setzero_si256();
    __m256i v8  = _mm256_setzero_si256();
    __m256i v16 = _mm256_setzero_si256();
    __m256i twosA, twosB, foursA, foursB, eightsA, eightsB;

    const uint64_t limit = size - size % 16;
    uint64_t i = 0;

    // /const uint32_t n_cycles = len / (4096 * (16*16));
    //const uint32_t n_total  = len / (16*16);
    uint16_t buffer[16];
    __m256i counter[16];

    // uint16_t x = 0;
    while (i < limit) {
        
        for (size_t i = 0; i < 16; i++) {
            counter[i] = _mm256_setzero_si256();
        }

        size_t thislimit = limit;
        if (thislimit - i >= (1 << 16))
            thislimit = i + (1 << 16) - 1;

        for (/**/; i < thislimit; i += 16) {
            POSPOPCNT_CSA_AVX2(&twosA,   &v1, _mm256_lddqu_si256(data + i + 0), _mm256_lddqu_si256(data + i + 1));
            POSPOPCNT_CSA_AVX2(&twosB,   &v1, _mm256_lddqu_si256(data + i + 2), _mm256_lddqu_si256(data + i + 3));
            POSPOPCNT_CSA_AVX2(&foursA,  &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX2(&twosA,   &v1, _mm256_lddqu_si256(data + i + 4), _mm256_lddqu_si256(data + i + 5));
            POSPOPCNT_CSA_AVX2(&twosB,   &v1, _mm256_lddqu_si256(data + i + 6), _mm256_lddqu_si256(data + i + 7));
            POSPOPCNT_CSA_AVX2(&foursB,  &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX2(&eightsA, &v4, foursA, foursB);
            POSPOPCNT_CSA_AVX2(&twosA,   &v1, _mm256_lddqu_si256(data + i + 8),  _mm256_lddqu_si256(data + i + 9));
            POSPOPCNT_CSA_AVX2(&twosB,   &v1, _mm256_lddqu_si256(data + i + 10), _mm256_lddqu_si256(data + i + 11));
            POSPOPCNT_CSA_AVX2(&foursA,  &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX2(&twosA,   &v1, _mm256_lddqu_si256(data + i + 12), _mm256_lddqu_si256(data + i + 13));
            POSPOPCNT_CSA_AVX2(&twosB,   &v1, _mm256_lddqu_si256(data + i + 14), _mm256_lddqu_si256(data + i + 15));
            POSPOPCNT_CSA_AVX2(&foursB,  &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX2(&eightsB, &v4, foursA, foursB);
            POSPOPCNT_CSA_AVX2(&v16,     &v8, eightsA, eightsB);

            for (size_t i = 0; i < 16; i++) {
                counter[i] = _mm256_add_epi16(counter[i], _mm256_and_si256(v16, _mm256_set1_epi16(1)));
                v16 = _mm256_srli_epi16(v16, 1);
            }
        }

        for (size_t i = 0; i < 16; i++) {
            _mm256_storeu_si256((__m256i*)buffer, counter[i]);
            for (size_t z = 0; z < 16; z++) {
                flags[i] += buffer[z] * 16;
            }
        }
    }

    _mm256_storeu_si256((__m256i*)buffer, v1);
    for (size_t i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += ((buffer[i] & (1 << j)) >> j);
        }
    }

    _mm256_storeu_si256((__m256i*)buffer, v2);
    for (size_t i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 2 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    _mm256_storeu_si256((__m256i*)buffer, v4);
    for (size_t i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 4 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    _mm256_storeu_si256((__m256i*)buffer, v8);
    for (size_t i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 8 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    return 0;
}

int pospopcnt_u16_avx2_mula_unroll4(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m256i* data_vectors = (const __m256i*)(array);
    const uint32_t n_cycles = len / 16;

    size_t i = 0;
    for (/**/; i + 4 <= n_cycles; i += 4) {
#define L(p) __m256i v##p = _mm256_loadu_si256(data_vectors+i+p);
        L(0) L(1) L(2) L(3)
        
#define U0(p,k) __m256i input##p = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0x00FF)), _mm256_slli_epi16(v##k, 8));
#define U1(p,k) __m256i input##k = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0xFF00)), _mm256_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)
       U(0,1) U(2, 3)
        
        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3)

#define P0(p) input##p = _mm256_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)
            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 16;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P
    
    return 0;
}

int pospopcnt_u16_avx2_mula_unroll8(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m256i* data_vectors = (const __m256i*)(array);
    const uint32_t n_cycles = len / 16;

    size_t i = 0;
    for (/**/; i + 8 <= n_cycles; i += 8) {
#define L(p) __m256i v##p = _mm256_loadu_si256(data_vectors+i+p);
        L(0) L(1) L(2) L(3)
        L(4) L(5) L(6) L(7) 
        
#define U0(p,k) __m256i input##p = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0x00FF)), _mm256_slli_epi16(v##k, 8));
#define U1(p,k) __m256i input##k = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0xFF00)), _mm256_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)
       U(0,1) U(2, 3) U(4, 5) U(6, 7)
        
        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3) A(4, 5) A(6, 7)

#define P0(p) input##p = _mm256_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)
            P(0,1) P(2, 3) P(4, 5) P(6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2, 3)
        
        for (int i = 0; i < 8; i++) {
            A(0,1) A( 2, 3)
            P(0,1) P( 2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 16;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P

    return 0;
}

int pospopcnt_u16_avx2_mula_unroll16(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m256i* data_vectors = (const __m256i*)(array);
    const uint32_t n_cycles = len / 16;

    size_t i = 0;
    for (/**/; i + 16 <= n_cycles; i += 16) {
#define L(p) __m256i v##p = _mm256_loadu_si256(data_vectors+i+p);
        L(0)  L(1)  L(2)  L(3)  
        L(4)  L(5)  L(6)  L(7) 
        L(8)  L(9)  L(10) L(11) 
        L(12) L(13) L(14) L(15)

#define U0(p,k) __m256i input##p = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0x00FF)), _mm256_slli_epi16(v##k, 8));
#define U1(p,k) __m256i input##k = _mm256_or_si256(_mm256_and_si256(v##p, _mm256_set1_epi16(0xFF00)), _mm256_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)
    U(0,1) U( 2, 3) U( 4, 5) U( 6, 7)
    U(8,9) U(10,11) U(12,13) U(14,15)

        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm256_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A( 2, 3) A( 4, 5) A( 6, 7)
            A(8,9) A(10,11) A(12,13) A(14,15)

#define P0(p) input##p = _mm256_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)
            P(0,1) P( 2, 3) P( 4, 5) P( 6, 7)
            P(8,9) P(10,11) P(12,13) P(14,15)
        }
    }

    for (/**/; i + 8 <= n_cycles; i += 8) {
        L(0) L(1) L(2) L(3)
        L(4) L(5) L(6) L(7) 
        U(0,1) U(2, 3) U(4, 5) U(6, 7)
        
        for (int i = 0; i < 8; i++) {
            A(0,1) A( 2, 3) A( 4, 5) A( 6, 7)
            P(0,1) P( 2, 3) P( 4, 5) P( 6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2, 3)
        
        for (int i = 0; i < 8; i++) {
            A(0,1) A( 2, 3)
            P(0,1) P( 2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P

    i *= 16;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }
    
    return 0;
}
#else
int pospopcnt_u16_avx2_lemire(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_lemire2(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_mula(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_mula_unroll4(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_mula_unroll8(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_mula_unroll16(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_mula3(const uint16_t* array, uint32_t len, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_csa(const uint16_t* array, uint32_t len, uint32_t* flags) { return(0); }
#endif

#if SIMD_VERSION >= 3
int pospopcnt_u16_sse_mula(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m128i* data_vectors = (const __m128i*)(array);
    const uint32_t n_cycles = len / 8;

    size_t i = 0;
    for (/**/; i + 2 <= n_cycles; i += 2) {
        __m128i v0 = _mm_loadu_si128(data_vectors + i);// don't assume alignment
        __m128i v1 = _mm_loadu_si128(data_vectors + i + 1);// don't assume alignment

        __m128i input0 = _mm_or_si128(_mm_and_si128(v0, _mm_set1_epi16(0x00FF)), _mm_slli_epi16(v1, 8));
        __m128i input1 = _mm_or_si128(_mm_and_si128(v0, _mm_set1_epi16(0xFF00)), _mm_srli_epi16(v1, 8));
        
        for (int i = 0; i < 8; i++) {
            flags[ 7 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input0));
            flags[15 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input1));
            input0 = _mm_add_epi8(input0, input0);
            input1 = _mm_add_epi8(input1, input1);
        }
    }

    i *= 8;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }
    
    return 0;
}

int pospopcnt_u16_sse_mula_unroll4(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m128i* data_vectors = (const __m128i*)(array);
    const uint32_t n_cycles = len / 8;

    size_t i = 0;
    for (/**/; i + 4 <= n_cycles; i += 4) {
#define L(p) __m128i v##p =  _mm_loadu_si128(data_vectors+i+p);
        L(0) L(1) L(2) L(3)

#define U0(p,k) __m128i input##p = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0x00FF)), _mm_slli_epi16(v##k, 8));
#define U1(p,k) __m128i input##k = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0xFF00)), _mm_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)

        U(0,1) U(2,3)
        
        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3)

#define P0(p) input##p = _mm_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)

            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 8;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P
    
    return 0;
}

int pospopcnt_u16_sse_mula_unroll8(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m128i* data_vectors = (const __m128i*)(array);
    const uint32_t n_cycles = len / 8;

    size_t i = 0;
    for (/**/; i + 8 <= n_cycles; i += 8) {
#define L(p) __m128i v##p = _mm_loadu_si128(data_vectors+i+p);
        L(0) L(1) L(2) L(3)
        L(4) L(5) L(6) L(7)

#define U0(p,k) __m128i input##p = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0x00FF)), _mm_slli_epi16(v##k, 8));
#define U1(p,k) __m128i input##k = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0xFF00)), _mm_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)

        U(0,1) U(2,3) U(4,5) U(6,7)
        
        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3) A(4,5) A(6, 7)

#define P0(p) input##p = _mm_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)

            P(0,1) P(2, 3) P(4,5) P(6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2,3)
        
        for (int i = 0; i < 8; i++) {
            A(0,1) A(2, 3)
            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 8;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P

    return 0;
}

int pospopcnt_u16_sse_mula_unroll16(const uint16_t* array, uint32_t len, uint32_t* flags) {
    const __m128i* data_vectors = (const __m128i*)(array);
    const uint32_t n_cycles = len / 8;

    size_t i = 0;
    for (/**/; i + 16 <= n_cycles; i += 16) {
#define L(p) __m128i v##p = _mm_loadu_si128(data_vectors+i+p);
        L(0)  L(1)  L(2)  L(3)  
        L(4)  L(5)  L(6)  L(7) 
        L(8)  L(9)  L(10) L(11) 
        L(12) L(13) L(14) L(15)

#define U0(p,k) __m128i input##p = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0x00FF)), _mm_slli_epi16(v##k, 8));
#define U1(p,k) __m128i input##k = _mm_or_si128(_mm_and_si128(v##p, _mm_set1_epi16(0xFF00)), _mm_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)

        U(0,1) U( 2, 3) U( 4, 5) U( 6, 7)
        U(8,9) U(10,11) U(12,13) U(14,15)
        
        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u32(_mm_movemask_epi8(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A( 2,  3) A( 4, 5) A( 6,  7)
            A(8,9) A(10, 11) A(12,13) A(14, 15)

#define P0(p) input##p = _mm_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)

            P(0,1) P( 2,  3) P( 4, 5) P( 6,  7)
            P(8,9) P(10, 11) P(12,13) P(14, 15)
        }
    }

    for (/**/; i + 8 <= n_cycles; i += 8) {
        L(0) L(1) L(2) L(3)
        L(4) L(5) L(6) L(7)

        U(0,1) U(2,3) U(4,5) U(6,7)
        
        for (int i = 0; i < 8; i++) {
            A(0,1) A(2, 3) A(4,5) A(6, 7)
            P(0,1) P(2, 3) P(4,5) P(6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2,3)
        
        for (int i = 0; i < 8; i++) {
            A(0,1) A(2, 3)
            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 8;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P
    
    return 0;
}
#else
int pospopcnt_u16_sse_mula(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_sse_mula_unroll4(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_sse_mula_unroll8(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_sse_mula_unroll16(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx2_csa(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
#endif

#if SIMD_VERSION >= 6
int pospopcnt_u16_avx512_mula(const uint16_t* data, uint32_t len, uint32_t* flags) { 
    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = len / 32;

    size_t i = 0;
    for (/**/; i + 2 <= n_cycles; i += 2) {
        __m512i v0 = _mm512_loadu_si512(data_vectors+i);
        __m512i v1 = _mm512_loadu_si512(data_vectors+i+1);

        __m512i input0 = _mm512_or_si512(_mm512_and_si512(v0, _mm512_set1_epi16(0x00FF)), _mm512_slli_epi16(v1, 8));
        __m512i input1 = _mm512_or_si512(_mm512_and_si512(v0, _mm512_set1_epi16(0xFF00)), _mm512_srli_epi16(v1, 8));
        
        for (int i = 0; i < 8; i++) {
            flags[ 7 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input0));
            flags[15 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input1));
            input0 = _mm512_add_epi8(input0, input0);
            input1 = _mm512_add_epi8(input1, input1);
        }
    }

    i *= 32;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    return 0;
}

int pospopcnt_u16_avx512_mula_unroll4(const uint16_t* data, uint32_t len, uint32_t* flags) { 
    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = len / 32;

    size_t i = 0;
    for (/**/; i + 4 <= n_cycles; i += 4) {
#define L(p) __m512i v##p = _mm512_loadu_si512(data_vectors+i+p);
        L(0) L(1) L(2) L(3)

#define U0(p,k) __m512i input##p = _mm512_or_si512(_mm512_and_si512(v##p, _mm512_set1_epi16(0x00FF)), _mm512_slli_epi16(v##k, 8));
#define U1(p,k) __m512i input##k = _mm512_or_si512(_mm512_and_si512(v##p, _mm512_set1_epi16(0xFF00)), _mm512_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)

        U(0,1) U(2,3)
        
        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3)

#define P0(p) input##p = _mm512_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)

            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 32;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P
    
    return 0;
}

int pospopcnt_u16_avx512_mula_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags) { 
    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = len / 32;

    size_t i = 0;
    for (/**/; i + 8 <= n_cycles; i += 8) {
#define L(p) __m512i v##p = _mm512_loadu_si512(data_vectors+i+p);
        L(0)  L(1)  L(2)  L(3)  
        L(4)  L(5)  L(6)  L(7) 

#define U0(p,k) __m512i input##p = _mm512_or_si512(_mm512_and_si512(v##p, _mm512_set1_epi16(0x00FF)), _mm512_slli_epi16(v##k, 8));
#define U1(p,k) __m512i input##k = _mm512_or_si512(_mm512_and_si512(v##p, _mm512_set1_epi16(0xFF00)), _mm512_srli_epi16(v##k, 8));
#define U(p, k)  U0(p,k) U1(p,k)

        U(0,1) U( 2, 3) U( 4, 5) U( 6, 7)
        
        for (int i = 0; i < 8; i++) {
#define A0(p) flags[ 7 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input##p));
#define A1(k) flags[15 - i] += _mm_popcnt_u64(_mm512_movepi8_mask(input##k));
#define A(p, k) A0(p) A1(k)
            A(0,1) A(2, 3) A(4,5) A(6, 7)

#define P0(p) input##p = _mm512_add_epi8(input##p, input##p);
#define P(p, k) input##p = P0(p) P0(k)

            P(0,1) P(2, 3) P(4,5) P(6, 7)
        }
    }

    for (/**/; i + 4 <= n_cycles; i += 4) {
        L(0) L(1) L(2) L(3)
        U(0,1) U(2,3)
        
        for (int i = 0; i < 8; i++) {
            A(0,1) A(2, 3)
            P(0,1) P(2, 3)
        }
    }

    for (/**/; i + 2 <= n_cycles; i += 2) {
        L(0) L(1)
        U(0,1)
        
        for (int i = 0; i < 8; i++) {
            A(0,1)
            P(0,1)
        }
    }

    i *= 32;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef U0
#undef U1
#undef U
#undef A0
#undef A1
#undef A
#undef P0
#undef P
    
    return 0;
}

int pospopcnt_u16_avx512_mula2(const uint16_t* data, uint32_t len, uint32_t* flags) {
    const __m512i* data_vectors = (const __m512i*)(data);
    const uint32_t n_cycles = len / 32;

    __m512i sum[16];
    for (int i=0; i < 16; i++)
        sum[i] = _mm512_setzero_si512();

    size_t i = 0;
    for (/**/; i + 2 <= n_cycles; i += 2) {
        __m512i v0 = _mm512_loadu_si512(data_vectors + i);
        __m512i v1 = _mm512_loadu_si512(data_vectors + i + 1);

        __m512i input0 = _mm512_or_si512(_mm512_and_si512(v0, _mm512_set1_epi16(0x00FF)), _mm512_slli_epi16(v1, 8));
        __m512i input1 = _mm512_or_si512(_mm512_and_si512(v0, _mm512_set1_epi16(0xFF00)), _mm512_srli_epi16(v1, 8));
        
        const __m512i bits_04 = _mm512_and_si512(input0, _mm512_set1_epi16(0x1111));
        const __m512i bits_15 = _mm512_and_si512(_mm512_srli_epi32(input0, 1), _mm512_set1_epi16(0x1111)); // 0001000100010001
        const __m512i bits_26 = _mm512_and_si512(_mm512_srli_epi32(input0, 2), _mm512_set1_epi16(0x1111));
        const __m512i bits_37 = _mm512_and_si512(_mm512_srli_epi32(input0, 3), _mm512_set1_epi16(0x1111));

        const __m512i bits_8c = _mm512_and_si512(input1, _mm512_set1_epi16(0x1111));
        const __m512i bits_9d = _mm512_and_si512(_mm512_srli_epi32(input1, 1), _mm512_set1_epi16(0x1111));
        const __m512i bits_ae = _mm512_and_si512(_mm512_srli_epi32(input1, 2), _mm512_set1_epi16(0x1111));
        const __m512i bits_bf = _mm512_and_si512(_mm512_srli_epi32(input1, 3), _mm512_set1_epi16(0x1111));

        const __m512i sum_04 = _mm512_sad_epu8(bits_04, _mm512_setzero_si512());
        const __m512i sum_15 = _mm512_sad_epu8(bits_15, _mm512_setzero_si512());
        const __m512i sum_26 = _mm512_sad_epu8(bits_26, _mm512_setzero_si512());
        const __m512i sum_37 = _mm512_sad_epu8(bits_37, _mm512_setzero_si512());

        const __m512i sum_8c = _mm512_sad_epu8(bits_8c, _mm512_setzero_si512());
        const __m512i sum_9d = _mm512_sad_epu8(bits_9d, _mm512_setzero_si512());
        const __m512i sum_ae = _mm512_sad_epu8(bits_ae, _mm512_setzero_si512());
        const __m512i sum_bf = _mm512_sad_epu8(bits_bf, _mm512_setzero_si512());

        sum[0x0] = _mm512_add_epi64(sum[0x0], _mm512_and_si512 (sum_04, _mm512_set1_epi64(0xf)));
        sum[0x4] = _mm512_add_epi64(sum[0x4], _mm512_srli_epi32(sum_04, 4));
        sum[0x1] = _mm512_add_epi64(sum[0x1], _mm512_and_si512 (sum_15, _mm512_set1_epi64(0xf)));
        sum[0x5] = _mm512_add_epi64(sum[0x5], _mm512_srli_epi32(sum_15, 4));
        sum[0x2] = _mm512_add_epi64(sum[0x2], _mm512_and_si512 (sum_26, _mm512_set1_epi64(0xf)));
        sum[0x6] = _mm512_add_epi64(sum[0x6], _mm512_srli_epi32(sum_26, 4));
        sum[0x3] = _mm512_add_epi64(sum[0x3], _mm512_and_si512 (sum_37, _mm512_set1_epi64(0xf)));
        sum[0x7] = _mm512_add_epi64(sum[0x7], _mm512_srli_epi32(sum_37, 4));
        sum[0x8] = _mm512_add_epi64(sum[0x8], _mm512_and_si512 (sum_8c, _mm512_set1_epi64(0xf)));
        sum[0xc] = _mm512_add_epi64(sum[0xc], _mm512_srli_epi32(sum_8c, 4));
        sum[0x9] = _mm512_add_epi64(sum[0x9], _mm512_and_si512 (sum_9d, _mm512_set1_epi64(0xf)));
        sum[0xd] = _mm512_add_epi64(sum[0xd], _mm512_srli_epi32(sum_9d, 4));
        sum[0xa] = _mm512_add_epi64(sum[0xa], _mm512_and_si512 (sum_ae, _mm512_set1_epi64(0xf)));
        sum[0xe] = _mm512_add_epi64(sum[0xe], _mm512_srli_epi32(sum_ae, 4));
        sum[0xb] = _mm512_add_epi64(sum[0xb], _mm512_and_si512 (sum_bf, _mm512_set1_epi64(0xf)));
        sum[0xf] = _mm512_add_epi64(sum[0xf], _mm512_srli_epi32(sum_bf, 4));
    }

    for (int j; j < 16; j++) {
        const __m128i dq0 = _mm512_extracti32x4_epi32(sum[j], 0);
        const __m128i dq1 = _mm512_extracti32x4_epi32(sum[j], 1);
        const __m128i dq2 = _mm512_extracti32x4_epi32(sum[j], 2);
        const __m128i dq3 = _mm512_extracti32x4_epi32(sum[j], 3);
        flags[j] = _mm_extract_epi64(dq0, 0)
                    + _mm_extract_epi64(dq0, 1)
                    + _mm_extract_epi64(dq1, 0)
                    + _mm_extract_epi64(dq1, 1)
                    + _mm_extract_epi64(dq2, 0)
                    + _mm_extract_epi64(dq2, 1)
                    + _mm_extract_epi64(dq3, 0)
                    + _mm_extract_epi64(dq3, 1);
    }

    i *= 32;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((data[i] & (1 << j)) >> j);
        }
    }

    return 0;
}

int pospopcnt_u16_avx512_mula3(const uint16_t* array, uint32_t len, uint32_t* flags) {
    __m512i counters[16];

    for (size_t i = 0; i < 16; i++) {
        counters[i] = _mm512_setzero_si512();
    }

    const __m512i mask1bit = _mm512_set1_epi16(0x5555); // 0101010101010101
    const __m512i mask2bit = _mm512_set1_epi16(0x3333); // 0011001100110011
    const __m512i mask4bit = _mm512_set1_epi16(0x0f0f); // 1111000011110000
    const __m512i mask8bit = _mm512_set1_epi16(0x00ff); // 0000000011111111
    
    const uint32_t n_cycles = len / (2048 * (16*32));
    const uint32_t n_total  = len / (16*32);
    uint16_t tmp[32];

/*------ Macros --------*/
#define LL(i,p,k)  const __m512i sum##p##k##_##i##bit_even = _mm512_add_epi8(input##p & mask##i##bit, input##k & mask##i##bit);
#define LO(i,p,k)  const __m512i sum##p##k##_##i##bit_odd  = _mm512_add_epi8(_mm512_srli_epi16(input##p, i) & mask##i##bit, _mm512_srli_epi16(input##k, i) & mask##i##bit);

#define LBLOCK(i)                                 \
    LL(i,0,1) LL(i,2,3)   LL(i,4,5)   LL(i,6,7)   \
    LL(i,8,9) LL(i,10,11) LL(i,12,13) LL(i,14,15) \
    LO(i,0,1) LO(i,2,3)   LO(i,4,5)   LO(i,6,7)   \
    LO(i,8,9) LO(i,10,11) LO(i,12,13) LO(i,14,15)

#define EVEN(b,i,k,p) input##i = sum##k##p##_##b##bit_even;
#define ODD(b,i,k,p)  input##i = sum##k##p##_##b##bit_odd;

#define UPDATE(i)                                                  \
    EVEN(i,0,0,1) EVEN(i,1,2,3)   EVEN(i,2,4,5)   EVEN(i,3,6,7)    \
    EVEN(i,4,8,9) EVEN(i,5,10,11) EVEN(i,6,12,13) EVEN(i,7,14,15)  \
     ODD(i,8,0,1)  ODD(i,9,2,3)    ODD(i,10,4,5)   ODD(i,11,6,7)   \
     ODD(i,12,8,9) ODD(i,13,10,11) ODD(i,14,12,13) ODD(i,15,14,15) \

#define UE(i,p,k) counters[i] = _mm512_add_epi16(counters[i], sum##p##k##_8bit_even);
#define UO(i,p,k) counters[i] = _mm512_add_epi16(counters[i], sum##p##k##_8bit_odd);

/*------ Start --------*/
#define L(p) __m512i input##p = _mm512_loadu_si512((__m512i*)(array + i*2048*512 + j*512 + p*32));
    size_t i = 0;
    for (/**/; i < n_cycles; ++i) {
        for (int j = 0; j < 2048; ++j) {
            // Load 16 registers.
            L(0)  L(1)  L(2)  L(3)  
            L(4)  L(5)  L(6)  L(7) 
            L(8)  L(9)  L(10) L(11) 
            L(12) L(13) L(14) L(15)

            // Perform updates for bits {1,2,4,8}.
            LBLOCK(1) UPDATE(1)
            LBLOCK(2) UPDATE(2)
            LBLOCK(4) UPDATE(4)
            LBLOCK(8) UPDATE(8)

            // Update accumulators.
            UE( 0,0,1) UE( 1, 2, 3) UE( 2, 4, 5) UE( 3, 6, 7)  
            UE( 4,8,9) UE( 5,10,11) UE( 6,12,13) UE( 7,14,15) 
            UO( 8,0,1) UO( 9, 2, 3) UO(10, 4, 5) UO(11, 6, 7) 
            UO(12,8,9) UO(13,10,11) UO(14,12,13) UO(15,14,15)
        }

        // Update.
        for (size_t i = 0; i < 16; i++) {
            _mm512_storeu_si512((__m512i*)tmp, counters[i]);
            for (int j = 0; j < 32; j++) flags[i] += tmp[j];
        }
        // Reset.
        for (size_t i = 0; i < 16; i++) {
            counters[i] = _mm512_setzero_si512();
        }
    }
#undef L
#define L(p) __m512i input##p = _mm512_loadu_si512((__m512i*)(array + i*512 + p*32));
    i *= 2048;
    for (/**/; i < n_total; ++i) {
        // Load 16 registers.
        L(0)  L(1)  L(2)  L(3)  
        L(4)  L(5)  L(6)  L(7) 
        L(8)  L(9)  L(10) L(11) 
        L(12) L(13) L(14) L(15)

        // Perform updates for bits {1,2,4,8}.
        LBLOCK(1) UPDATE(1)
        LBLOCK(2) UPDATE(2)
        LBLOCK(4) UPDATE(4)
        LBLOCK(8) UPDATE(8)

        // Update accumulators.
        UE( 0,0,1) UE( 1, 2, 3) UE( 2, 4, 5) UE( 3, 6, 7)  
        UE( 4,8,9) UE( 5,10,11) UE( 6,12,13) UE( 7,14,15) 
        UO( 8,0,1) UO( 9, 2, 3) UO(10, 4, 5) UO(11, 6, 7) 
        UO(12,8,9) UO(13,10,11) UO(14,12,13) UO(15,14,15)
    }

    i *= 512;
    for (/**/; i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

#undef L
#undef UPDATE
#undef ODD
#undef EVEN
#undef LBLOCK
#undef LL
#undef LO
#undef UO
#undef UE

    for (size_t i = 0; i < 16; i++) {
        _mm512_storeu_si512((__m512i*)tmp, counters[i]);
        for (int j = 0; j < 32; j++) flags[i] += tmp[j];
    }
    return 0;
}

int pospopcnt_u16_avx512_csa(const uint16_t* array, uint32_t len, uint32_t* flags) {
    //for (size_t i = 0; i < 16; i++) flags[i] = 0;
    for (uint32_t i = len - (len % (32 * 16)); i < len; ++i) {
        for (int j = 0; j < 16; ++j) {
            flags[j] += ((array[i] & (1 << j)) >> j);
        }
    }

    const __m512i* data = (const __m512i*)array;
    __m512i v1  = _mm512_setzero_si512();
    __m512i v2  = _mm512_setzero_si512();
    __m512i v4  = _mm512_setzero_si512();
    __m512i v8  = _mm512_setzero_si512();
    __m512i v16 = _mm512_setzero_si512();
    __m512i twosA, twosB, foursA, foursB, eightsA, eightsB;

    const size_t size = len / 32;
    const uint64_t limit = size - size % 16;

    uint16_t buffer[32];

    uint64_t i = 0;
    while (i < limit) {
        __m512i counter[16];
        for (size_t i = 0; i < 16; i++) {
            counter[i] = _mm512_setzero_si512();
        }

        size_t thislimit = limit;
        if (thislimit - i >= (1 << 16))
            thislimit = i + (1 << 16) - 1;

        for (/**/; i < thislimit; i += 16) {
            POSPOPCNT_CSA_AVX512(&twosA,    &v1, _mm512_loadu_si512(data + i + 0), _mm512_loadu_si512(data + i + 1));
            POSPOPCNT_CSA_AVX512(&twosB,    &v1, _mm512_loadu_si512(data + i + 2), _mm512_loadu_si512(data + i + 3));
            POSPOPCNT_CSA_AVX512(&foursA,   &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX512(&twosA,    &v1, _mm512_loadu_si512(data + i + 4), _mm512_loadu_si512(data + i + 5));
            POSPOPCNT_CSA_AVX512(&twosB,    &v1, _mm512_loadu_si512(data + i + 6), _mm512_loadu_si512(data + i + 7));
            POSPOPCNT_CSA_AVX512(&foursB,   &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX512(&eightsA,  &v4, foursA, foursB);
            POSPOPCNT_CSA_AVX512(&twosA,    &v1, _mm512_loadu_si512(data + i + 8),  _mm512_loadu_si512(data + i + 9));
            POSPOPCNT_CSA_AVX512(&twosB,    &v1, _mm512_loadu_si512(data + i + 10), _mm512_loadu_si512(data + i + 11));
            POSPOPCNT_CSA_AVX512(&foursA,   &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX512(&twosA,    &v1, _mm512_loadu_si512(data + i + 12), _mm512_loadu_si512(data + i + 13));
            POSPOPCNT_CSA_AVX512(&twosB,    &v1, _mm512_loadu_si512(data + i + 14), _mm512_loadu_si512(data + i + 15));
            POSPOPCNT_CSA_AVX512(&foursB,   &v2, twosA, twosB);
            POSPOPCNT_CSA_AVX512(&eightsB,  &v4, foursA, foursB);
            POSPOPCNT_CSA_AVX512(&v16,      &v8, eightsA, eightsB);

            for (size_t i = 0; i < 16; i++) {
                counter[i] = _mm512_add_epi16(counter[i], _mm512_and_si512(v16, _mm512_set1_epi16(1)));
                v16 = _mm512_srli_epi16(v16, 1);
            }
        }
        
        for (size_t i = 0; i < 16; i++) {
            _mm512_storeu_si512((__m512i*)buffer, counter[i]);
            for (size_t z = 0; z < 32; z++) {
                flags[i] += buffer[z] * 16;
            }
        }
    }

    _mm512_storeu_si512((__m512i*)buffer, v1);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 1 * ((buffer[i] & (1 << j)) >> j);
        }
    }

    _mm512_storeu_si512((__m512i*)buffer, v2);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 2 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    
    _mm512_storeu_si512((__m512i*)buffer, v4);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 4 * ((buffer[i] & (1 << j)) >> j);
        }
    }

    _mm512_storeu_si512((__m512i*)buffer, v8);
    for (size_t i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            flags[j] += 8 * ((buffer[i] & (1 << j)) >> j);
        }
    }
    return 0;
}
#else
int pospopcnt_u16_avx512_mula(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512_mula_unroll4(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512_mula_unroll8(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512_mula3(const uint16_t* data, uint32_t n, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512_csa(const uint16_t* array, uint32_t len, uint32_t* flags) { return(0); }
int pospopcnt_u16_avx512_mula2(const uint16_t* array, uint32_t len, uint32_t* flags) { return(0); }
#endif
