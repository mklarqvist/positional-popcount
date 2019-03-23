/*
* Copyright (c) 2019
* Author(s): Marcus D. R. Klarqvist and Daniel Lemire
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
#ifndef FAST_FLAGSTATS_H_
#define FAST_FLAGSTATS_H_

#include <stdint.h> //types
#include <math.h> //floor
#include <string.h> //memset
#include <assert.h> //assert

/****************************
*  SIMD definitions
****************************/
#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
     #include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
     #include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
     #include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
     #include <spe.h>
#endif

#if defined(__AVX512F__) && __AVX512F__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    6
#define SIMD_ALIGNMENT  64
#define SIMD_WIDTH      512
#elif defined(__AVX2__) && __AVX2__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    5
#define SIMD_ALIGNMENT  32
#define SIMD_WIDTH      256
#elif defined(__AVX__) && __AVX__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    4
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#elif defined(__SSE4_1__) && __SSE4_1__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    3
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#elif defined(__SSE2__) && __SSE2__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#elif defined(__SSE__) && __SSE__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#else
#define SIMD_AVAILABLE  0
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      0
#endif

#ifdef _mm_popcnt_u64
#define PIL_POPCOUNT _mm_popcnt_u64
#else
#define PIL_POPCOUNT __builtin_popcountll
#endif

#if SIMD_AVAILABLE
__attribute__((always_inline))
static inline void PIL_POPCOUNT_SSE(uint64_t* a, const __m128i n) {
    *a += PIL_POPCOUNT(_mm_cvtsi128_si64(n)) + PIL_POPCOUNT(_mm_cvtsi128_si64(_mm_unpackhi_epi64(n, n)));
}
#endif // endif simd_available

#if SIMD_VERSION >= 5
#ifndef PIL_POPCOUNT_AVX2
#define PIL_POPCOUNT_AVX2(A, B) {                  \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif
#endif // endif simd_version >= 5

#if SIMD_VERSION >= 6
// By Wojciech Mula
// @see https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx512-harley-seal.cpp#L3
// @see https://arxiv.org/abs/1611.07612
__attribute__((always_inline))
static inline __m512i avx512_popcount(const __m512i v) {
    const __m512i m1 = _mm512_set1_epi8(0x55);
    const __m512i m2 = _mm512_set1_epi8(0x33);
    const __m512i m4 = _mm512_set1_epi8(0x0F);

    const __m512i t1 = _mm512_sub_epi8(v,       (_mm512_srli_epi16(v,  1)  & m1));
    const __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2)  & m2));
    const __m512i t3 = _mm512_add_epi8(t2,       _mm512_srli_epi16(t2, 4)) & m4;
    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}
#endif // endif simd_version >= 6

/*------ Core enums --------*/
typedef enum {
    PPOPCNT_AUTO,
    PPOPCNT_SCALAR,
    PPOPCNT_SCALAR_PARTITION,
    PPOPCNT_SCALAR_HIST1X4,
    PPOPCNT_AVX2_POPCNT,
    PPOPCNT_AVX2,
    PPOPCNT_AVX2_POPCNT_NAIVE,
    PPOPCNT_AVX2_SINGLE,
    PPOPCNT_SSE_SINGLE,
    PPOPCNT_AVX512,
    PPOPCNT_AVX512_MASK32,
    PPOPCNT_AVX512_MASK64,
    PPOPCNT_AVX512_POPCNT,
    PPOPCNT_AVX2_LEMIRE1,
    PPOPCNT_AVX2_LEMIRE2,
    PPOPCNT_AVX2_MULA,
    PPOPCNT_AVX2_MULA_UR4,
    PPOPCNT_AVX2_MULA_UR8,
    PPOPCNT_AVX2_MULA_UR16,
    PPOPCNT_SSE_MULA,
    PPOPCNT_SSE_MULA_UR4,
    PPOPCNT_SSE_MULA_UR8,
    PPOPCNT_SSE_MULA_UR16
} PPOPCNT_U16_METHODS;

int pospopcnt_u16_scalar_naive(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_scalar_partition(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_hist1x4(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_popcnt(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_naive_counter(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_single(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_sse_single(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx512(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx512_popcnt32_mask(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx512_popcnt64_mask(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx512_popcnt(const uint16_t* data, uint32_t n, uint32_t* flags);

int pospopcnt_u16_avx2_lemire(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_lemire2(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_mula(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_mula_unroll4(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_mula_unroll8(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_avx2_mula_unroll16(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_sse_mula(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_sse_mula_unroll4(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_sse_mula_unroll8(const uint16_t* data, uint32_t n, uint32_t* flags);

// Wrapper function for calling the best available algorithm during compilation
// time.
int pospopcnt_u16(const uint16_t* data, uint32_t n, uint32_t* flags);
int pospopcnt_u16_method(PPOPCNT_U16_METHODS method, const uint16_t* data, uint32_t n, uint32_t* flags);

#endif /* FAST_FLAGSTATS_H_ */
