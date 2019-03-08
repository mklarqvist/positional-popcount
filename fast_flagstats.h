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
#ifndef FAST_FLAGSTATS_H_
#define FAST_FLAGSTATS_H_

#include <iostream>
#include <random>
#include <chrono>
#include <cstring> //memset
#include <cassert> //assert

//#define __AVX2__ 1 // temp

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
#define SIMD_WIDTH      512
#define SIMD_ALIGNMENT  64
#elif defined(__AVX2__) && __AVX2__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    5
#define SIMD_WIDTH      256
#define SIMD_ALIGNMENT  32
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
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    2
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      128
#elif defined(__SSE__) && __SSE__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    1
#define SIMD_ALIGNMENT  16
#define SIMD_WIDTH      0
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
static inline void PIL_POPCOUNT_SSE(uint64_t& a, const __m128i n) {
    a += PIL_POPCOUNT(_mm_cvtsi128_si64(n)) + PIL_POPCOUNT(_mm_cvtsi128_si64(_mm_unpackhi_epi64(n, n)));
}
#endif

template <uint32_t(f)(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags)>
uint32_t flag_stats_wrapper(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags) {
    return((*f)(data, n, flags));
}

uint32_t flag_stats_scalar_naive(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
uint32_t flag_stats_scalar_partition(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);

uint32_t flag_stats_hist1x4(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);

#if SIMD_VERSION >= 5

#ifndef PIL_POPCOUNT_AVX2
#define PIL_POPCOUNT_AVX2(A, B) {                  \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

uint32_t flag_stats_avx2_popcnt(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
uint32_t flag_stats_avx2(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
uint32_t flag_stats_avx2_naive_counter(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
uint32_t flag_stats_avx2_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
#else
uint32_t flag_stats_avx2_popcnt(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
uint32_t flag_stats_avx2(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
uint32_t flag_stats_avx2_naive_counter(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
uint32_t flag_stats_avx2_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
#endif

#if SIMD_VERSION >= 2
uint32_t flag_stats_sse_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
#else
uint32_t flag_stats_sse_single(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
#endif

#if SIMD_VERSION >= 6
uint32_t flag_stats_avx512(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
#else
uint32_t flag_stats_avx512(const uint16_t* __restrict__ data, uint32_t n, uint32_t* __restrict__ flags);
#endif

#endif /* FAST_FLAGSTATS_H_ */
