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
/*
 * Notice taken from the positional-popcount website
 * (https://github.com/mklarqvist/positional-popcount):
 *
 * These functions compute the novel "positional population count" (`pospopcnt`)
 * statistics using fast SIMD instructions. Given a stream of k-bit words, we
 * seek to count the number of set bits in positions 0, 1, 2, ..., k-1. This
 * problem is a generalization of the population-count problem where we count
 * the sum total of set bits in a k-bit word.
 *
 * These functions can be applied to any packed 1-hot 16-bit primitive, for
 * example in machine learning/deep learning. Using large registers (AVX-512),
 * we can achieve ~50 GB/s (~0.120 CPU cycles / int) throughput (25 billion
 * 16-bit integers / second or 200 billion one-hot vectors / second).
 *
 * This benchmark shows the speedup of the 3 `pospopcnt` algorithms used on x86
 * CPUs compared to the efficient auto-vectorization of
 * `pospopcnt_u16_scalar_naive` for different array sizes (in number of 2-byte
 * values).
 *
 * | Algorithm                   | 128  | 256  | 512  | 1024 | 2048 | 4096 | 8192 | 65536 |
 * |-----------------------------|------|------|------|------|------|------|------|-------|
 * | sse_blend_popcnt_unroll8    | 2.09 | 3.16 | 2.35 | 1.88 | 1.67 | 1.56 | 1.5  | 1.44  |
 * | avx512_blend_popcnt_unroll8 | 1.78 | 3.61 | 3.61 | 3.59 | 3.68 | 3.65 | 3.67 | 3.7   |
 * | avx512_adder_forest         | 0.77 | 0.9  | 3.24 | 3.96 | 4.96 | 5.87 | 6.52 | 7.24  |
 * | avx512_harley_seal          | 0.52 | 0.74 | 1.83 | 2.64 | 4.06 | 6.43 | 9.41 | 16.28 |
 *
 * Compared to a naive unvectorized solution (`pospopcnt_u16_scalar_naive_nosimd`):
 *
 * | Algorithm                   | 128  | 256   | 512   | 1024  | 2048  | 4096  | 8192  | 65536  |
 * |-----------------------------|------|-------|-------|-------|-------|-------|-------|--------|
 * | sse_mula_unroll8            | 8.28 | 9.84  | 10.55 | 11    | 11.58 | 11.93 | 12.13 | 12.28  |
 * | avx512_blend_popcnt_unroll8 | 7.07 | 11.25 | 16.21 | 21    | 25.49 | 27.91 | 29.73 | 31.55  |
 * | avx512_adder_forest         | 3.05 | 2.82  | 14.53 | 23.13 | 34.37 | 44.91 | 52.78 | 61.68  |
 * | avx512_harley_seal          | 2.07 | 2.3   | 8.21  | 15.41 | 28.17 | 49.14 | 76.11 | 138.71 |
 *
*/
#ifndef POSPOPCNT_H_2359235897293
#define POSPOPCNT_H_2359235897293

#ifdef __cplusplus
extern "C" {
#endif

/* ****************************
*  Definitions
******************************/
#include <stdint.h> //types

/* ****************************
*  SIMD definitions
******************************/
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
#define POSPOPCNT_SIMD_VERSION    6
#define POSPOPCNT_SIMD_ALIGNMENT  64
#elif defined(__AVX2__) && __AVX2__ == 1
#define POSPOPCNT_SIMD_VERSION    5
#define POSPOPCNT_SIMD_ALIGNMENT  32
#elif defined(__AVX__) && __AVX__ == 1
#define POSPOPCNT_SIMD_VERSION    4
#define POSPOPCNT_SIMD_ALIGNMENT  16
#elif defined(__SSE4_1__) && __SSE4_1__ == 1
#define POSPOPCNT_SIMD_VERSION    3
#define POSPOPCNT_SIMD_ALIGNMENT  16
#elif defined(__SSE2__) && __SSE2__ == 1
#define POSPOPCNT_SIMD_VERSION    0
#define POSPOPCNT_SIMD_ALIGNMENT  16
#elif defined(__SSE__) && __SSE__ == 1
#define POSPOPCNT_SIMD_VERSION    0
#define POSPOPCNT_SIMD_ALIGNMENT  16
#else
#define POSPOPCNT_SIMD_VERSION    0
#define POSPOPCNT_SIMD_ALIGNMENT  16
#endif

/* ****************************
 *  API modifier
 ******************************/
# if defined(__GNUC__)
#    define PPOPCNT_INLINE static __inline __attribute__((unused))
#  elif defined (__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
#    define PPOPCNT_INLINE static inline
#  elif defined(_MSC_VER)
#    define PPOPCNT_INLINE static __inline
#  else
     /* this version may generate warnings for unused static functions */
#    define PPOPCNT_INLINE static
# endif

/* ****************************
 *  Support functions
 ******************************/
#ifdef _mm_popcnt_u64
#define PIL_POPCOUNT _mm_popcnt_u64
#else
#define PIL_POPCOUNT __builtin_popcountll
#endif

// Not supported on MSVC
#ifndef _MSC_VER
PPOPCNT_INLINE
uint64_t pospopcnt_umul128(uint64_t a, uint64_t b, uint64_t* hi) {
    unsigned __int128 x = (unsigned __int128)a * (unsigned __int128)b;
    *hi = (uint64_t)(x >> 64);
    return (uint64_t)x;
}

PPOPCNT_INLINE
uint64_t pospopcnt_loadu_u64(const void* ptr) {
    uint64_t data;
    memcpy(&data, ptr, sizeof(data));
    return data;
}
#endif

#if POSPOPCNT_SIMD_VERSION >= 3

/**
 * Carry-save adder update step.
 * @see https://en.wikipedia.org/wiki/Carry-save_adder#Technical_details
 *
 * Steps:
 * 1)  U = *L ⊕ B
 * 2) *H = (*L ^ B) | (U ^ C)
 * 3) *L = *L ⊕ B ⊕ C = U ⊕ C
 *
 * B and C are 16-bit staggered registers such that &C - &B = 1.
 *
 * Example usage:
 * pospopcnt_csa_sse(&twosA, &v1, _mm_loadu_si128(data + i + 0), _mm_loadu_si128(data + i + 1));
 *
 * @param h
 * @param l
 * @param b
 * @param c
 */
PPOPCNT_INLINE
void pospopcnt_csa_sse(__m128i* __restrict__ h,
                       __m128i* __restrict__ l,
                       const __m128i b, const __m128i c)
{
     const __m128i u = _mm_xor_si128(*l, b);
     *h = _mm_or_si128(*l & b, u & c); // shift carry (sc_i).
     *l = _mm_xor_si128(u, c); // partial sum (ps).
}
#endif

#if POSPOPCNT_SIMD_VERSION >= 5
#ifndef PIL_POPCOUNT_AVX2
#define PIL_POPCOUNT_AVX2(A, B) {                  \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

PPOPCNT_INLINE
void pospopcnt_csa_avx2(__m256i* __restrict__ h,
                        __m256i* __restrict__ l,
                        const __m256i b, const __m256i c)
{
     const __m256i u = _mm256_xor_si256(*l, b);
     *h = _mm256_or_si256(*l & b, u & c);
     *l = _mm256_xor_si256(u, c);
}
#endif // endif simd_version >= 5

#if POSPOPCNT_SIMD_VERSION >= 6
// By Wojciech Muła
// @see https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx512-harley-seal.cpp#L3
// @see https://arxiv.org/abs/1611.07612
__attribute__((always_inline))
static inline __m512i avx512_popcount(const __m512i v) {
    const __m512i m1 = _mm512_set1_epi8(0x55); // 01010101
    const __m512i m2 = _mm512_set1_epi8(0x33); // 00110011
    const __m512i m4 = _mm512_set1_epi8(0x0F); // 00001111

    const __m512i t1 = _mm512_sub_epi8(v,       (_mm512_srli_epi16(v,  1)  & m1));
    const __m512i t2 = _mm512_add_epi8(t1 & m2, (_mm512_srli_epi16(t1, 2)  & m2));
    const __m512i t3 = _mm512_add_epi8(t2,       _mm512_srli_epi16(t2, 4)) & m4;
    return _mm512_sad_epu8(t3, _mm512_setzero_si512());
}

// 512i-version of carry-save adder subroutine.
PPOPCNT_INLINE
void pospopcnt_csa_avx512(__m512i* __restrict__ h,
                          __m512i* __restrict__ l,
                          __m512i b, __m512i c)
{
     *h = _mm512_ternarylogic_epi32(c, b, *l, 0xE8); // 11101000
     *l = _mm512_ternarylogic_epi32(c, b, *l, 0x96); // 10010110
}
#endif // endif simd_version >= 6

/* ****************************
*  Support definitions
******************************/
typedef enum {
    PPOPCNT_AUTO,
    PPOPCNT_SCALAR,
    PPOPCNT_SCALAR_NOSIMD,
    PPOPCNT_SCALAR_PARTITION,
    PPOPCNT_SCALAR_HIST1X4,
    PPOPCNT_SCALAR_UMUL128,
    PPOPCNT_SCALAR_UMUL128_UR2,
    PPOPCNT_SSE_SINGLE,
    PPOPCNT_SSE_BLEND_POPCNT,
    PPOPCNT_SSE_BLEND_POPCNT_UR4,
    PPOPCNT_SSE_BLEND_POPCNT_UR8,
    PPOPCNT_SSE_BLEND_POPCNT_UR16,
    PPOPCNT_SSE_SAD,
    PPOPCNT_SSE_HARLEY_SEAL,
    PPOPCNT_AVX2_POPCNT,
    PPOPCNT_AVX2,
    PPOPCNT_AVX2_POPCNT_NAIVE,
    PPOPCNT_AVX2_SINGLE,
    PPOPCNT_AVX2_LEMIRE1,
    PPOPCNT_AVX2_LEMIRE2,
    PPOPCNT_AVX2_BLEND_POPCNT,
    PPOPCNT_AVX2_BLEND_POPCNT_UR4,
    PPOPCNT_AVX2_BLEND_POPCNT_UR8,
    PPOPCNT_AVX2_BLEND_POPCNT_UR16,
    PPOPCNT_AVX2_ADDER_FOREST,
    PPOPCNT_AVX2_HARLEY_SEAL,
    PPOPCNT_AVX512,
    PPOPCNT_AVX512BW_MASK32,
    PPOPCNT_AVX512BW_MASK64,
    PPOPCNT_AVX512_MASKED_OPS,
    PPOPCNT_AVX512_POPCNT,
    PPOPCNT_AVX512BW_BLEND_POPCNT,
    PPOPCNT_AVX512BW_BLEND_POPCNT_UR4,
    PPOPCNT_AVX512BW_BLEND_POPCNT_UR8,
    PPOPCNT_AVX512_MULA2,
    PPOPCNT_AVX512BW_ADDER_FOREST,
    PPOPCNT_AVX512BW_HARLEY_SEAL,
    PPOPCNT_AVX512VBMI_HARLEY_SEAL,
    //
    PPOPCNT_NUMBER_METHODS
} PPOPCNT_U16_METHODS;

static const char * const pospopcnt_u16_method_names[] = {
    "pospopcnt_u16",
    "pospopcnt_u16_scalar_naive",
    "pospopcnt_u16_scalar_naive_nosimd",
    "pospopcnt_u16_scalar_partition",
    "pospopcnt_u16_scalar_hist1x4",
    "pospopcnt_u16_scalar_umul128",
    "pospopcnt_u16_scalar_umul128_unroll2",
    "pospopcnt_u16_sse_single",
    "pospopcnt_u16_sse_blend_popcnt",
    "pospopcnt_u16_sse_blend_popcnt_unroll4",
    "pospopcnt_u16_sse_blend_popcnt_unroll8",
    "pospopcnt_u16_sse_blend_popcnt_unroll16",
    "pospopcnt_u16_sse_sad",
    "pospopcnt_u16_sse_harley_seal",
    "pospopcnt_u16_avx2_popcnt",
    "pospopcnt_u16_avx2",
    "pospopcnt_u16_avx2_naive_counter",
    "pospopcnt_u16_avx2_single",
    "pospopcnt_u16_avx2_lemire",
    "pospopcnt_u16_avx2_lemire2",
    "pospopcnt_u16_avx2_blend_popcnt",
    "pospopcnt_u16_avx2_blend_popcnt_unroll4",
    "pospopcnt_u16_avx2_blend_popcnt_unroll8",
    "pospopcnt_u16_avx2_blend_popcnt_unroll16",
    "pospopcnt_u16_avx2_adder_forest",
    "pospopcnt_u16_avx2_harley_seal",
    "pospopcnt_u16_avx512",
    "pospopcnt_u16_avx512bw_popcnt32_mask",
    "pospopcnt_u16_avx512bw_popcnt64_mask",
    "pospopcnt_u16_avx512_masked_ops",
    "pospopcnt_u16_avx512_popcnt",
    "pospopcnt_u16_avx512bw_blend_popcnt",
    "pospopcnt_u16_avx512bw_blend_popcnt_unroll4",
    "pospopcnt_u16_avx512bw_blend_popcnt_unroll8",
    "pospopcnt_u16_avx512_mula2",
    "pospopcnt_u16_avx512bw_adder_forest",
    "pospopcnt_u16_avx512bw_harley_seal",
    "pospopcnt_u16_avx512vbmi_harley_seal"};

typedef enum {
    PPOPCNT_U8_AUTO,
    PPOPCNT_U8_SCALAR,
    PPOPCNT_U8_SCALAR_NOSIMD,
    PPOPCNT_U8_SCALAR_PARTITION,
    PPOPCNT_U8_SCALAR_HIST1X4,
    PPOPCNT_U8_SCALAR_UMUL128,
    PPOPCNT_U8_SCALAR_UMUL128_UR2,
    PPOPCNT_U8_SSE_SINGLE,
    PPOPCNT_U8_SSE_BLEND_POPCNT,
    PPOPCNT_U8_SSE_BLEND_POPCNT_UR4,
    PPOPCNT_U8_SSE_BLEND_POPCNT_UR8,
    PPOPCNT_U8_SSE_BLEND_POPCNT_UR16,
    PPOPCNT_U8_SSE_SAD,
    PPOPCNT_U8_SSE_HARLEY_SEAL,
    PPOPCNT_U8_SSE_POPCNT4BIT,
    PPOPCNT_U8_AVX2_POPCNT,
    PPOPCNT_U8_AVX2,
    PPOPCNT_U8_AVX2_POPCNT_NAIVE,
    PPOPCNT_U8_AVX2_SINGLE,
    PPOPCNT_U8_AVX2_LEMIRE1,
    PPOPCNT_U8_AVX2_LEMIRE2,
    PPOPCNT_U8_AVX2_BLEND_POPCNT,
    PPOPCNT_U8_AVX2_BLEND_POPCNT_UR4,
    PPOPCNT_U8_AVX2_BLEND_POPCNT_UR8,
    PPOPCNT_U8_AVX2_BLEND_POPCNT_UR16,
    PPOPCNT_U8_AVX2_ADDER_FOREST,
    PPOPCNT_U8_AVX2_HARLEY_SEAL,
    PPOPCNT_U8_AVX2_POPCNT4BIT,
    PPOPCNT_U8_AVX512,
    PPOPCNT_U8_AVX512BW_MASK32,
    PPOPCNT_U8_AVX512BW_MASK64,
    PPOPCNT_U8_AVX512_MASKED_OPS,
    PPOPCNT_U8_AVX512_POPCNT,
    PPOPCNT_U8_AVX512BW_BLEND_POPCNT,
    PPOPCNT_U8_AVX512BW_BLEND_POPCNT_UR4,
    PPOPCNT_U8_AVX512BW_BLEND_POPCNT_UR8,
    PPOPCNT_U8_AVX512_MULA2,
    PPOPCNT_U8_AVX512BW_ADDER_FOREST,
    PPOPCNT_U8_AVX512BW_HARLEY_SEAL,
    PPOPCNT_U8_AVX512BW_POPCNT4BIT,
    PPOPCNT_U8_AVX512VBMI_HARLEY_SEAL,
    //
    PPOPCNT_U8_NUMBER_METHODS
} PPOPCNT_U8_METHODS;

static const char * const pospopcnt_u8_method_names[] = {
    "pospopcnt_u8",
    "pospopcnt_u8_scalar_naive",
    "pospopcnt_u8_scalar_naive_nosimd",
    "pospopcnt_u8_scalar_partition",
    "pospopcnt_u8_scalar_hist1x4",
    "pospopcnt_u8_scalar_umul128",
    "pospopcnt_u8_scalar_umul128_unroll2",
    "pospopcnt_u8_sse_single",
    "pospopcnt_u8_sse_blend_popcnt",
    "pospopcnt_u8_sse_blend_popcnt_unroll4",
    "pospopcnt_u8_sse_blend_popcnt_unroll8",
    "pospopcnt_u8_sse_blend_popcnt_unroll8",
    "pospopcnt_u8_sse2_sad",
    "pospopcnt_u8_sse2_harley_seal",
    "pospopcnt_u8_sse_popcnt4bit",
    "pospopcnt_u8_avx2_popcnt",
    "pospopcnt_u8_avx2",
    "pospopcnt_u8_avx2_naive_counter",
    "pospopcnt_u8_avx2_single",
    "pospopcnt_u8_avx2_lemire",
    "pospopcnt_u8_avx2_lemire2",
    "pospopcnt_u8_avx2_blend_popcnt",
    "pospopcnt_u8_avx2_blend_popcnt_unroll4",
    "pospopcnt_u8_avx2_blend_popcnt_unroll8",
    "pospopcnt_u8_avx2_blend_popcnt_unroll8",
    "pospopcnt_u8_avx2_adder_forest",
    "pospopcnt_u8_avx2_harley_seal",
    "pospopcnt_u8_avx2_popcnt4bit",
    "pospopcnt_u8_avx512",
    "pospopcnt_u8_avx512bw_popcnt32_mask",
    "pospopcnt_u8_avx512bw_popcnt64_mask",
    "pospopcnt_u8_avx512_masked_ops",
    "pospopcnt_u8_avx512_popcnt",
    "pospopcnt_u8_avx512bw_blend_popcnt",
    "pospopcnt_u8_avx512bw_blend_popcnt_unroll4",
    "pospopcnt_u8_avx512bw_blend_popcnt_unroll8",
    "pospopcnt_u8_avx512_mula2",
    "pospopcnt_u8_avx512bw_adder_forest",
    "pospopcnt_u8_avx512bw_harley_seal",
    "pospopcnt_u8_avx512bw_popcnt4bit",
    "pospopcnt_u8_avx512vbmi_harley_seal"};

typedef enum {
    PPOPCNT_U32_AUTO,
    PPOPCNT_U32_SCALAR,
    PPOPCNT_U32_SSE_HARLEY_SEAL,
    PPOPCNT_U32_AVX2_HARLEY_SEAL,
    //
    PPOPCNT_U32_NUMBER_METHODS
} PPOPCNT_U32_METHODS;

static const char * const pospopcnt_u32_method_names[] = {
    "pospopcnt_u32",
    "pospopcnt_u32_scalar_naive",
    "pospopcnt_u32_sse_harley_seal",
    "pospopcnt_u32_avx2_harley_seal"
};
/*-**********************************************************************
*  This section contains the higher level functions for computing the
*  positional population count.
************************************************************************/

// Function pointer definition.
typedef int(*pospopcnt_u16_method_type)(const uint16_t* data, uint32_t len, uint32_t* flags);
typedef void(*pospopcnt_u8_method_type)(const uint8_t* data, size_t len, uint32_t* flags);
typedef void(*pospopcnt_u32_method_type)(const uint32_t* data, size_t len, uint32_t* flags);

/**
 * @brief Default function for computing the positional popcnt statistics.
 *        Redirects to the best available algorithm during compilation time.
 *
 * Example usage:
 *
 * pospopcnt_u16(data, len, flags);
 *
 * @param data  Input uint16_t data.
 * @param len   Number of input integers.
 * @param flags Output target flags.
 * @return int  Returns 0.
 */
int pospopcnt_u16(const uint16_t* data, uint32_t len, uint32_t* flags);

/**
 * @brief Execute the target ppopcnt function with the argument data.
 *
 * Example usage:
 *
 * pospopcnt_u16_method(PPOPCNT_AVX2_MULA_UR8, data, len, flags);
 *
 * @param method Target function (PPOPCNT_U16_METHODS).
 * @param data   Input uint16_t data.
 * @param len    Number of input integers.
 * @param flags  Output target flags.
 * @return int   Returns 0.
 */
int pospopcnt_u16_method(PPOPCNT_U16_METHODS method, const uint16_t* data, uint32_t len, uint32_t* flags);

/**
 * @brief Retrieve the target pospopcnt_u16_method pointer.
 *
 * Example usage:
 *
 * pospopcnt_u16_method_type f = get_pospopcnt_u16_method(PPOPCNT_AVX2_HARLEY_SEAL);
 * (*f)(data, len, flags);
 *
 * @param method                     Target function (PPOPCNT_U16_METHODS).
 * @return pospopcnt_u16_method_type Returns the target function pointer.
 */
pospopcnt_u16_method_type get_pospopcnt_u16_method(PPOPCNT_U16_METHODS method);


/*-**********************************************************************
*  This section contains declarations for individual pospopcnt subroutines.
*
*  Declarations for pospopcnt_u16_* functions parameterized by the input
*  stream of 16-bit integers (data), the number of integers (len), and
*  the output counter destination (flags). The output counter destination
*  must be pre-allocated to sixteen 32-bit integers and initiated to zero
*  before calling pospopcnt_u16_* the first time.
*
*  Function names are prefixed by its target instruction set:
*  [scalar, sse, avx2, avx512]. For example, pospopcnt_u16_avx2_harley_seal.
************************************************************************/
int pospopcnt_u16_scalar_naive(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_naive_nosimd(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_partition(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_hist1x4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_umul128(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_umul128_unroll2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_single(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_blend_popcnt(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_blend_popcnt_unroll4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_blend_popcnt_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_blend_popcnt_unroll16(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_sad(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_harley_seal(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_popcnt(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_naive_counter(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_single(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_lemire(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_lemire2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_blend_popcnt(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_mula2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_adder_forest(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_blend_popcnt_unroll4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_blend_popcnt_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_blend_popcnt_unroll16(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_harley_seal(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_popcnt32_mask(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_popcnt64_mask(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512_masked_ops(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512_popcnt(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_blend_popcnt(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_blend_popcnt_unroll4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_blend_popcnt_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512_mula2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_adder_forest(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_harley_seal(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512vbmi_harley_seal(const uint16_t* data, uint32_t len, uint32_t* flags);

/**
 * @brief Retrieve the target pospopcnt_u8_method pointer.
 *
 * Example usage:
 *
 * pospopcnt_u8_method_type f = get_pospopcnt_u8_method(PPOPCNT_U8_AVX2_HARLEY_SEAL);
 * (*f)(data, len, flags);
 *
 * @param method                    Target function (PPOPCNT_U8_METHODS).
 * @return pospopcnt_u8_method_type Returns the target function pointer.
 */
pospopcnt_u8_method_type get_pospopcnt_u8_method(PPOPCNT_U8_METHODS method);

void pospopcnt_u8_scalar_naive(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_scalar_naive_nosimd(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_scalar_partition(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_scalar_hist1x4(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_scalar_umul128(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_scalar_umul128_unroll2(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_single(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_blend_popcnt(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_blend_popcnt_unroll4(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_blend_popcnt_unroll8(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_blend_popcnt_unroll16(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_sad(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_harley_seal(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_sse_popcnt4bit(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_popcnt(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_naive_counter(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_single(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_lemire(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_lemire2(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_blend_popcnt(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_mula2(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_adder_forest(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_blend_popcnt_unroll4(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_blend_popcnt_unroll8(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_blend_popcnt_unroll16(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_harley_seal(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx2_popcnt4bit(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_popcnt32_mask(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_popcnt64_mask(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512_masked_ops(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512_popcnt(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_blend_popcnt(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_blend_popcnt_unroll4(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_blend_popcnt_unroll8(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512_mula2(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_adder_forest(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_harley_seal(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512bw_popcnt4bit(const uint8_t* data, size_t len, uint32_t* flags);
void pospopcnt_u8_avx512vbmi_harley_seal(const uint8_t* data, size_t len, uint32_t* flags);

/**
 * @brief Retrieve the target pospopcnt_u32_method pointer.
 *
 * Example usage:
 *
 * pospopcnt_u32_method_type f = get_pospopcnt_u32_method(PPOPCNT_U32_SCALAR);
 * (*f)(data, len, flags);
 *
 * @param method                     Target function (PPOPCNT_U32_METHODS).
 * @return pospopcnt_u32_method_type Returns the target function pointer.
 */
pospopcnt_u32_method_type get_pospopcnt_u32_method(PPOPCNT_U32_METHODS method);

void pospopcnt_u32_scalar_naive(const uint32_t* data, size_t len, uint32_t* flags);
void pospopcnt_u32_sse_harley_seal(const uint32_t* data, size_t len, uint32_t* flags);
void pospopcnt_u32_avx2_harley_seal(const uint32_t* data, size_t len, uint32_t* flags);

/*======   Support   ======*/
// Wrapper for avx512*_harley_seal
int pospopcnt_u16_avx512_harley_seal(const uint16_t* data, uint32_t len, uint32_t* flags);

#ifdef __cplusplus
}
#endif

#endif /* POSPOPCNT_H_2359235897293 */
