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
 * These functions compute the novel "positional population count" 
 * (pospopcnt) statistics using fast SIMD instructions. Given a stream
 * of k-bit words, we seek to count the number of set bits in positions 
 * 0, 1, 2, ..., k-1. This problem is a generalization of the 
 * population-count problem where we count the sum total of set bits 
 * in a k-bit word.
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

#ifdef _mm_popcnt_u64
#define PIL_POPCOUNT _mm_popcnt_u64
#else
#define PIL_POPCOUNT __builtin_popcountll
#endif

PPOPCNT_INLINE uint64_t pospopcnt_umul128(uint64_t a, uint64_t b, uint64_t* hi) {
    unsigned __int128 x = (unsigned __int128)a * (unsigned __int128)b;
    *hi = (uint64_t)(x >> 64);
    return (uint64_t)x;
}

PPOPCNT_INLINE uint64_t pospopcnt_loadu_u64(const void* ptr) {
    uint64_t data;
    memcpy(&data, ptr, sizeof(data));
    return data;
}

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
PPOPCNT_INLINE void pospopcnt_csa_sse(__m128i* __restrict__ h, 
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

PPOPCNT_INLINE void pospopcnt_csa_avx2(__m256i* __restrict__ h, 
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

// 512i-version of pospopcnt_csa_AVX2
PPOPCNT_INLINE void pospopcnt_csa_avx512(__m512i* __restrict__ h, 
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

#define PPOPCNT_NUMBER_METHODS 38

typedef enum {
    PPOPCNT_AUTO,           PPOPCNT_SCALAR,
    PPOPCNT_SCALAR_NOSIMD,  PPOPCNT_SCALAR_PARTITION,
    PPOPCNT_SCALAR_HIST1X4,
    PPOPCNT_SCALAR_UMUL128, PPOPCNT_SCALAR_UMUL128_UR2,
    PPOPCNT_SSE_SINGLE,
    PPOPCNT_SSE_MULA,       PPOPCNT_SSE_MULA_UR4,
    PPOPCNT_SSE_MULA_UR8,   PPOPCNT_SSE_MULA_UR16,
    PPOPCNT_SSE_SAD,        PPOPCNT_SSE_CSA,
    PPOPCNT_AVX2_POPCNT,
    PPOPCNT_AVX2,           PPOPCNT_AVX2_POPCNT_NAIVE,
    PPOPCNT_AVX2_SINGLE,    PPOPCNT_AVX2_LEMIRE1,
    PPOPCNT_AVX2_LEMIRE2,   PPOPCNT_AVX2_MULA,
    PPOPCNT_AVX2_MULA_UR4,  PPOPCNT_AVX2_MULA_UR8,
    PPOPCNT_AVX2_MULA_UR16, PPOPCNT_AVX2_MULA3,
    PPOPCNT_AVX2_CSA,       PPOPCNT_AVX512,
    PPOPCNT_AVX512BW_MASK32,  PPOPCNT_AVX512BW_MASK64,
    PPOSCNT_AVX512_MASKED_OPS,
    PPOPCNT_AVX512_POPCNT,  PPOPCNT_AVX512BW_MULA,
    PPOPCNT_AVX512BW_MULA_UR4,PPOPCNT_AVX512BW_MULA_UR8,
    PPOPCNT_AVX512_MULA2,   PPOPCNT_AVX512BW_MULA3,
    PPOPCNT_AVX512BW_CSA, PPOPCNT_AVX512VBMI_CSA
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
    "pospopcnt_u16_sse_mula",
    "pospopcnt_u16_sse_mula_unroll4",
    "pospopcnt_u16_sse_mula_unroll8",
    "pospopcnt_u16_sse_mula_unroll16",
    "pospopcnt_u16_sse2_sad",
    "pospopcnt_u16_sse2_csa",
    "pospopcnt_u16_avx2_popcnt",
    "pospopcnt_u16_avx2",
    "pospopcnt_u16_avx2_naive_counter",
    "pospopcnt_u16_avx2_single",
    "pospopcnt_u16_avx2_lemire",
    "pospopcnt_u16_avx2_lemire2",
    "pospopcnt_u16_avx2_mula",
    "pospopcnt_u16_avx2_mula_unroll4",
    "pospopcnt_u16_avx2_mula_unroll8",
    "pospopcnt_u16_avx2_mula_unroll16",
    "pospopcnt_u16_avx2_mula3",
    "pospopcnt_u16_avx2_csa",
    "pospopcnt_u16_avx512",
    "pospopcnt_u16_avx512bw_popcnt32_mask",
    "pospopcnt_u16_avx512bw_popcnt64_mask",
    "pospopcnt_u16_avx512_masked_ops",
    "pospopcnt_u16_avx512_popcnt",
    "pospopcnt_u16_avx512bw_mula",
    "pospopcnt_u16_avx512bw_mula_unroll4",
    "pospopcnt_u16_avx512bw_mula_unroll8",
    "pospopcnt_u16_avx512_mula2",
    "pospopcnt_u16_avx512bw_mula3",
    "pospopcnt_u16_avx512bw_csa",
    "pospopcnt_u16_avx512vbmi_csa"};

/*-**********************************************************************
*  This section contains the higher level functions for computing the
*  positional population count. 
*
************************************************************************/

// Function pointer definition.
typedef int(*pospopcnt_u16_method_type)(const uint16_t* data, uint32_t len, uint32_t* flags);

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
 * pospopcnt_u16_method_type f = get_pospopcnt_u16_method(PPOPCNT_AVX2_CSA);
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
*  [scalar, sse, avx2, avx512]. For example, pospopcnt_u16_avx2_csa.
************************************************************************/

int pospopcnt_u16_scalar_naive(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_naive_nosimd(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_partition(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_hist1x4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_umul128(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_scalar_umul128_unroll2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_single(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_mula(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_mula_unroll4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_mula_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_mula_unroll16(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_sad(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_sse_csa(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_popcnt(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_naive_counter(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_single(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_lemire(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_lemire2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_mula(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_mula2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_mula3(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_mula_unroll4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_mula_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_mula_unroll16(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx2_csa(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_popcnt32_mask(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_popcnt64_mask(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512_masked_ops(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512_popcnt(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_mula(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_mula_unroll4(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_mula_unroll8(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512_mula2(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_mula3(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512bw_csa(const uint16_t* data, uint32_t len, uint32_t* flags);
int pospopcnt_u16_avx512vbmi_csa(const uint16_t* data, uint32_t len, uint32_t* flags);

// Support
// Wrapper for avx512_csa
int pospopcnt_u16_avx512_csa(const uint16_t* data, uint32_t len, uint32_t* flags);

#ifdef __cplusplus
}
#endif

#endif /* POSPOPCNT_H_2359235897293 */
