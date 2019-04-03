# positional-popcount

These functions compute the novel "positional [population count](https://en.wikipedia.org/wiki/Hamming_weight)" (`pospopcnt`) statistics using fast [SIMD instructions](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions). Given a stream of k-bit words, we seek to count the number of set bits in positions 0, 1, 2, ..., k-1. This problem is a generalization of the population-count problem where we count the sum total of set bits in a k-bit word.

These functions can be applied to any packed [1-hot](https://en.wikipedia.org/wiki/One-hot) 16-bit primitive, for example in machine learning/deep learning. Using large registers (AVX-512), we can achieve >16 GB/s (~0.125 CPU cycles / int) throughput (8.4 billion 16-bit integers / second or 134 billion one-hot vectors / second).

### Speedup

This benchmark shows the speedup of the 3 `pospopcnt` algorithms used on x86 CPUs compared to the efficient auto-vectorization of `pospopcnt_u16_scalar_naive` for different array sizes (in number of 2-byte values). See [Results](#results) for additional information.

| Algorithm       | 32   | 128  | 512   | 2048  | 8192  | 16384 | 32768 |
|--------------|------|------|-------|-------|-------|-------|-------|
| SSE_SAD      | **5.74** | **5.02** | 3.2   | 2.36  | 2.11  | 2.09  | 2.06  |
| AVX512_MULA3 | 0.76 | 0.9  | **4.07**  | **5.31**  | 6.73  | 6.94  | 7.25  |
| AVX512_CSA   | 0.53 | 0.65 | 2     | 4.64  | **10.65** | **13.35** | **16.77** |

Compared to a naive unvectorized solution (`pospopcnt_u16_scalar_naive_nosimd`):

| Method       | 32    | 128   | 512    | 2048   | 8192  | 16384  | 32768  |
|--------------|-------|-------|--------|--------|-------|--------|--------|
| SSE_SAD      | **5.67**  | **10.54** | 14.93  | 16.64  | 17.21 | 17.39  | 17.38  |
| AVX512_MULA3 | 0.75  | 1.89  | **18.98**  | **37.39**  | 54.8  | 57.78  | 61.06  |
| AVX512_CSA   | 0.52  | 1.37  | 9.34   | 32.63  | **86.66** | **111.16** | **141.32** |

The host architecture used is a 10 nm Cannon Lake [Core i3-8121U](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html) with gcc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5).

### Usage

Compile the test suite with: `make` and run `./bench`. The test suite require `c++11` whereas the example and functions require only `c99`. For more detailed test, see [Instrumented tests (Linux specific)](#instrumented-tests-linux-specific).

Include `pospopcnt.h` and `pospopcnt.c` in your project. Then use the wrapper function for `pospopcnt`:
```c
pospopcnt_u16(datain, length, target_counters);
```

See `example.c` for a complete example. Compile with `make example`.

### Note

This is a collaborative effort between Marcus D. R. Klarqvist ([@klarqvist](https://github.com/mklarqvist/)), Wojciech Muła ([@WojciechMula](https://github.com/WojciechMula)), and Daniel Lemire ([@lemire](https://github.com/lemire/)).

### History

These functions were developed for [pil](https://github.com/mklarqvist/pil) but can be applied to any 1-hot count problem.

### Table of contents

  - [Problem statement](#problem-statement)
  - [Goals](#goals)
  - [Technical approach](#technical-approach)
    - [Approach 0: Naive iterator (scalar)](#approach-0-naive-iterator-scalar)
    - [Approach 1: Byte-partition accumulator (scalar)](#approach-1-byte-partition-accumulator-scalar)
    - [Approach 2: Shift-pack popcount accumulator (SIMD)](#approach-2-shift-pack-popcount-accumulator-simd)
    - [Approach 3: Register accumulator and aggregator (AVX2)](#approach-3-register-accumulator-and-aggregator-avx2)
    - [Approach 3b: Register accumulator and aggregator (AVX512)](#approach-3b-register-accumulator-and-aggregator-avx512)
    - [Approach 4a: Interlaced register accumulator and aggregator (AVX2)](#approach-4a-interlaced-register-accumulator-and-aggregator-avx2)
    - [Approach 4b: Interlaced register accumulator and aggregator (SSE4.1)](#approach-4b-interlaced-register-accumulator-and-aggregator-sse41)
    - [Approach 5: Popcount predicate-mask accumulator (AVX-512)](#approach-5-popcount-predicate-mask-accumulator-avx-512)
    - [Approach 6: Partial-sum accumulator and aggregator (AVX-512)](#approach-6-partial-sum-accumulator-and-aggregator-avx-512)
    - [Approach 7: Shift-mask accumulator [Muła] (SIMD)](#approach-7-shift-mask-accumulator-muła-simd)
  - [Results](#results)
  - [Instrumented tests (Linux specific)](#instrumented-tests-linux-specific)

---

## Problem statement

The FLAG field in the [SAM interchange format](https://github.com/samtools/hts-specs) is defined as the union of [1-hot](https://en.wikipedia.org/wiki/One-hot) encoded states for a given read. For example, the following three states evaluating to true

```
00000001: read paired
01000000: first in pair
00001000: mate unmapped
--------
01001001: Decimal (73)
```

are stored in a packed 16-bit value (only the LSB is shown here). There are 12 states described in the SAM format:

| One-hot           | Description                               |
|-------------------|-------------------------------------------|
| 00000000 00000001 | Read paired                               |
| 00000000 00000010 | Read mapped in proper pair                |
| 00000000 00000100 | Read unmapped                             |
| 00000000 00001000 | Mate unmapped                             |
| 00000000 00010000 | Read reverse strand                       |
| 00000000 00100000 | Mate reverse strand                       |
| 00000000 01000000 | First in pair                             |
| 00000000 10000000 | Second in pair                            |
| 00000001 00000000 | Not primary alignment                     |
| 00000010 00000000 | Read fails platform/vendor quality checks |
| 00000100 00000000 | Read is PCR or optical duplicate          |
| 00001000 00000000 | Supplementary alignment                   |

Computing FLAG statistics from readsets involves iteratively incrementing up to 16 counters. The native properties of a column-oriented storage, specifically column projection, already deliver good performance because of data locality (memory contiguity) and value typing. We want to maximize compute on large arrays of values by exploiting vectorized instructions, if available.

## Goals

* Achieve high-performance on large arrays of values.
* Support machines without SIMD (scalar).
* Specialized algorithms for SSE2 up to AVX512.

## Technical approach

### Approach 0: Naive iterator (scalar)

We compare our proposed algorithms to a naive implementation using standard incrementors:

```python
for i in 1..n # n -> n_records
    for j in 1..16 # every possible 1-hot state
        flags[j] += ((data[i] & (1 << j)) >> j) # predicate add
```

This branchless code will optimize extremely well on most machines. Knowledge of the host-architecture by the compiler makes this codes difficult to outperform on average.

### Approach 1: Byte-partition accumulator (scalar)

There is no dependency between bits as they represent 1-hot encoded values. Because of this, we can partition bytes into 512 distinct bins (256 MSB and 256 LSB) and compute their frequencies. Then, in a final reduce step, we projects the bin counts into the target flag accumulator. In total, there are two updates per record instead of 16, followed by a single fixed-sized projection step.  

Projecting bin counts into the FLAG array can be done using branchless updates with Boolean multiplications (multiplying a value with a Boolean). The following block describes this approach using psuedo-code:

```python
low[256]  init 0
high[256] init 0

# Iterate over elements in an array of FLAG values
for i in 1..n # n -> n_records
    low  += data[n] & 255
    high += (data[n] >> 8) & 255

flags[16] init 0

# Up to 16 * 255 updates compared to 16*n updates
# Low byte (LSB) of FLAG space
for i in 0..256 # Iterate over all 256 possible bytes
    skip = (low[i] == 0) # Skip empty bytes
    for k in 0..(8*skip) # Iterate over the bits in i
        flags[k] += ((i & (1 << k)) >> k) * low[i] # predicate multiply
                   # ^ bit k set in value i?
                   #                      ^ multipy predicate with count

# High byte (MSB)
for i in 0..256
    skip = (high[i] == 0)
    for k in 0..(8*skip)
        flags[k+8] += ((i & (1 << k)) >> k) * high[i]
```

### Approach 2: Shift-pack popcount accumulator (SIMD)

Shift in 16 one-hot values into a `uint16_t` primitive and compute the [population bit-count](https://en.wikipedia.org/wiki/Hamming_weight) (`popcnt`) and increment the target counter.

Psuedo-code for the conceptual model:

```python
for c in 1..n, c+=16 # 1->n with stride 16
    for i in 1..16 # FLAG 1->16 in range
        for j in 1..16 # Each 1-hot vector state
            y[j] |= (((x[c+i] & (1 << j)) >> j) << i)
    
    for i in 1..16 # 1->16 packed element
        out[i] += popcnt(y[i]) # popcount
        y[j] = 0 # reset
```

Example C++ implementation using AVX2:

```c++
__m256i masks[16]; // one register bitmask / 1-hot
__m256i stubs[16]; // location for shift-packing

for(int i = 0; i < 16; ++i) {
    masks[i] = _mm256_set1_epi16(1 << i); // i-th 16-bit mask
    stubs[i] = _mm256_set1_epi16(0); // zero allocation
}

uint32_t out_counters[16] = {0}; // accumulators
const __m256i* data_vectors = (const __m256i*)(data);

// Define a macro UPDATE representing a single update step:
#define UPDATE(idx, shift) stubs[idx] = _mm256_or_si256(stubs[idx], _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(data_vectors[pos], masks[idx]),  idx), shift));
// Unroll the inner loop iterations for each 1-hot
#define ITERATION(idx) {                                           \
    UPDATE(idx,0);  UPDATE(idx,1);  UPDATE(idx,2);  UPDATE(idx,3); \
    UPDATE(idx,4);  UPDATE(idx,5);  UPDATE(idx,6);  UPDATE(idx,7); \
    UPDATE(idx,8);  UPDATE(idx,9);  UPDATE(idx,10); UPDATE(idx,11);\
    UPDATE(idx,12); UPDATE(idx,13); UPDATE(idx,14); UPDATE(idx,15);\
    ++pos;                                                         \
}
// Unroll the out loop iterators for each block of 16 FLAGs
#define BLOCK {                                                \
    ITERATION(0);  ITERATION(1);  ITERATION(2);  ITERATION(3); \
    ITERATION(4);  ITERATION(5);  ITERATION(6);  ITERATION(7); \
    ITERATION(8);  ITERATION(9);  ITERATION(10); ITERATION(11);\
    ITERATION(12); ITERATION(13); ITERATION(14); ITERATION(15);\
}

uint32_t pos = 0;
for(int i = 0; i < n_cycles_updates; ++i) {
    BLOCK // unrolled

    for(int j = 0; j < 16; ++j) {
        PIL_POPCOUNT_AVX2(out_counters[j], stubs[j]) // popcnt
        stubs[j] = _mm256_set1_epi16(0); // reset
    }
}

// Residual FLAGs that are not multiple of 16
// Scalar approach:
for(int i = pos*16; i < n; ++i) {
    for(int j = 0; j < 16; ++j) {
        out_counters[j] += ((data[i] & (1 << j)) >> j);
    }
}
```

### Approach 3a: Register accumulator and aggregator (AVX2)

Accumulate up to 16 * 2^16 partial sums of a 1-hot in a single register followed by a horizontal sum update. By using 16-bit partial sum accumulators we must perform a secondary accumulation step every 2^16 iterations to prevent overflowing the 16-bit primitives.

Psuedo-code for the conceptual model:

```python
for i in 1..n, i+=65536 # 1->n with stride 65536
    for c in 1..65536 # section of 65536 iterations to prevent overflow
        for j in 1..16 # Each 1-hot vector state
            y[j] += ((x[c+i] & (1 << j)) >> j)
    
    for j in 1..16 # 1->16 packed element
        out[j] += y[j] # accumulate
        y[j] = 0 # reset
```

Example C++ implementation using AVX2:

```c++
__m256i masks[16];
__m256i counters[16];
for(int i = 0; i < 16; ++i) {
    masks[i]    = _mm256_set1_epi16(1 << i); // one register bitmask / 1-hot
    counters[i] = _mm256_set1_epi16(0); // partial accumulators
}
uint32_t out_counters[16] = {0}; // larger accumulators

const __m256i hi_mask = _mm256_set1_epi32(0xFFFF0000); // MSB mask of 32-bit
const __m256i lo_mask = _mm256_set1_epi32(0x0000FFFF); // LSB mask of 32-bit
const __m256i* data_vectors = (const __m256i*)(data);
const uint32_t n_cycles = n / 16;
const uint32_t n_update_cycles = std::floor((double)n_cycles / 65536);

// Define a macro UPDATE representing a single update step:
#define UPDATE(idx) counters[idx]  = _mm256_add_epi16(counters[idx],  _mm256_srli_epi16(_mm256_and_si256(data_vectors[pos], masks[idx]),  idx))
// Unroll 16 updates
#define ITERATION  {                               \
    UPDATE(0);  UPDATE(1);  UPDATE(2);  UPDATE(3); \
    UPDATE(4);  UPDATE(5);  UPDATE(6);  UPDATE(7); \
    UPDATE(8);  UPDATE(9);  UPDATE(10); UPDATE(11);\
    UPDATE(12); UPDATE(13); UPDATE(14); UPDATE(15);\
    ++pos; ++k;                                    \
}

uint32_t pos = 0; // total offset
for(int i = 0; i < n_update_cycles; ++i) { // each block of 2^16 values
    for(int k = 0; k < 65536; ) { // max sum of each 16-bit value in a register
        ITERATION // unrolled
    }

    // Compute vector sum
    for(int k = 0; k < 16; ++k) { // each flag register        
        // Expand 16-bit counters into 32-bit counters
        // ((16-bit high & 16 high) >> 16) + (16-bit low & 16-low)
        __m256i x = _mm256_add_epi32(
                      _mm256_srli_epi32(_mm256_and_si256(counters[k], hi_mask), 16),
                      _mm256_and_si256(counters[k], lo_mask));
        // Compute vector sum
        __m256i t1 = _mm256_hadd_epi32(x,x);
        __m256i t2 = _mm256_hadd_epi32(t1,t1);
        __m128i t4 = _mm_add_epi32(_mm256_castsi256_si128(t2),_mm256_extractf128_si256(t2,1));
        out_counters[k] += _mm_cvtsi128_si32(t4); // accumulate
        counters[k]      = _mm256_set1_epi16(0); // reset
    }
}

// Residual FLAGs that are not multiple of 16
// Scalar approach:
for(int i = pos*16; i < n; ++i) {
    for(int j = 0; j < 16; ++j) {
        out_counters[j] += ((data[i] & (1 << j)) >> j);
    }
}
```

### Approach 3b: Register accumulator and aggregator (AVX512)

This algorithm is a AVX-512 implementation of approach 3a using 512-bit registers
by computing 16 * 2^32 partial sums. The AVX512 instruction set do not provide
native instructions to perform 16-bit-wise sums of registers. By being restricted to 32-bit accumulators while consuming 16-bit primitives we must performed a second 16-bit shift-add operation to mimic 32-bit behavior. Unlike the AVX2 algortihm, the 32-bit accumulators in this version do not require blocking under the expectation that the total count in either slot do not exceed 2^32.

Psuedo-code for the conceptual model:

```python
for c in 1..n # primitive type is now uint32_t (2x uint16_t)
    for j in 1..16 # Each 1-hot vector state
        y[j] += ((x[i] & (1 << j)) >> j) + ((x[i] & (1 << (16+j))) >> (j+16))
    
    for i in 1..16 # 1->16 packed element
        out[i] += y[i] # accumulate
        y[j] = 0 # reset
```

Example C++ implementation using AVX-512:

```c++
__m512i masks[16];
__m512i counters[16];
const __m512i one_mask = _mm512_set1_epi32(1);
for(int i = 0; i < 16; ++i) {
    masks[i]    = _mm512_set1_epi16(1 << i);  // one register bitmask / 1-hot
    counters[i] = _mm512_set1_epi32(0); // partial accumulators
}
uint32_t out_counters[16] = {0};  // output accumulators

const __m512i* data_vectors = (const __m512i*)(data);
const uint32_t n_cycles = n / 32;

// Define a macro UPDATE representing a single update step:
#define UPDATE(pos) {                                        \
__m512i a   = _mm512_and_epi32(data_vectors[i], masks[pos]); \
__m512i d   = _mm512_add_epi32(_mm512_and_epi32(_mm512_srli_epi32(a, pos), one_mask), _mm512_srli_epi32(a, pos+16)); \
counters[pos] = _mm512_add_epi32(counters[pos], d);          \
}
// Unroll 16 updates
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

for(int i = 0; i < n_cycles; ++i) {
    BLOCK
}

#undef BLOCK
#undef UPDATE

// Residual FLAGs that are not multiple of 16
// Scalar approach:
for(int i = n_cycles*32; i < n; ++i) {
    for(int j = 0; j < 16; ++j)
        out_counters[j] += ((data[i] & (1 << j)) >> j);
}

// Transfer SIMD aggregator to output scalar counters.
for(int i = 0; i < 16; ++i) {
    uint32_t* v = (uint32_t*)(&counters[i]);
    for(int j = 0; j < 16; ++j)
        out_counters[i] += v[j];
}
```

### Approach 4a: Interlaced register accumulator and aggregator (AVX2)

Instead of having 16 registers of 16 values of partial sums for each 1-hot state we have a single register with 16 partial sums for the different 1-hot states. We achieve this by broadcasting a single integer to all slots in a register and performing a 16-way comparison.

Psuedo-code for the conceptual model:

```python
for i in 1..n, i+=4096 # 1->n with stride 4096
    for c in 1..4096 # Block of 4096 iterations to prevent overflow
        f = {x[c], x[c], x[c], x[c], ..., x[c]} # 16 copies of x[c]
        for j in 1..16 # Each 1-hot vector state
            y[j] += (((f[j] & (1 << j)) == (1 << j)) & 1)
    
    for j in 1..16 # 1->16 packed element
        out[j] += y[j] # accumulate
        y[j] = 0 # reset
```

Example C++ implementation using AVX2:

```c++
__m256i counter = _mm256_set1_epi16(0);
const __m256i one_mask =  _mm256_set1_epi16(1);
// set_epi is parameterized in reverse order (15->0)
const __m256i masks = _mm256_set_epi16(
        1 << 15, 1 << 14, 1 << 13, 1 << 12,
        1 << 11, 1 << 10, 1 << 9,  1 << 8,
        1 << 7,  1 << 6,  1 << 5,  1 << 4,
        1 << 3,  1 << 2,  1 << 1,  1 << 0);
uint32_t out_counters[16] = {0};
const __m256i* data_vectors = (const __m256i*)(data);
const uint32_t n_cycles = n / 16;
const uint32_t n_update_cycles = std::floor((double)n_cycles / 4096);

// Define a macro UPDATE representing a single update step:
#define UPDATE(idx) counter = _mm256_add_epi16(counter, _mm256_and_si256(_mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16(_mm256_extract_epi16(data_vectors[pos], idx)), masks), masks), one_mask));
// Unroll 16 updates in a register
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

// Residual FLAGs that are not multiple of 16
// Scalar approach:
for(int i = pos*16; i < n; ++i) {
    for(int j = 0; j < 16; ++j)
        out_counters[j] += ((data[i] & (1 << j)) >> j);
}
```

### Approach 4b: Interlaced register accumulator and aggregator (SSE4.1)

This algorithm is a SSE-based version of approach 4a using 128-bit registers.

Example C++ implementation using SSE4.1:

```c++
__m128i counterLo = _mm_set1_epi16(0);
__m128i counterHi = _mm_set1_epi16(0);
const __m128i one_mask =  _mm_set1_epi16(1);
// set_epi is parameterized in reverse order (7->0)
const __m128i masksLo = _mm_set_epi16(
        1 << 15, 1 << 14, 1 << 13, 1 << 12,
        1 << 11, 1 << 10, 1 << 9,  1 << 8);
const __m128i masksHi = _mm_set_epi16(
        1 << 7,  1 << 6,  1 << 5,  1 << 4,
        1 << 3,  1 << 2,  1 << 1,  1 << 0);

uint32_t out_counters[16] = {0};
const __m128i* data_vectors = (const __m128i*)(data);
const uint32_t n_cycles = n / 8;
const uint32_t n_update_cycles = std::floor((double)n_cycles / 4096);

// Define a macro UPDATE for the LSB:
#define UPDATE_LO(idx) counterLo = _mm_add_epi16(counterLo, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(_mm_extract_epi16(data_vectors[pos], idx)), masksLo), masksLo), one_mask));
// Define a macro UPDATE for the MSB:
#define UPDATE_HI(idx) counterHi = _mm_add_epi16(counterHi, _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(_mm_extract_epi16(data_vectors[pos], idx)), masksHi), masksHi), one_mask));
// Unroll the update for 16 values
#define BLOCK {                                         \
    UPDATE_LO(0) UPDATE_LO(1) UPDATE_LO(2) UPDATE_LO(3) \
    UPDATE_LO(4) UPDATE_LO(5) UPDATE_LO(6) UPDATE_LO(7) \
    UPDATE_HI(0) UPDATE_HI(1) UPDATE_HI(2) UPDATE_HI(3) \
    UPDATE_HI(4) UPDATE_HI(5) UPDATE_HI(6) UPDATE_HI(7) \
}
// Define a macro for counting slots in the low register
#define UH(idx) out_counters[idx] += _mm_extract_epi16(counterLo, idx - 8);
// Define a macro for counting slots in the high register
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

// Residual FLAGs that are not multiple of 16
// Scalar approach:
for(int i = pos*8; i < n; ++i) {
    for(int j = 0; j < 16; ++j)
        out_counters[j] += ((data[i] & (1 << j)) >> j);
}
```

### Approach 5: Popcount predicate-mask accumulator (AVX-512)

The AVX-512 instruction set comes with the new `vpcmpuw` (`_mm512_cmpeq_epu16_mask`) instruction
that returns the equality predicate of two registers as a packed 32-bit integer (`__mask32`). This
algorithm combines this packed integer with a 32-bit `popcnt` operation. In the second approach (64-bit) we pack two 32-bit masks into a 64-bit primitive before performing a `popcnt` operation on the packed mask.

Example C++ implementation using 32-bit-mask:

```c++
__m512i masks[16];
for(int i = 0; i < 16; ++i) {
    masks[i] = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
}
uint32_t out_counters[16] = {0};

const __m512i* data_vectors = (const __m512i*)(data);
const uint32_t n_cycles = n / 32;

#define UPDATE(pos) out_counters[pos] += PIL_POPCOUNT((uint64_t)_mm512_cmpeq_epu16_mask(_mm512_and_epi32(data_vectors[i], masks[pos]), masks[pos]));
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

for(int i = 0; i < n_cycles; ++i) {
    BLOCK
}

#undef BLOCK
#undef UPDATE

// Residual FLAGs that are not multiple of 32
// Scalar approach:
for(int i = n_cycles*32; i < n; ++i) {
    for(int j = 0; j < 16; ++j)
        out_counters[j] += ((data[i] & (1 << j)) >> j);
}
```

Example C++ implementation packing two 32-bit masks into a 64-bit primitive:

```c++
__m512i masks[16];
for(int i = 0; i < 16; ++i) {
    masks[i] = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
}
uint32_t out_counters[16] = {0};

const __m512i* data_vectors = (const __m512i*)(data);
const uint32_t n_cycles = n / 32;

#define UPDATE(pos,add) (uint64_t)_mm512_cmpeq_epu16_mask(_mm512_and_epi32(data_vectors[i+add], masks[pos]), masks[pos])
#define UP(pos) out_counters[pos] += PIL_POPCOUNT((UPDATE(pos,0) << 32) | UPDATE(pos,1));
#define BLOCK {                 \
    UP(0)  UP(1)  UP(2)  UP(3)  \
    UP(4)  UP(5)  UP(6)  UP(7)  \
    UP(8)  UP(9)  UP(10) UP(11) \
    UP(12) UP(13) UP(14) UP(15) \
}

for(int i = 0; i < n_cycles; i += 2) {
    BLOCK
}

#undef BLOCK
#undef UP
#undef UPDATE

// Residual FLAGs that are not multiple of 32
// Scalar approach:
for(int i = n_cycles*32; i < n; ++i) {
    for(int j = 0; j < 16; ++j)
        out_counters[j] += ((data[i] & (1 << j)) >> j);
}
```

### Approach 6: Partial-sum accumulator and aggregator (AVX-512)

This algoritmh computes the partial sums by a 16-way predicate incrementation `popcnt`.

Example C++ implementation:

```c++
__m512i masks[16]; // 1-hot masks
__m512i counters[16]; // partial sums aggregators
for(int i = 0; i < 16; ++i) {
    masks[i]    = _mm512_set1_epi32(((1 << i) << 16) | (1 << i));
    counters[i] = _mm512_set1_epi32(0);
}
uint32_t out_counters[16] = {0};

const __m512i* data_vectors = (const __m512i*)(data);
const uint32_t n_cycles = n / 32;

// Define a macro UPDATE representing a single update step:
#define UPDATE(pos) counters[pos] = _mm512_add_epi32(counters[pos], avx512_popcount(_mm512_and_epi32(data_vectors[i], masks[pos])));
// Unroll the update for 16 values
#define BLOCK {                                 \
    UPDATE(0)  UPDATE(1)  UPDATE(2)  UPDATE(3)  \
    UPDATE(4)  UPDATE(5)  UPDATE(6)  UPDATE(7)  \
    UPDATE(8)  UPDATE(9)  UPDATE(10) UPDATE(11) \
    UPDATE(12) UPDATE(13) UPDATE(14) UPDATE(15) \
}

for(int i = 0; i < n_cycles; i+=16) { // each block of 2^16 values
    BLOCK
}

#undef BLOCK
#undef UPDATE

// Residual FLAGs that are not multiple of 16
// Scalar approach:
for(int i = n_cycles*32; i < n; ++i) {
    for(int j = 0; j < 16; ++j)
        out_counters[j] += ((data[i] & (1 << j)) >> j);
}

// Reduce phase: transfer partial sums to final aggregators
for(int i = 0; i < 16; ++i) {
    uint32_t* v = (uint32_t*)(&counters[i]);
    for(int j = 0; j < 16; ++j)
        out_counters[i] += v[j];
}
```

### Approach 7: Shift-mask accumulator [Muła] (SIMD)

```c++
const __m128i* data_vectors = (const __m128i*)(array);
const uint32_t n_cycles = len / 8;

size_t i = 0;
for (/**/; i + 2 <= n_cycles; i += 2) {
    __m128i v0 = data_vectors[i+0];
    __m128i v1 = data_vectors[i+1];

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
```

### Results

We simulated 100 million FLAG fields using a uniform distrubtion U(min,max) with the arguments [{1,8},{1,16},{1,64},{1,256},{1,512},{1,1024},{1,4096},{1,65536}] for 20 repetitions using a single core. Numbers represent the average throughput in MB/s (1 MB = 1024b) or average number of CPU cycles per integer. There is no difference in throughput speed between the different distributions (now shown). Because of this we report only results for the distribution {1,65536}. The host architecture used is a 10 nm Cannon Lake [Core i3-8121U](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html), a 14 nm Sky Lake [Xeon W-2104](https://ark.intel.com/content/www/us/en/ark/products/125039/intel-xeon-w-2104-processor-8-25m-cache-3-20-ghz.html), and a 22 nm Haswell [Xeon E5-2697 v3](https://ark.intel.com/content/www/us/en/ark/products/81059/intel-xeon-processor-e5-2697-v3-35m-cache-2-60-ghz.html).

Throughput in CPU cycles / 16-bit integer (lower is better):

| Method                             | Cannon Lake | Sky Lake | Haswell |
|------------------------------------|-------------|----------|---------|
| pospopcnt_u16_scalar_naive_nosimd  | 17.525      | 17.653   | 18.012  |
| pospopcnt_u16_scalar_naive         | 2.058       | 3.014    | 3.769   |
| pospopcnt_u16_scalar_partition     | 3.094       | 3.018    | 3.402   |
| pospopcnt_u16_hist1x4              | 2.865       | 2.918    | 3.169   |
| pospopcnt_u16_sse_single           | 3.614       | 3.829    | 4.277   |
| pospopcnt_u16_sse_mula             | 2.082       | 1.629    | 2.139   |
| pospopcnt_u16_sse_mula_unroll4     | 1.578       | 1.4      | 1.625   |
| pospopcnt_u16_sse_mula_unroll8     | 1.433       | 1.351    | 1.544   |
| pospopcnt_u16_sse_mula_unroll16    | 1.386       | 1.421    | 1.543   |
| pospopcnt_u16_avx2_popcnt          | 2.406       | 2.437    | 3.012   |
| pospopcnt_u16_avx2                 | 2.035       | 3.019    | 4.019   |
| pospopcnt_u16_avx2_naive_counter   | 2.033       | 3.025    | 3.834   |
| pospopcnt_u16_avx2_single          | 2.03        | 3.013    | 3.925   |
| pospopcnt_u16_avx2_lemire          | 2.862       | 1.916    | 2.192   |
| pospopcnt_u16_avx2_lemire2         | 1.695       | 1.127    | 1.469   |
| pospopcnt_u16_avx2_mula            | 1.105       | 1.4      | 1.111   |
| pospopcnt_u16_avx2_mula2           | 1.468       | 1.423    | 1.268   |
| pospopcnt_u16_avx2_mula3           | 0.435       | 0.418    | 0.478   |
| pospopcnt_u16_avx2_mula_unroll4    | 0.848       | 0.855    | 0.842   |
| pospopcnt_u16_avx2_mula_unroll8    | 0.757       | 0.725    | 0.773   |
| pospopcnt_u16_avx2_mula_unroll16   | 0.736       | 0.747    | 0.814   |
| pospopcnt_u16_avx512               | 1.511       | 1.604    | -       |
| pospopcnt_u16_avx512_popcnt32_mask | 0.819       | 1.421    | -       |
| pospopcnt_u16_avx512_popcnt64_mask | 0.838       | 1.011    | -       |
| pospopcnt_u16_avx512_popcnt        | 1.676       | 1.751    | -       |
| pospopcnt_u16_avx512_mula          | 0.75        | 0.778    | -       |
| pospopcnt_u16_avx512_mula_unroll4  | 0.623       | 0.579    | -       |
| pospopcnt_u16_avx512_mula_unroll8  | 0.555       | 0.577    | -       |
| pospopcnt_u16_avx2_mula3           | 0.434       | 0.417    | 0.479   |
| pospopcnt_u16_avx512_mula3         | 0.288       | 0.304    | -       |
| pospopcnt_u16_avx2_csa             | 0.258       | 0.28     | 0.327   |
| pospopcnt_u16_avx512_csa           | **0.165**       | 0.191    | -       |

Throughput in MB/s (higher is better):

| Algorithm                          | Cannon Lake | Sky Lake | Haswell |
|------------------------------------|-------------|----------|---------|
| pospopcnt_u16_scalar_naive         | **2855.31**     | 1988.6   | 1141.51 |
| pospopcnt_u16_scalar_naive_nosimd  | 326.381     | **337.654**  | 269.845 |
| pospopcnt_u16_scalar_partition     | 1879.44     | **1979.75**  | 1363.66 |
| pospopcnt_u16_scalar_hist1x4       | 2015.88     | **2033.72**  | 1594.51 |
| pospopcnt_u16_sse_single           | **1508.57**     | 1456.84  | 1156.74 |
| pospopcnt_u16_sse_mula             | 2840.6      | **3485.52**  | 2342.6  |
| pospopcnt_u16_sse_mula_unroll4     | 3708.2      | **4043.73**  | 2860.88 |
| pospopcnt_u16_sse_mula_unroll8     | 4007.71     | **4197.79**  | 3215.9  |
| pospopcnt_u16_sse_mula_unroll16    | **4089.86**     | 4051.38  | 3320.59 |
| pospopcnt_u16_sse2_sad             | **5441.64**     | 5422.76  | 3589.29 |
| pospopcnt_u16_sse2_csa             | **10802.2**     | 10229.8  | 10794.3 |
| pospopcnt_u16_avx2_popcnt          | 2442.47     | **2689.55**  | 1506.48 |
| pospopcnt_u16_avx2                 | **5162.67**     | 4708.57  | 1135.6  |
| pospopcnt_u16_avx2_naive_counter   | **4266.14**     | 3969.01  | 1126.54 |
| pospopcnt_u16_avx2_single          | **2182.57**     | 2086.88  | 1420    |
| pospopcnt_u16_avx2_lemire          | 2065.53     | **3011.9**   | 2272.55 |
| pospopcnt_u16_avx2_lemire2         | 3409.39     | **4837.06**  | 3449.72 |
| pospopcnt_u16_avx2_mula            | 5083.96     | **5519.11**  | 3847.79 |
| pospopcnt_u16_avx2_mula_unroll4    | 6606.68     | **6840.79**  | 5499.85 |
| pospopcnt_u16_avx2_mula_unroll8    | **7276.9**      | 6907.68  | 6445.92 |
| pospopcnt_u16_avx2_mula_unroll16   | **7614.47**     | 6592.75  | 6005.51 |
| pospopcnt_u16_avx2_mula3           | **10901.6**     | 9316.41  | 9295.07 |
| pospopcnt_u16_avx2_csa             | 14196.9     | 11280.1  | **15494.3** |
| pospopcnt_u16_avx512               | **3929.03**     | 3475.55  | 0       |
| pospopcnt_u16_avx512_popcnt32_mask | **6586.83**     | 5460.8   | 0       |
| pospopcnt_u16_avx512_popcnt64_mask | **6739.99**     | 5587.01  | 0       |
| pospopcnt_u16_avx512_popcnt        | **3504.8**      | 3078.9   | 0       |
| pospopcnt_u16_avx512_mula          | 6565.97     | **6930.77 ** | 0       |
| pospopcnt_u16_avx512_mula_unroll4  | **8055.02**     | 7151.4   | 0       |
| pospopcnt_u16_avx512_mula_unroll8  | **9560.17**     | 7238.51  | 0       |
| pospopcnt_u16_avx512_mula2         | **9388.87**     | 8012.72  | 0       |
| pospopcnt_u16_avx512_mula3         | **11355.3**     | 8983.79  | 0       |
| pospopcnt_u16_avx512_csa           | **16437**       | 12441.1  | 0       |

## Instrumented tests (Linux specific)

If you are running Linux, you can run tests that take advantage of Performance Counters for Linux (PCL). This allows for programmatic discovery and enumeration of all counters and events. Compile this test suite with: `make instrumented_benchmark`. Running the output executable requires root (sudo) access to the host machine. Pass the `-v` (verbose) flag to get a detailed report of performance counters (example output from the Cannon Lake machine):

```bash
$ make instrumented_benchmark 
$ sudo ./instrumented_benchmark -v
n = 10000000 
pospopcnt_u16_scalar_naive_nosimd       	instructions per cycle 3.82, cycles per 16-bit word:  17.524, instructions per 16-bit word 67.002 
min:   675172 cycles,  2581464 instructions, 	       1 branch mis.,        4 cache ref.,        0 cache mis.
avg: 681460.1 cycles, 2581464.2 instructions, 	     2.2 branch mis.,     41.2 cache ref.,      0.6 cache mis.

pospopcnt_u16_scalar_naive              	instructions per cycle 1.99, cycles per 16-bit word:  2.059, instructions per 16-bit word 4.104 
min:    79312 cycles,   158101 instructions, 	       1 branch mis.,       83 cache ref.,        0 cache mis.
avg:  79454.0 cycles, 158101.0 instructions, 	     2.0 branch mis.,    133.4 cache ref.,      0.0 cache mis.

pospopcnt_u16_scalar_partition          	instructions per cycle 2.60, cycles per 16-bit word:  3.090, instructions per 16-bit word 8.038 
min:   119056 cycles,   309690 instructions, 	       1 branch mis.,        4 cache ref.,        0 cache mis.
avg: 120536.1 cycles, 309690.0 instructions, 	     1.2 branch mis.,     31.8 cache ref.,      0.3 cache mis.

pospopcnt_u16_hist1x4                   	instructions per cycle 2.03, cycles per 16-bit word:  2.851, instructions per 16-bit word 5.788 
min:   109856 cycles,   223008 instructions, 	       2 branch mis.,       10 cache ref.,        0 cache mis.
avg: 111365.9 cycles, 223008.0 instructions, 	     2.4 branch mis.,     61.1 cache ref.,      0.3 cache mis.

pospopcnt_u16_sse_single                	instructions per cycle 2.73, cycles per 16-bit word:  3.616, instructions per 16-bit word 9.874 
min:   139336 cycles,   380410 instructions, 	       3 branch mis.,      129 cache ref.,        0 cache mis.
avg: 139815.0 cycles, 380410.0 instructions, 	     5.8 branch mis.,    206.7 cache ref.,      0.8 cache mis.

pospopcnt_u16_sse_mula                  	instructions per cycle 3.00, cycles per 16-bit word:  2.084, instructions per 16-bit word 6.252 
min:    80282 cycles,   240888 instructions, 	       1 branch mis.,        7 cache ref.,        0 cache mis.
avg:  80492.4 cycles, 240888.0 instructions, 	     2.1 branch mis.,     29.6 cache ref.,      0.4 cache mis.

pospopcnt_u16_sse_mula_unroll4          	instructions per cycle 3.35, cycles per 16-bit word:  1.577, instructions per 16-bit word 5.284 
min:    60757 cycles,   203573 instructions, 	       2 branch mis.,       29 cache ref.,        0 cache mis.
avg:  61098.4 cycles, 203573.0 instructions, 	     2.3 branch mis.,     64.2 cache ref.,      0.2 cache mis.

pospopcnt_u16_sse_mula_unroll8          	instructions per cycle 4.08, cycles per 16-bit word:  1.433, instructions per 16-bit word 5.846 
min:    55194 cycles,   225221 instructions, 	       2 branch mis.,       12 cache ref.,        0 cache mis.
avg:  55304.1 cycles, 225221.0 instructions, 	     2.2 branch mis.,     35.0 cache ref.,      0.4 cache mis.

pospopcnt_u16_sse_mula_unroll16         	instructions per cycle 3.62, cycles per 16-bit word:  1.385, instructions per 16-bit word 5.018 
min:    53377 cycles,   193332 instructions, 	       2 branch mis.,       45 cache ref.,        0 cache mis.
avg:  53743.4 cycles, 193332.0 instructions, 	     2.4 branch mis.,     93.8 cache ref.,      0.5 cache mis.

pospopcnt_u16_sse_sad                   	instructions per cycle 3.08, cycles per 16-bit word:  0.985, instructions per 16-bit word 3.033 
min:    37943 cycles,   116844 instructions, 	       1 branch mis.,       65 cache ref.,        0 cache mis.
avg:  38069.8 cycles, 116844.0 instructions, 	     2.1 branch mis.,    100.1 cache ref.,      0.1 cache mis.

pospopcnt_u16_avx2_popcnt               	instructions per cycle 2.56, cycles per 16-bit word:  2.379, instructions per 16-bit word 6.100 
min:    91645 cycles,   235038 instructions, 	       1 branch mis.,      154 cache ref.,        0 cache mis.
avg:  92870.1 cycles, 235038.0 instructions, 	     4.2 branch mis.,    243.9 cache ref.,      2.5 cache mis.

pospopcnt_u16_avx2                      	instructions per cycle 2.01, cycles per 16-bit word:  2.038, instructions per 16-bit word 4.105 
min:    78506 cycles,   158140 instructions, 	       2 branch mis.,      131 cache ref.,        0 cache mis.
avg:  78592.1 cycles, 158140.0 instructions, 	     2.3 branch mis.,    197.1 cache ref.,      0.8 cache mis.

pospopcnt_u16_avx2_naive_counter        	instructions per cycle 2.02, cycles per 16-bit word:  2.034, instructions per 16-bit word 4.104 
min:    78360 cycles,   158124 instructions, 	       1 branch mis.,      102 cache ref.,        0 cache mis.
avg:  78479.2 cycles, 158124.0 instructions, 	     2.3 branch mis.,    162.3 cache ref.,      0.8 cache mis.

pospopcnt_u16_avx2_single               	instructions per cycle 2.02, cycles per 16-bit word:  2.029, instructions per 16-bit word 4.104 
min:    78191 cycles,   158113 instructions, 	       1 branch mis.,      125 cache ref.,        0 cache mis.
avg:  78324.6 cycles, 158113.0 instructions, 	     2.3 branch mis.,    179.0 cache ref.,      0.6 cache mis.

pospopcnt_u16_avx2_lemire               	instructions per cycle 2.80, cycles per 16-bit word:  2.861, instructions per 16-bit word 8.006 
min:   110245 cycles,   308436 instructions, 	       1 branch mis.,       44 cache ref.,        0 cache mis.
avg: 110818.9 cycles, 308436.0 instructions, 	     1.1 branch mis.,    103.3 cache ref.,      0.4 cache mis.

pospopcnt_u16_avx2_lemire2              	instructions per cycle 3.17, cycles per 16-bit word:  1.697, instructions per 16-bit word 5.382 
min:    65395 cycles,   207368 instructions, 	       1 branch mis.,        8 cache ref.,        0 cache mis.
avg:  66261.5 cycles, 207368.0 instructions, 	     1.1 branch mis.,     42.6 cache ref.,      0.6 cache mis.

pospopcnt_u16_avx2_mula                 	instructions per cycle 2.89, cycles per 16-bit word:  1.104, instructions per 16-bit word 3.190 
min:    42544 cycles,   122897 instructions, 	       2 branch mis.,       10 cache ref.,        0 cache mis.
avg:  42804.7 cycles, 122897.0 instructions, 	     3.1 branch mis.,     40.5 cache ref.,      0.4 cache mis.

pospopcnt_u16_avx2_mula2                	instructions per cycle 2.94, cycles per 16-bit word:  1.454, instructions per 16-bit word 4.269 
min:    56002 cycles,   164472 instructions, 	       3 branch mis.,        8 cache ref.,        0 cache mis.
avg:  56256.0 cycles, 164472.0 instructions, 	     3.3 branch mis.,     43.8 cache ref.,      0.4 cache mis.

pospopcnt_u16_avx2_mula_unroll4         	instructions per cycle 3.19, cycles per 16-bit word:  0.847, instructions per 16-bit word 2.706 
min:    32631 cycles,   104244 instructions, 	       1 branch mis.,       10 cache ref.,        0 cache mis.
avg:  32779.7 cycles, 104244.0 instructions, 	     1.9 branch mis.,     38.2 cache ref.,      0.3 cache mis.

pospopcnt_u16_avx2_mula_unroll8         	instructions per cycle 3.95, cycles per 16-bit word:  0.756, instructions per 16-bit word 2.986 
min:    29123 cycles,   115056 instructions, 	       2 branch mis.,       10 cache ref.,        0 cache mis.
avg:  29201.8 cycles, 115056.0 instructions, 	     2.2 branch mis.,     41.4 cache ref.,      0.5 cache mis.

pospopcnt_u16_avx2_mula_unroll16        	instructions per cycle 3.49, cycles per 16-bit word:  0.738, instructions per 16-bit word 2.574 
min:    28416 cycles,    99177 instructions, 	       3 branch mis.,       31 cache ref.,        0 cache mis.
avg:  28535.7 cycles,  99177.0 instructions, 	     3.6 branch mis.,     84.7 cache ref.,      0.5 cache mis.

pospopcnt_u16_avx512                    	instructions per cycle 2.07, cycles per 16-bit word:  1.511, instructions per 16-bit word 3.133 
min:    58218 cycles,   120696 instructions, 	       1 branch mis.,       30 cache ref.,        0 cache mis.
avg:  58299.8 cycles, 120696.0 instructions, 	     2.1 branch mis.,     80.8 cache ref.,      0.6 cache mis.

pospopcnt_u16_avx512_popcnt32_mask      	instructions per cycle 3.21, cycles per 16-bit word:  0.818, instructions per 16-bit word 2.629 
min:    31534 cycles,   101304 instructions, 	       1 branch mis.,        9 cache ref.,        0 cache mis.
avg:  31807.5 cycles, 101304.0 instructions, 	     2.1 branch mis.,     47.0 cache ref.,      0.3 cache mis.

pospopcnt_u16_avx512_popcnt64_mask      	instructions per cycle 3.10, cycles per 16-bit word:  0.835, instructions per 16-bit word 2.593 
min:    32189 cycles,    99912 instructions, 	       1 branch mis.,       60 cache ref.,        0 cache mis.
avg:  32391.7 cycles,  99912.0 instructions, 	     2.6 branch mis.,     93.8 cache ref.,      0.7 cache mis.

pospopcnt_u16_avx512_popcnt             	instructions per cycle 1.96, cycles per 16-bit word:  1.676, instructions per 16-bit word 3.287 
min:    64590 cycles,   126635 instructions, 	       1 branch mis.,       90 cache ref.,        0 cache mis.
avg:  64755.2 cycles, 126635.0 instructions, 	     2.3 branch mis.,    169.1 cache ref.,      1.3 cache mis.

pospopcnt_u16_avx512_mula               	instructions per cycle 2.41, cycles per 16-bit word:  0.752, instructions per 16-bit word 1.815 
min:    28975 cycles,    69920 instructions, 	       1 branch mis.,       17 cache ref.,        0 cache mis.
avg:  29182.1 cycles,  69920.0 instructions, 	     1.7 branch mis.,     50.0 cache ref.,      0.2 cache mis.

pospopcnt_u16_avx512_mula_unroll4       	instructions per cycle 2.52, cycles per 16-bit word:  0.624, instructions per 16-bit word 1.573 
min:    24054 cycles,    60599 instructions, 	       1 branch mis.,       28 cache ref.,        0 cache mis.
avg:  24189.0 cycles,  60599.0 instructions, 	     1.3 branch mis.,     69.2 cache ref.,      0.4 cache mis.

pospopcnt_u16_avx512_mula_unroll8       	instructions per cycle 3.08, cycles per 16-bit word:  0.556, instructions per 16-bit word 1.713 
min:    21410 cycles,    66003 instructions, 	       1 branch mis.,       16 cache ref.,        0 cache mis.
avg:  21472.3 cycles,  66003.0 instructions, 	     1.9 branch mis.,     47.8 cache ref.,      0.5 cache mis.

pospopcnt_u16_avx2_mula3                	instructions per cycle 3.02, cycles per 16-bit word:  0.436, instructions per 16-bit word 1.318 
min:    16804 cycles,    50796 instructions, 	       1 branch mis.,       49 cache ref.,        0 cache mis.
avg:  17025.3 cycles,  50796.0 instructions, 	     3.0 branch mis.,    135.1 cache ref.,      1.1 cache mis.

pospopcnt_u16_avx512_mula3              	instructions per cycle 2.20, cycles per 16-bit word:  0.291, instructions per 16-bit word 0.640 
min:    11194 cycles,    24644 instructions, 	       0 branch mis.,       76 cache ref.,        0 cache mis.
avg:  11295.2 cycles,  24644.0 instructions, 	     0.2 branch mis.,    147.0 cache ref.,      1.5 cache mis.

pospopcnt_u16_avx2_csa                  	instructions per cycle 2.99, cycles per 16-bit word:  0.256, instructions per 16-bit word 0.765 
min:     9854 cycles,    29492 instructions, 	       1 branch mis.,      193 cache ref.,        0 cache mis.
avg:  10114.0 cycles,  29492.0 instructions, 	     1.3 branch mis.,    288.8 cache ref.,      1.9 cache mis.

pospopcnt_u16_avx512_csa                	instructions per cycle 2.09, cycles per 16-bit word:  0.168, instructions per 16-bit word 0.350 
min:     6463 cycles,    13496 instructions, 	       0 branch mis.,      342 cache ref.,        0 cache mis.
avg:   6656.0 cycles,  13496.0 instructions, 	     1.0 branch mis.,    434.4 cache ref.,      2.3 cache mis.

pospopcnt_u16_avx512_mula2              	instructions per cycle 1.99, cycles per 16-bit word:  0.517, instructions per 16-bit word 1.029 
min:    19917 cycles,    39640 instructions, 	       1 branch mis.,       94 cache ref.,        0 cache mis.
avg:  19988.3 cycles,  39640.0 instructions, 	     1.4 branch mis.,    166.3 cache ref.,      0.2 cache mis.
```
