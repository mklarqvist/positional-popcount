# FastFlagStats

These functions compute the novel "positional [population count](https://en.wikipedia.org/wiki/Hamming_weight)" (`pospopcnt`) statistics using fast [SIMD instructions](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions). Given a stream of k-bit words, we seek to count the number of set bits in positions 0, 1, 2, ..., k-1. This problem is a generalization of the population-count problem where we count the sum total of set bits in a k-bit word

These functions can be applied to any packed [1-hot](https://en.wikipedia.org/wiki/One-hot) 16-bit primitive, for example in machine learning/deep learning. Using large registers (AVX-512), we can achieve ~7 GB/s (~0.7 CPU cycles / int) throughput (3.7 billion 16-bit integers / second or 58 billion one-hot vectors / second).

### Usage

Compile the test suite with: `make` and run `./fast_flag_stats`. The test suite require `c++11` whereas the example and functions require only `c99`. If you run the test suite you **must** set the `MHZ` argument in `main.cpp` to the speed of your processor! For more detailed test, see [Instrumented tests (Linux specific)](#instrumented-tests-linux-specific).

Include `fast_flagstats.h` and `fast_flagstats.c` in your project. Then use the wrapper function for `pospopcnt`:
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
    - [Intel Xeon Skylake (AVX-512)](#intel-xeon-skylake-avx-512)
    - [Intel Xeon Haswell (AVX-256)](#intel-xeon-haswell-avx-256)
    - [Intel Ivy Bridge (AVX)](#intel-ivy-bridge-avx)
  - [Reference systems information](#reference-systems-information)
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

We simulated 100 million FLAG fields using a uniform distrubtion U(min,max) with the arguments [{1,8},{1,16},{1,64},{1,256},{1,512},{1,1024},{1,4096},{1,65536}] for 20 repetitions using a single core. Numbers represent the average throughput in MB/s (1 MB = 1024b).

#### Intel Xeon Skylake (AVX-512)

The reference system uses a Intel Xeon Skylake CPU @ 2.6 GHz. Throughput in MB/s (higher is better):

| Method                | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|----------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Scalar naïve         | 2805.65 | 2756.28 | 2806.71 | 2808.16 | 2786.45 | 2808.72 | 2804.58 | 2792.36 |
| Scalar partition     | 1308.84 | 1311.98 | 1402.56 | 1435    | 1797.03 | 1912.25 | 1917.16 | 1962.24 |
| Hist1x4              | 1264.73 | 1307.86 | 1357.43 | 1367.89 | 2014.17 | 2135.63 | 2109.54 | 2170.56 |
| SSE-4.1 interlaced pack-popcnt           | 1429.9  | 1402.51 | 1430.28 | 1402.59 | 1398.18 | 1420.15 | 1423.24 | 1400.03 |
| SSE-4.1 Muła             | 2661.49 | 2612.78 | 2670.75 | 2620.91 | 2633.46 | 2639    | 2659.3  | 2634.51 |
| SSE-4.1 Muła unrolled-4            | 3704.9  | 3735.7  | 3701.21 | 3658.3  | 3686.89 | 3659.84 | 3658.52 | 3650.64 |
| SSE-4.1 Muła unrolled-8            | 3918.73 | 3917.98 | 3890.27 | 3891.26 | 3891.61 | 3859.21 | 3885.87 | 3892.99 |
| SSE-4.1 Muła unrolled-16           | 3932.16 | 4008.64 | 3979.87 | 3937.13 | 3920.65 | 3928.13 | 3941.98 | 3953.28 |
| AVX-2 accumulator                 | 3619.13 | 3643.25 | 3632.33 | 3543.53 | 3636.62 | 3621.6  | 3595.37 | 3605.44 |
| AVX-2 pack-popcnt          | 2379.88 | 2395.88 | 2389.42 | 2386.99 | 2379.05 | 2398.23 | 2388.58 | 2391.92 |
| AVX-2 interlaced pack-popcnt          | 1958.68 | 1935.88 | 1956.41 | 1942.11 | 1933.05 | 1947.29 | 1950.01 | 1934.35 |
| AVX-2 accumulator naïve           | 3619.34 | 3643.3  | 3637.36 | 3525.18 | 3641.28 | 3597.78 | 3592.95 | 3618.17 |
| AVX-2 Lemire          | 4257.26 | 4252.58 | 4258.34 | 4197.8  | 4242.69 | 4200.67 | 4179.82 | 4221.69 |
| AVX-2 Lemire2         | 4436.06 | 4432.91 | 4413.86 | 4418.15 | 4443.94 | 4403.71 | 4375.1  | 4404.94 |
| AVX-2 Muła            | 4357.32 | 4457.61 | 4428.08 | 4390.74 | 4419.83 | 4407.5  | 4370.61 | 4413.22 |
| AVX-2 Muła unrolled-4           | 5439.47 | 5805.19 | 5610.29 | 5519.51 | 5721.24 | 5525.97 | 5515.82 | 5641.12 |
| AVX-2 Muła unrolled-8           | 5783.56 | 6012.93 | 5734.33 | 5623.83 | 5910.72 | 5754.99 | 5712.83 | 5840.26 |
| AVX-2 Muła unrolled-16          | 5656.18 | 5826.65 | 5647.29 | 5614.13 | 5764.72 | 5572.66 | 5549.68 | 5682.57 |
| AVX-512 pack-popcnt        | 3329.5  | 3313.45 | 3330.31 | 3322.2  | 3279.32 | 3309.86 | 3308.45 | 3342.31 |
| AVX-512 popcnt32 mask | 4721.69 | 4785.4  | 4588.68 | 4693.69 | 4737.47 | 4597.93 | 4650.84 | 4733.53 |
| AVX-512 popcnt64 mask               | 4812.84 | 4965.19 | 4821.19 | 4791.98 | 4955.43 | 4752.94 | 4754.39 | 4876.91 |
| AVX-512 shift-add accumulator | 3733.21 | 3727.23 | 3719.81 | 3683.83 | 3725.68 | 3679.73 | 3628.22 | 3699.64 |
| AVX-512 Muła          | 5378.98 | 5469.05 | 5234.17 | 5495.86 | 5532.67 | 5479.31 | 5222.89 | 5269.21 |
| AVX-512 Muła unrolled-4         | 6755.07 | 6745.95 | 6412.74 | 6829.13 | 6825.1  | 6717.08 | 6404.52 | 6507.1  |
| AVX-512 Muła unrolled-8         | **6834.59** | **6793.59** | **6528.55** | **6916.25** | **6866.25** | **6736.37** | **6473.77** | **6646.93** |

Workload in CPU cycles / integer (lower is better):

| Method                | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|----------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| Scalar naïve         | 1.76754  | 1.7992   | 1.76687  | 1.76596  | 1.77972  | 1.76561  | 1.76822  | 1.77595  |
| Scalar partition     | 3.78894  | 3.77987  | 3.53576  | 3.45582  | 2.75962  | 2.59333  | 2.58669  | 2.52727  |
| Hist1x4              | 3.92108  | 3.79176  | 3.65331  | 3.62537  | 2.46211  | 2.32208  | 2.3508   | 2.28472  |
| SSE-4.1 interlaced pack-popcnt           | 3.46815  | 3.53588  | 3.46723  | 3.53567  | 3.54683  | 3.49195  | 3.48437  | 3.54214  |
| SSE-4.1 Muła             | 1.86328  | 1.89802  | 1.85682  | 1.89213  | 1.88312  | 1.87916  | 1.86481  | 1.88237  |
| SSE-4.1 Muła unrolled-4            | 1.33853  | 1.32749  | 1.33986  | 1.35558  | 1.34507  | 1.35501  | 1.3555   | 1.35842  |
| SSE-4.1 Muła unrolled-8            | 1.26549  | 1.26573  | 1.27475  | 1.27442  | 1.27431  | 1.28501  | 1.27619  | 1.27385  |
| SSE-4.1 Muła unrolled-16           | 1.26117  | 1.2371   | 1.24605  | 1.25957  | 1.26487  | 1.26246  | 1.25802  | 1.25443  |
| AVX-2 accumulator                 | 1.37025  | 1.36118  | 1.36527  | 1.39948  | 1.36366  | 1.36931  | 1.3793   | 1.37545  |
| AVX-2 pack-popcnt          | 2.08376  | 2.06985  | 2.07545  | 2.07756  | 2.08449  | 2.06782  | 2.07617  | 2.07327  |
| AVX-2 interlaced pack-popcnt          | 2.53186  | 2.56168  | 2.5348   | 2.55346  | 2.56544  | 2.54667  | 2.54312  | 2.5637   |
| AVX-2 accumulator naïve           | 1.37017  | 1.36116  | 1.36338  | 1.40677  | 1.36191  | 1.37838  | 1.38023  | 1.37061  |
| AVX-2 Lemire          | 1.16486  | 1.16614  | 1.16456  | 1.18136  | 1.16886  | 1.18055  | 1.18644  | 1.17467  |
| AVX-2 Lemire2         | 1.11791  | 1.1187   | 1.12353  | 1.12244  | 1.11593  | 1.12612  | 1.13348  | 1.12581  |
| AVX-2 Muła            | 1.13811  | 1.1125   | 1.11992  | 1.12945  | 1.12201  | 1.12515  | 1.13465  | 1.12369  |
| AVX-2 Muła unrolled-4           | 0.91169  | 0.854253 | 0.88393  | 0.898469 | 0.866788 | 0.897419 | 0.89907  | 0.879099 |
| AVX-2 Muła unrolled-8           | 0.857449 | 0.824741 | 0.864809 | 0.881803 | 0.839002 | 0.861705 | 0.868065 | 0.849124 |
| AVX-2 Muła unrolled-16          | 0.876759 | 0.851107 | 0.87814  | 0.883327 | 0.860252 | 0.889899 | 0.893584 | 0.872687 |
| AVX-512 pack-popcnt        | 1.48944  | 1.49666  | 1.48908  | 1.49272  | 1.51224  | 1.49828  | 1.49892  | 1.48373  |
| AVX-512 popcnt32 mask | 1.05028  | 1.0363   | 1.08073  | 1.05655  | 1.04678  | 1.07855  | 1.06628  | 1.04765  |
| AVX-512 popcnt64 mask               | 1.03039  | 0.998774 | 1.02861  | 1.03488  | 1.00074  | 1.04338  | 1.04306  | 1.01685  |
| AVX-512 shift-add accumulator | 1.32838  | 1.33051  | 1.33316  | 1.34618  | 1.33106  | 1.34768  | 1.36681  | 1.34043  |
| AVX-512 Muła          | 0.921942 | 0.906758 | 0.947448 | 0.902335 | 0.896332 | 0.90506  | 0.949494 | 0.941148 |
| AVX-512 Muła unrolled-4         | 0.734131 | 0.735124 | 0.773321 | 0.72617  | 0.726599 | 0.738283 | 0.774314 | 0.762107 |
| AVX-512 Muła unrolled-8         | **0.72559**  | **0.729968** | **0.759603** | **0.717023** | **0.722244** | **0.736169** | **0.76603**  | **0.746075** |

The AVX-512 Muła unrolled-8 (approach 7) is ~2.5-fold faster then auto-vectorization. Unexpectedly, all the SIMD algorithms have a uniform performance profile indepedent of data entropy. We achieve an average throughput rate of ~3.6 billion FLAG values / second when AVX-512 is available.

#### Intel Xeon Haswell (AVX-256)

The reference system uses a Intel Xeon Haswell CPU @ 2.8 GHz. Throughput in MB/s (higher is better):

| Method                | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Scalar naïve     | 1249.64 | 1252.99 | 1217.65 | 1235.85 | 1264.64 | 1229.62 | 1268.97 | 1199.54 |
| Scalar partition | 735.76  | 760.804 | 727.24  | 772.929 | 1305.81 | 1346.54 | 1362.07 | 1427.51 |
| Hist1x4          | 819.808 | 838.266 | 831.704 | 845.315 | 1403.94 | 1462.8  | 1535.78 | 1539.54 |
| SSE-4.1 interlaced pack-popcnt       | 1185.53 | 1209.15 | 1191.96 | 1189.94 | 1188.57 | 1160.03 | 1188.9  | 1173.87 |
| SSE-4.1 Muła         | 2266.45 | 2335.94 | 2345.77 | 2346.45 | 2352.37 | 2283.44 | 2356.58 | 2284.79 |
| SSE-4.1 Muła unrolled-4        | 2777.19 | 2808.28 | 2760.65 | 2824.2  | 2802.04 | 2743.29 | 2769.49 | 2827.6  |
| SSE-4.1 Muła unrolled-8        | 3207.79 | 3265.11 | 3216.37 | 3312.06 | 3262.78 | 3165.2  | 3046.18 | 3222.98 |
| SSE-4.1 Muła unrolled-16       | 3315.6  | 3356.53 | 3329.12 | 3365.62 | 3341.78 | 3271.96 | 3375.42 | 3325.82 |
| AVX-2 accumulator naïve             | 3290.34 | 3352.53 | 3353.13 | 3347.02 | 3357.37 | 3194.14 | 3255.07 | 3322.7  |
| AVX-2 pack-popcnt      | 1617.35 | 1636.68 | 1625.22 | 1660.91 | 1644.12 | 1615.75 | 1657.97 | 1645.03 |
| AVX-2 interlaced pack-popcnt      | 1418.35 | 1457.56 | 1436.79 | 1465.96 | 1458.05 | 1420.54 | 1448.36 | 1452.47 |
| AVX-2 accumulator naïve       | 2398.38 | 2461.53 | 2418.32 | 2450.2  | 2437.71 | 2422.74 | 2453.36 | 2442.3  |
| AVX-2 Lemire      | 3124.77 | 3184.65 | 3202.8  | 3179.12 | 3151.64 | 3118.83 | 3186.25 | 3173.46 |
| AVX-2 Lemire2     | 3226.5  | 3234.79 | 3258.07 | 3246.26 | 3213.48 | 3190.42 | 3231.79 | 3224.92 |
| AVX-2 Muła        | 3569.23 | 3497.57 | 3486.35 | 3521.74 | 3518.61 | 3576.5  | 3560.18 | 3493.98 |
| AVX-2 Muła unrolled-4       | 5109.84 | 5206.57 | 5075.19 | 5127.82 | 5166.87 | 5083.46 | 5199.64 | 5124.15 |
| AVX-2 Muła unrolled-8       | **5897.98** | **5793.78** | **5865.32** | **5970.18** | **5610.54** | **5828.91** | **5968.62** | **5928.65** |
| AVX-2 Muła unrolled-16      | 5394.25 | 5481.15 | 5444.64 | 5537.79 | 5511.56 | 5441.22 | 5517.94 | 5520.98 |

Workload in CPU cycles / integer (lower is better):

| Method                | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| Scalar naïve     | 4.27369  | 4.26225  | 4.38595  | 4.32139  | 4.223    | 4.34328  | 4.20859  | 4.4522   |
| Scalar partition | 7.25859  | 7.01965  | 7.34362  | 6.90953  | 4.08985  | 3.96614  | 3.92093  | 3.74119  |
| Hist1x4          | 6.51442  | 6.37098  | 6.42125  | 6.31786  | 3.804    | 3.65093  | 3.47743  | 3.46894  |
| SSE-4.1 interlaced pack-popcnt       | 4.5048   | 4.41679  | 4.4805   | 4.48809  | 4.49327  | 4.60384  | 4.49202  | 4.54954  |
| SSE-4.1 Muła         | 2.35636  | 2.28626  | 2.27669  | 2.27603  | 2.27029  | 2.33883  | 2.26624  | 2.33745  |
| SSE-4.1 Muła unrolled-4        | 1.92301  | 1.90173  | 1.93453  | 1.89101  | 1.90596  | 1.94678  | 1.92836  | 1.88873  |
| SSE-4.1 Muła unrolled-8        | 1.66488  | 1.63565  | 1.66043  | 1.61246  | 1.63682  | 1.68728  | 1.7532   | 1.65703  |
| SSE-4.1 Muła unrolled-16       | 1.61074  | 1.5911   | 1.6042   | 1.5868   | 1.59813  | 1.63223  | 1.5822   | 1.60579  |
| AVX-2 accumulator naïve             | 1.62311  | 1.593    | 1.59271  | 1.59562  | 1.5907   | 1.67199  | 1.64069  | 1.6073   |
| AVX-2 pack-popcnt      | 3.30205  | 3.26305  | 3.28605  | 3.21544  | 3.24829  | 3.30532  | 3.22115  | 3.24649  |
| AVX-2 interlaced pack-popcnt      | 3.76535  | 3.66406  | 3.71702  | 3.64305  | 3.66282  | 3.75953  | 3.68731  | 3.6769   |
| AVX-2 accumulator naïve       | 2.22675  | 2.16962  | 2.20838  | 2.17965  | 2.19082  | 2.20435  | 2.17684  | 2.1867   |
| AVX-2 Lemire      | 1.70911  | 1.67697  | 1.66747  | 1.67989  | 1.69454  | 1.71237  | 1.67613  | 1.68289  |
| AVX-2 Lemire2     | 1.65522  | 1.65098  | 1.63918  | 1.64515  | 1.66193  | 1.67394  | 1.65251  | 1.65603  |
| AVX-2 Muła        | 1.49628  | 1.52694  | 1.53185  | 1.51646  | 1.51781  | 1.49324  | 1.50009  | 1.52851  |
| AVX-2 Muła unrolled-4       | 1.04516  | 1.02574  | 1.05229  | 1.04149  | 1.03362  | 1.05058  | 1.0271   | 1.04224  |
| AVX-2 Muła unrolled-8       | **0.905492** | **0.921777** | **0.910535** | **0.894541** | **0.951882** | **0.916222** | **0.894776** | **0.900808** |
| AVX-2 Muła unrolled-16      | 0.990049 | 0.974352 | 0.980888 | 0.964387 | 0.968976 | 0.981504 | 0.967856 | 0.967324 |

The AVX-2 Muła unrolled-8 accumulator (approach 7) is ~5-fold faster then auto-vectorization. Unexpectedly, all the SIMD algorithms have a uniform performance 
profile indepedent of data entropy. We achieve an average throughput rate of ~3.1 billion FLAG values / second when AVX-256 is available.

#### Intel Ivy Bridge (AVX)

The reference system is a MacBook Air (2012) using an Intel Core i5-3427U CPU @ 1.80GHz. Throughput in MB/s (higher is better):

| Method           | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Scalar naïve     | 726.794 | 815.608 | 816.682 | 839.78  | 855.888 | 848.907 | 844.767 | 854.564 |
| Scalar partition | 722.505 | 691.519 | 679.515 | 722.76  | 1184.48 | 1167.57 | 1134.61 | 1224.93 |
| Hist1x4          | 724.161 | 692.48  | 679.271 | 746.612 | 1334.28 | 1334.56 | 1350.59 | 1367.4  |
| SSE-4.1 interlaced pack-popcnt       | 1022.54 | 1003.11 | 989.33  | 1021.07 | 1057.14 | 1047.27 | 1032.17 | 1058.03 |
| SSE-4.1 Muła              | 1935.72 | 1961.71 | 1963.55 | 2071.69 | 2073.1  | 2019.7  | 1983.61 | 2093.92 |
| SSE-4.1 Muła unrolled-4   | 2439.31 | 2336.41 | 2373.36 | 2485.46 | 2500.95 | 2525.73 | 2380.38 | 2496.23 |
| SSE-4.1 Muła unrolled-8   | 2793.1  | 2686.94 | 2737.13 | 2776.02 | 2817.14 | 2785.25 | 2540.06 | 2953.62 |
| SSE-4.1 Muła unrolled-16  | **2812.62** | **2772.29** | **2894.43** | **2919.36** | **2972.27** | **2986.87** | **2853.3**  | **3057.02** |

Workload in CPU cycles / integer (lower is better):

| Method           | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Scalar naïve     | 4.7238  | 4.20941 | 4.20387 | 4.08825 | 4.0113  | 4.04429 | 4.06411 | 4.01752 |
| Scalar partition | 4.75184 | 4.96477 | 5.05247 | 4.75017 | 2.89851 | 2.94048 | 3.02591 | 2.80281 |
| Hist1x4          | 4.74097 | 4.95787 | 5.05428 | 4.59841 | 2.5731  | 2.57255 | 2.54202 | 2.51078 |
| SSE-4.1 interlaced pack-popcnt       | 3.35754 | 3.42258 | 3.47026 | 3.36239 | 3.24765 | 3.27826 | 3.32621 | 3.24494 |
| SSE-4.1 Muła               | 1.77362 | 1.75012 | 1.74848 | 1.65722 | 1.65608 | 1.69987 | 1.7308  | 1.63962 |
| SSE-4.1 Muła unrolled-4    | 1.40746 | 1.46945 | 1.44657 | 1.38133 | 1.37277 | 1.3593  | 1.4423  | 1.37537 |
| SSE-4.1 Muła unrolled-8    | 1.22918 | 1.27774 | 1.25432 | 1.23674 | 1.21869 | 1.23265 | 1.35163 | 1.16238 |
| SSE-4.1 Muła unrolled-16   | **1.22065** | **1.23841** | **1.18615** | **1.17602** | **1.15509** | **1.14944** | **1.20325** | **1.12306** |

The SSE-4.1-based interlaced pack-popcnt (approach 7) is >3.8-fold faster then auto-vectorization. We achieve an average throughput rate of ~1.6 billion FLAG values / second when SSE4.1 is available.

### Reference systems information

Intel Xeon Skylake (AVX-512)

```bash
$ lscpu
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                60
On-line CPU(s) list:   0-59
Thread(s) per core:    1
Core(s) per socket:    1
Socket(s):             60
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel Xeon Processor (Skylake, IBRS)
Stepping:              4
CPU MHz:               2599.998
BogoMIPS:              5199.99
Hypervisor vendor:     KVM
Virtualization type:   full
L1d cache:             32K
L1i cache:             32K
L2 cache:              4096K
L3 cache:              16384K
NUMA node0 CPU(s):     0-59
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology eagerfpu pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single spec_ctrl ibpb_support fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 arat
```

```bash
$ hostnamectl
    Virtualization: kvm
  Operating System: Red Hat Enterprise Linux
       CPE OS Name: cpe:/o:redhat:enterprise_linux:7.4:GA:server
            Kernel: Linux 3.10.0-693.21.1.el7.x86_64
      Architecture: x86-64
```

Intel Xeon Haswell (AVX-256)

```bash
$ lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              28
On-line CPU(s) list: 0-27
Thread(s) per core:  2
Core(s) per socket:  14
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               63
Model name:          Genuine Intel(R) CPU @ 2.20GHz
Stepping:            1
CPU MHz:             1200.167
CPU max MHz:         2800.0000
CPU min MHz:         1200.0000
BogoMIPS:            4400.30
Virtualisation:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            35840K
NUMA node0 CPU(s):   0-27
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm cpuid_fault invpcid_single pti intel_ppin tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm xsaveopt cqm_llc cqm_occup_llc dtherm ida arat pln pts
```

```bash
$ hostnamectl
  Operating System: Ubuntu 18.04.2 LTS
            Kernel: Linux 4.15.0-46-generic
      Architecture: x86-64
```

Intel Ivy Bridge (AVX)

```bash
$ sysctl -n machdep.cpu.brand_string
Intel(R) Core(TM) i5-3427U CPU @ 1.80GHz
```

```bash
$ system_profiler SPHardwareDataType
Hardware:

    Hardware Overview:

      Model Name: MacBook Air
      Model Identifier: MacBookAir5,2
      Processor Name: Intel Core i5
      Processor Speed: 1.8 GHz
      Number of Processors: 1
      Total Number of Cores: 2
      L2 Cache (per Core): 256 KB
      L3 Cache: 3 MB
      Memory: 4 GB
      Boot ROM Version: 253.0.0.0.0
      SMC Version (system): 2.5f9
```

## Instrumented tests (Linux specific)

If you are running Linux, you can run tests that take advantage of Performance Counters for Linux (PCL). This allows for programmatic discovery and enumeration of all counters and events. Compile this test suite with: `make instrumented_benchmark`. Running the output exectuable requires root (sudo) access to the host machine. Pass the `-v` (verbose) flag to get a detailed report of performance counters:

```bash
$ make instrumented_benchmark 
$ sudo ./instrumented_benchmark -v
n = 10000000 
pospopcnt_u16_scalar_naive                       all tests ok.
min:   116160 cycles,   313320 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg: 118214.9 cycles, 313320.0 instructions,         2.9 branch mis.,      0.2 cache ref.,      0.1 cache mis.
min: instructions per cycle 2.70, cycles per 16-bit word:  3.01, instructions per 16-bit word 8.13 

pospopcnt_u16_scalar_partition                   all tests ok.
min:   116492 cycles,   310869 instructions,           5 branch mis.,        0 cache ref.,        0 cache mis.
avg: 117607.2 cycles, 310869.0 instructions,         6.3 branch mis.,      0.7 cache ref.,      0.2 cache mis.
min: instructions per cycle 2.67, cycles per 16-bit word:  3.02, instructions per 16-bit word 8.07 

pospopcnt_u16_hist1x4                            all tests ok.
min:   113291 cycles,   224184 instructions,           4 branch mis.,        0 cache ref.,        0 cache mis.
avg: 114189.4 cycles, 224184.0 instructions,         5.9 branch mis.,      0.2 cache ref.,      0.2 cache mis.
min: instructions per cycle 1.98, cycles per 16-bit word:  2.94, instructions per 16-bit word 5.82 

pospopcnt_u16_sse_single                         all tests ok.
min:   144016 cycles,   415813 instructions,           3 branch mis.,        0 cache ref.,        0 cache mis.
avg: 150550.7 cycles, 415813.1 instructions,         5.2 branch mis.,      0.5 cache ref.,      0.5 cache mis.
min: instructions per cycle 2.89, cycles per 16-bit word:  3.74, instructions per 16-bit word 10.79 

pospopcnt_u16_sse_mula                           all tests ok.
min:    62453 cycles,   226434 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  62716.5 cycles, 226434.0 instructions,         2.1 branch mis.,      0.3 cache ref.,      0.3 cache mis.
min: instructions per cycle 3.63, cycles per 16-bit word:  1.62, instructions per 16-bit word 5.88 

pospopcnt_u16_sse_mula_unroll4                   all tests ok.
min:    54080 cycles,   199953 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  54216.8 cycles, 199953.0 instructions,         2.1 branch mis.,      0.4 cache ref.,      0.4 cache mis.
min: instructions per cycle 3.70, cycles per 16-bit word:  1.40, instructions per 16-bit word 5.19 

pospopcnt_u16_sse_mula_unroll8                   all tests ok.
min:    52003 cycles,   210164 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  52292.1 cycles, 210164.0 instructions,         2.1 branch mis.,      0.3 cache ref.,      0.3 cache mis.
min: instructions per cycle 4.04, cycles per 16-bit word:  1.35, instructions per 16-bit word 5.45 

pospopcnt_u16_sse_mula_unroll16                  all tests ok.
min:    54264 cycles,   195124 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  54648.6 cycles, 195124.0 instructions,         2.1 branch mis.,      0.5 cache ref.,      0.5 cache mis.
min: instructions per cycle 3.60, cycles per 16-bit word:  1.41, instructions per 16-bit word 5.06 

pospopcnt_u16_avx2_popcnt                        all tests ok.
min:    93288 cycles,   246858 instructions,           3 branch mis.,        0 cache ref.,        0 cache mis.
avg:  94320.5 cycles, 246858.0 instructions,         3.7 branch mis.,      3.3 cache ref.,      2.4 cache mis.
min: instructions per cycle 2.65, cycles per 16-bit word:  2.42, instructions per 16-bit word 6.41 

pospopcnt_u16_avx2                               all tests ok.
min:   116269 cycles,   313340 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg: 116368.9 cycles, 313340.0 instructions,         2.1 branch mis.,      0.5 cache ref.,      0.5 cache mis.
min: instructions per cycle 2.69, cycles per 16-bit word:  3.02, instructions per 16-bit word 8.13 

pospopcnt_u16_avx2_naive_counter                 all tests ok.
min:   116549 cycles,   313342 instructions,           3 branch mis.,        0 cache ref.,        0 cache mis.
avg: 116933.3 cycles, 313342.0 instructions,         3.1 branch mis.,      0.5 cache ref.,      0.5 cache mis.
min: instructions per cycle 2.69, cycles per 16-bit word:  3.03, instructions per 16-bit word 8.13 

pospopcnt_u16_avx2_single                        all tests ok.
min:   116408 cycles,   313352 instructions,           1 branch mis.,        0 cache ref.,        0 cache mis.
avg: 116544.5 cycles, 313352.0 instructions,         2.7 branch mis.,      0.4 cache ref.,      0.4 cache mis.
min: instructions per cycle 2.69, cycles per 16-bit word:  3.02, instructions per 16-bit word 8.13 

pospopcnt_u16_avx2_lemire                        all tests ok.
min:    77288 cycles,   231345 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  77312.9 cycles, 231345.0 instructions,         2.0 branch mis.,      0.3 cache ref.,      0.3 cache mis.
min: instructions per cycle 2.99, cycles per 16-bit word:  2.01, instructions per 16-bit word 6.00 

pospopcnt_u16_avx2_lemire2                       all tests ok.
min:    43453 cycles,   130276 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  44136.5 cycles, 130276.0 instructions,         2.1 branch mis.,      0.4 cache ref.,      0.4 cache mis.
min: instructions per cycle 3.00, cycles per 16-bit word:  1.13, instructions per 16-bit word 3.38 

pospopcnt_u16_avx2_mula                          all tests ok.
min:    54414 cycles,   113260 instructions,           1 branch mis.,        0 cache ref.,        0 cache mis.
avg:  55164.6 cycles, 113260.0 instructions,         1.8 branch mis.,      0.4 cache ref.,      0.3 cache mis.


pospopcnt_u16_avx2_mula_unroll4                  all tests ok.
min:    32856 cycles,   100023 instructions,           1 branch mis.,        0 cache ref.,        0 cache mis.
avg:  32997.0 cycles, 100023.0 instructions,         1.6 branch mis.,      0.4 cache ref.,      0.4 cache mis.
min: instructions per cycle 3.04, cycles per 16-bit word:  0.85, instructions per 16-bit word 2.60 

pospopcnt_u16_avx2_mula_unroll8                  all tests ok.
min:    28018 cycles,   105116 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  28204.2 cycles, 105116.0 instructions,         2.2 branch mis.,      0.3 cache ref.,      0.3 cache mis.
min: instructions per cycle 3.75, cycles per 16-bit word:  0.73, instructions per 16-bit word 2.73 

pospopcnt_u16_avx2_mula_unroll16                 all tests ok.
min:    28842 cycles,    97633 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  29124.6 cycles,  97633.0 instructions,         2.2 branch mis.,      0.4 cache ref.,      0.4 cache mis.
min: instructions per cycle 3.39, cycles per 16-bit word:  0.75, instructions per 16-bit word 2.53 

pospopcnt_u16_avx512                             all tests ok.
min:    61818 cycles,   120801 instructions,           1 branch mis.,        0 cache ref.,        0 cache mis.
avg:  62077.7 cycles, 120801.0 instructions,         2.7 branch mis.,      1.0 cache ref.,      0.9 cache mis.
min: instructions per cycle 1.95, cycles per 16-bit word:  1.60, instructions per 16-bit word 3.14 

pospopcnt_u16_avx512_popcnt32_mask               all tests ok.
min:    52202 cycles,   101294 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  52479.1 cycles, 101294.0 instructions,         2.1 branch mis.,      0.3 cache ref.,      0.2 cache mis.
min: instructions per cycle 1.94, cycles per 16-bit word:  1.35, instructions per 16-bit word 2.63 

pospopcnt_u16_avx512_popcnt64_mask               all tests ok.
min:    37527 cycles,   100062 instructions,           3 branch mis.,        0 cache ref.,        0 cache mis.
avg:  37802.1 cycles, 100062.0 instructions,         3.1 branch mis.,      0.5 cache ref.,      0.5 cache mis.
min: instructions per cycle 2.67, cycles per 16-bit word:  0.97, instructions per 16-bit word 2.60 

pospopcnt_u16_avx512_popcnt                      all tests ok.
min:    67169 cycles,   122452 instructions,           2 branch mis.,        0 cache ref.,        0 cache mis.
avg:  67245.6 cycles, 122452.0 instructions,         2.2 branch mis.,      1.1 cache ref.,      1.1 cache mis.
min: instructions per cycle 1.82, cycles per 16-bit word:  1.74, instructions per 16-bit word 3.18 

pospopcnt_u16_avx512_mula                        all tests ok.
min:    30083 cycles,    65702 instructions,           1 branch mis.,        0 cache ref.,        0 cache mis.
avg:  30242.8 cycles,  65702.0 instructions,         1.5 branch mis.,      0.2 cache ref.,      0.2 cache mis.
min: instructions per cycle 2.18, cycles per 16-bit word:  0.78, instructions per 16-bit word 1.71 

pospopcnt_u16_avx512_mula_unroll4                all tests ok.
min:    22311 cycles,    59088 instructions,           1 branch mis.,        0 cache ref.,        0 cache mis.
avg:  22715.2 cycles,  59088.0 instructions,         1.2 branch mis.,      0.5 cache ref.,      0.4 cache mis.
min: instructions per cycle 2.65, cycles per 16-bit word:  0.58, instructions per 16-bit word 1.53 

pospopcnt_u16_avx512_mula_unroll8                all tests ok.
min:    22467 cycles,    61643 instructions,           1 branch mis.,        0 cache ref.,        0 cache mis.
avg:  22852.7 cycles,  61643.0 instructions,         1.8 branch mis.,      0.3 cache ref.,      0.3 cache mis.
min: instructions per cycle 2.74, cycles per 16-bit word:  0.58, instructions per 16-bit word 1.60 
```
