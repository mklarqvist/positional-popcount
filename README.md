# FastFlagStats

These functions compute SAM FLAG statistics using fast [SIMD instructions](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions). These functions can be applied to any packed [1-hot](https://en.wikipedia.org/wiki/One-hot) 16-bit primitive, for example in machine learning/deep learning. Using large registers (AVX-512), we can achieve ~5.6 GB/s throughput (3 billion 1-hot vectors / second).

Compile test suite with: `make` and run `./fast_flag_stats`

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
  - [Results](#results)
    - [Xeon Skylake (AVX-512)](#xeon-skylake-avx-512)
    - [Xeon Haswell (AVX-256)](#xeon-haswell-avx-256)
  - [Reference systems information](#reference-systems-information)
  
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
const __m256i* data_vectors = reinterpret_cast<const __m256i*>(data);

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

### Approach 3: Register accumulator and aggregator (AVX2)

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
const __m256i* data_vectors = reinterpret_cast<const __m256i*>(data);
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

const __m512i* data_vectors = reinterpret_cast<const __m512i*>(data);
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
    uint32_t* v = reinterpret_cast<uint32_t*>(&counters[i]);
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
const __m256i* data_vectors = reinterpret_cast<const __m256i*>(data);
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
const __m128i* data_vectors = reinterpret_cast<const __m128i*>(data);
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

const __m512i* data_vectors = reinterpret_cast<const __m512i*>(data);
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

const __m512i* data_vectors = reinterpret_cast<const __m512i*>(data);
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

const __m512i* data_vectors = reinterpret_cast<const __m512i*>(data);
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
    uint32_t* v = reinterpret_cast<uint32_t*>(&counters[i]);
    for(int j = 0; j < 16; ++j)
        out_counters[i] += v[j];
}
```

### Results

We simulated 100 million FLAG fields using a uniform distrubtion U(min,max) with the arguments [{1,8},{1,16},{1,64},{1,256},{1,512},{1,1024},{1,4096},{1,65536}] for 20 repetitions using a single core. Numbers represent the average throughput in MB/s (1 MB = 1024b). 

#### Xeon Skylake (AVX-512)

The reference system uses a Intel Skylake @ 2.6 GHz.

| Method                | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|-----------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Scalar                | 2760.23 | 2761.38 | 2773.69 | 2741.66 | 2823.47 | 2724.87 | 2767.36 | 2732.9  |
| Byte-partition      | 1343.91 | 1346.64 | 1376.01 | 1392.99 | 1832.47 | 1963.46 | 1974.73 | 2007.54 |
| Byte-partition 1x4               | 1223.41 | 1265.07 | 1299.47 | 1307.85 | 2069.54 | 2141.29 | 2175.91 | 2161.69 |
| SSE-4.1 interlaced pack-popcnt           | 1502.92 | 1506.68 | 1516.68 | 1499.56 | 1515.12 | 1517.95 | 1512.9  | 1510.84 |
| AVX-2 accumulator                 | 4196.06 | 4180.18 | 4208.02 | 4218.07 | 4229.95 | 4201.68 | 4178.67 | 4167.9  |
| AVX-2 pack-popcnt        | 4364.42 | 4351.03 | 4354.08 | 4350.36 | 4383.91 | 4358.26 | 4361.08 | 4320.24 |
| AVX-2 interlaced pack-popcnt          | 2299.15 | 2289.23 | 2294.31 | 2278.04 | 2306.02 | 2304.01 | 2292.18 | 2295.98 |
| AVX-2 accumulator naïve           | 4219.21 | 4202.06 | 4211.51 | 4191.71 | 4235.51 | 4234.23 | 4217.96 | 4208.33 |
| AVX-512 pack-popcnt        | 3487.53 | 3467.2  | 3493.57 | 3489.54 | 3480.4  | 3556.69 | 3480.66 | 3520.44 |
| AVX-512 popcnt32 mask | 5009.68 | 4975.99 | 5069.15 | 5096.2  | 5053.76 | 5066.13 | 5010.83 | 5043.13 |
| AVX-512 popcnt64 mask | **5554.44** | **5573.42** | **5626.61** | **5564.12** | **5614.47** | **5613.98** | **5604.84** | **5632.98** |
| AVX-512 shift-add accumulator              | 3926.99 | 3983.02 | 3995.27 | 3951.7  | 3978.21 | 3993.42 | 3985.31 | 3967.86 |

The AVX-512-implementation of the partial-sum accumulator algorithm (approach 5, 64-bit mask) is >2-fold faster then auto-vectorization. Unexpectedly, all the SIMD algorithms have a uniform performance 
profile indepedent of data entropy. We achieve an average throughput rate of ~2.9 billion FLAG values / second when AVX-512 is available.

#### Xeon Haswell (AVX-256)

The reference system uses a Intel Haswell @ 2.8 GHz.

| Method                | [1,8]       | [1,16]      | [1,64]      | [1,256]     | [1,512]     | [1,1024]    | [1,4096]    | [1,65536]   |
|-----------------------|---------|---------|---------|---------|---------|---------|---------|---------|
| Scalar                      | 1224.05 | 1261.06 | 1273.68 | 1254.79 | 1227.9  | 1205.23 | 1257.53 | 1230.78 |
| Byte-partition              | 712.347 | 733.356 | 724.289 | 749.943 | 1250.84 | 1326.97 | 1352.35 | 1355.92 |
| Byte-partition 1x4          | 789.993 | 831.818 | 832.912 | 852.281 | 1375.49 | 1425.63 | 1511.18 | 1504.61 |
| SSE-4.1 interlaced pack-popcnt | 1103.78 | 1169.42 | 1184.13 | 1172.35 | 1140.6  | 1094.57 | 1152.21 | 1125.54 |
| AVX-2 accumulator           | **3226.72** | **3330.12** | **3368.8**  | **3351.59** | **3281.49** | 3149.27 | **3313.62** | 3130.18 |
| AVX-2 pack-popcnt           | 2346.7  | 2414.26 | 2454.94 | 2438.04 | 2379.12 | 2354.46 | 2392.86 | 2357.76 |
| AVX-2 interlaced pack-popcnt  | 1384.21 | 1433.15 | 1462.52 | 1446.17 | 1406.57 | 1395.35 | 1425.04 | 1403.56 |
| AVX-2 accumulator naïve     | 3178.91 | 3295.09 | 3356.51 | 3334.89 | 3220.32 | **3169.8**  | 3251.75 | **3196.52** |

The AVX-256 accumulator (approach 3) is >2.6-fold faster then auto-vectorization. Unexpectedly, all the SIMD algorithms have a uniform performance 
profile indepedent of data entropy. We achieve an average throughput rate of ~1.7 billion FLAG values / second when AVX-256 is available.

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