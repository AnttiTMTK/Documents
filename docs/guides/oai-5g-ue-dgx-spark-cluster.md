# OAI 5G NR UE Stack on NVIDIA DGX Spark / ASUS Ascent GX10 Cluster

## Challenges, Porting Work, and Implementation Guide

**Target Platform:** 2x NVIDIA DGX Spark or ASUS Ascent GX10 (GB10 Grace Blackwell Superchip)
**Objective:** Real-time 5G NR UE software stack using OpenAirInterface with CUDA-accelerated PHY

---

## Table of Contents

1. [Platform Overview](#1-platform-overview)
2. [5G NR UE Real-Time Timing Requirements](#2-5g-nr-ue-real-time-timing-requirements)
3. [OAI UE Thread Architecture](#3-oai-ue-thread-architecture)
4. [SIMD Porting: AVX-512 to NEON/SVE2](#4-simd-porting-avx-512-to-neonsve2)
5. [OAI PHY Functions Requiring SIMD Porting](#5-oai-phy-functions-requiring-simd-porting)
6. [CUDA PHY Acceleration Strategy](#6-cuda-phy-acceleration-strategy)
7. [Real-Time Kernel and OS Configuration](#7-real-time-kernel-and-os-configuration)
8. [CPU Core Isolation and Thread Pinning](#8-cpu-core-isolation-and-thread-pinning)
9. [Radio Fronthaul Options](#9-radio-fronthaul-options)
10. [Dual-Node Stacking Architecture](#10-dual-node-stacking-architecture)
11. [Memory Bandwidth Analysis](#11-memory-bandwidth-analysis)
12. [Known Issues and Risks](#12-known-issues-and-risks)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [References](#14-references)

---

## 1. Platform Overview

### GB10 Grace Blackwell Superchip

| Component | Specification |
|---|---|
| **CPU** | 20-core ARM: 10x Cortex-X925 (perf, up to 4.0 GHz) + 10x Cortex-A725 (efficiency, 2.8 GHz) |
| **GPU** | NVIDIA Blackwell, 6,144 CUDA cores, 1 PFLOP FP4 sparse |
| **Memory** | 128 GB LPDDR5x unified (CPU+GPU shared), 273 GB/s |
| **Storage** | Up to 4 TB NVMe PCIe 4.0/5.0 |
| **Networking** | ConnectX-7 SmartNIC (QSFP28, up to 200 GbE) |
| **Architecture** | ARMv9.2-A, NEON (mandatory), SVE2 (128-bit implementation) |
| **OS** | DGX OS 7.4.0 (Ubuntu 24.04 LTS), Kernel 6.17 |
| **GPU Driver** | 580.126.09, CUDA 13.0.2 |
| **TDP** | 140W SoC + 100W peripherals, 240W PSU |

### Dual-Node Cluster (Stacked)

| Resource | Single Node | Dual-Node Stacked |
|---|---|---|
| CPU Cores | 20 (10 perf + 10 eff) | 40 (20 perf + 20 eff) |
| Memory | 128 GB | 256 GB |
| GPU CUDA Cores | 6,144 | 12,288 |
| AI Performance | 1 PFLOP | 2 PFLOP |
| Interconnect | -- | 200 Gbps ConnectX-7 QSFP |

### CPU Cache Hierarchy (Per Node)

| Level | Cortex-X925 | Cortex-A725 |
|---|---|---|
| L1I / L1D | 64 KB / 64 KB | 64 KB / 64 KB |
| L2 | 2 MB per core | 512 KB per core |
| L3 (Cluster 0) | 8 MB shared | 8 MB shared |
| L3 (Cluster 1) | 16 MB shared | 16 MB shared |
| SLC | 16 MB system-level cache | |

**Key observation:** Cluster 1 X925 cores have 2x L3 (16 MB) and lower L3 latency (~14 ns vs ~21 ns). PHY-critical threads should be pinned to Cluster 1.

---

## 2. 5G NR UE Real-Time Timing Requirements

### Slot Duration by Subcarrier Spacing

| SCS (kHz) | Numerology | Slots/Subframe | Slot Duration | Use Case |
|---|---|---|---|---|
| 15 | 0 | 1 | 1000 us | FR1 low-BW |
| **30** | **1** | **2** | **500 us** | **FR1 primary (target)** |
| 60 | 2 | 4 | 250 us | FR1/FR2 |
| 120 | 3 | 8 | 125 us | FR2 mmWave |

### UE Processing Budget (30 kHz SCS, 500 us slot)

```
Slot N received                    Slot N+K UL TX
    |                                    |
    v                                    v
    |------ 500 us slot budget ---------|
    |                                    |
    [FFT/OFDM demod] [Chan Est] [Equal] [LDPC decode] [MAC/RLC] [UL encode] [IFFT]
    |--- ~50 us ---|  ~30 us    ~20 us   ~120-700 us    ~20 us    ~80 us     ~50 us
```

**Critical constraint:** LDPC decoding dominates. On GB10 CPU alone at 20 iterations: **~710 us** -- exceeds the 500 us slot budget by 1.4x. GPU offload is mandatory.

### HARQ Timing

- K1 (DL HARQ feedback): UE must send ACK/NACK 4-8 slots after PDSCH reception
- K2 (UL scheduling): UE must transmit PUSCH K2 slots after receiving UL grant
- Processing time capability 1: N1 = 8 OFDM symbols for 30 kHz SCS
- Processing time capability 2: N1 = 3 OFDM symbols (requires faster processing)

---

## 3. OAI UE Thread Architecture

### Thread Map

| Thread | Function | RT Critical | Scheduling | Pin To |
|---|---|---|---|---|
| **UE_thread** | Sample acquisition, slot boundary | YES | SCHED_FIFO (max-1) | X925 core 0 |
| **UE_processing** (x3) | DL slot decode + UL encode | YES | SCHED_FIFO | X925 cores 1-3 |
| **LDPC decoder** (x8) | Parallel code block decoding | YES | SCHED_FIFO | X925 cores 4-9 |
| **Thread pool (Tpool)** | General PHY worker tasks | Moderate | SCHED_FIFO | X925 remaining |
| **ITTI threads** | NAS, RRC, MAC messaging | No | SCHED_OTHER | A725 cores |
| **NFAPI/FAPI** | MAC-PHY interface | Moderate | SCHED_OTHER | A725 cores |
| **T_TRACER** | Debug tracing (disable in RT) | No | SCHED_OTHER | A725 cores |

### Processing Pipeline Per Slot

```
1. nr_slot_indication()          -- Slot boundary trigger
2. nr_ue_pdcch_procedures()      -- PDCCH blind search, DCI decode (DL control)
3. nr_ue_pdsch_procedures()      -- PDSCH demodulation + LDPC decode (DL data)
4. nr_ue_scheduler()             -- UL scheduling decisions
5. nr_ue_ulsch_procedures()      -- PUSCH LDPC encode + modulation (UL data)
6. pucch_procedures_ue_nr()      -- PUCCH encode (UL control: ACK/NACK, CSI, SR)
```

### Inter-Thread Communication

- `resp_L1` FIFO: RX processing completion signals
- `respDecode` FIFO: LDPC decoding completion signals
- Configurable parallelism: `--nrUE-threads N` and `--Tpool <core_list>`

---

## 4. SIMD Porting: AVX-512 to NEON/SVE2

This is the most labor-intensive challenge. OAI PHY is heavily optimized with x86 SIMD intrinsics.

### 4.1 Architecture Comparison

| Feature | AVX-512 | ARM NEON | ARM SVE2 (X925) |
|---|---|---|---|
| **Register width** | 512-bit (ZMM0-31) | 128-bit (Q0-Q31) | 128-bit (scalable, but X925 implements 128-bit) |
| **Register count** | 32 x 512-bit | 32 x 128-bit | 32 x scalable + 16 predicate |
| **Throughput ratio** | 64 bytes/op | 16 bytes/op (4x narrower) | 16 bytes/op on X925 |
| **SIMD pipes (X925)** | 2 x 512-bit (typical Xeon) | 6 x 128-bit | 6 x 128-bit |
| **Effective ops/cycle** | 2 (x 512-bit = 1024 bits) | 6 (x 128-bit = 768 bits) | 6 (x 128-bit = 768 bits) |
| **FMA** | vfmadd (FP32/FP64) | FMLA (FP16/FP32/FP64) | FMLA + complex MLA |
| **Masking/Predication** | 8 opmask registers (k0-k7) | Bitwise select (VBSL) | 16 predicate registers (P0-P15) |
| **Saturating arithmetic** | Separate instructions | First-class "Q" variants | First-class + wider ops |
| **Shuffle/Permute** | Rich (vpermi2, vpshufb) | TBL/TBX (128-bit only) | TBL + SPLICE |
| **Horizontal ops** | hadd, hsub, dpbusd | Limited pairwise add | ADDV, FADDV reductions |
| **Gather/Scatter** | All element sizes | Not available | 32/64-bit elements only |

### 4.2 Key Porting Challenges

#### 4.2.1 Vector Width Gap (4:1)

AVX-512 processes 64 bytes per instruction vs NEON's 16 bytes. Each AVX-512 loop iteration becomes **4 NEON iterations**.

```c
// x86 AVX-512: Process 32 int16 elements at once
__m512i a = _mm512_load_si512(ptr);
__m512i b = _mm512_adds_epi16(a, c);

// ARM NEON equivalent: 4 iterations of 8 elements
int16x8_t a0 = vld1q_s16(ptr);
int16x8_t a1 = vld1q_s16(ptr + 8);
int16x8_t a2 = vld1q_s16(ptr + 16);
int16x8_t a3 = vld1q_s16(ptr + 24);
int16x8_t b0 = vqaddq_s16(a0, c);
int16x8_t b1 = vqaddq_s16(a1, c);
int16x8_t b2 = vqaddq_s16(a2, c);
int16x8_t b3 = vqaddq_s16(a3, c);
```

The X925's 6 SIMD pipes partially compensate: with sufficient ILP (instruction-level parallelism), effective throughput ratio narrows from 4:1 to approximately **1.3:1**.

#### 4.2.2 Shuffle/Permute Operations (Hardest to Port)

x86 has extremely rich permutation instructions that operate across the full 512-bit register. NEON TBL only works within 128 bits.

```c
// x86: Cross-lane byte shuffle across 64 bytes
__m512i result = _mm512_permutexvar_epi8(idx, data);

// ARM NEON: Must decompose into multiple 128-bit TBL operations
// with manual lane index remapping -- significantly more complex
```

**Impact on OAI:** LDPC check-node processing relies heavily on shuffles for belief propagation message routing. This is the single most difficult porting task.

#### 4.2.3 _mm_movemask_epi8 (No NEON Equivalent)

This common x86 pattern extracts MSBs of each byte into a scalar bitmask. Used extensively in LDPC and bit manipulation code.

```c
// x86: Single instruction
int mask = _mm_movemask_epi8(cmp_result);

// ARM NEON: Multi-instruction workaround
uint8x16_t msb = vshrq_n_u8(cmp_result, 7);
uint64x2_t paired = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(msb)));
uint64_t mask = vgetq_lane_u64(paired, 0) | (vgetq_lane_u64(paired, 1) << 8);
```

#### 4.2.4 Mask Register Operations

AVX-512's per-lane predication via opmask registers has no NEON equivalent. SVE2's predicate registers are the closer match but at 128-bit width, the benefit over NEON bitwise select is marginal.

#### 4.2.5 Integer Type Signedness

x86 intrinsics are sign-agnostic (`__m128i` holds both signed and unsigned). NEON types encode signedness (`int16x8_t` vs `uint16x8_t`). Mismatches cause subtle bugs in compare and shift operations.

#### 4.2.6 Memory Alignment

AVX-512 code assumes 64-byte alignment. NEON LD1 needs only element-size alignment, but LDR Qx needs 16-byte alignment. Audit all `posix_memalign`, `_mm_malloc`, `memalign` calls.

### 4.3 Available Translation Tools

| Tool | Coverage | Approach | AVX-512 Support | Recommendation |
|---|---|---|---|---|
| **SIMDe** (SIMD Everywhere) | SSE-AVX2: ~100%, AVX-512: ~33% | Header-only drop-in replacement | Partial (1688/5160 functions) | **Primary tool** -- already used by OAI |
| **sse2neon** | SSE through SSE4.2: ~100% | Header-only, SSE-to-NEON | None | Merged into SIMDe |
| **MIPP** | SSE/AVX/AVX-512/NEON/SVE | C++11 wrapper, requires rewrite | Yes | Alternative for new code |
| **xsimd** | Full x86 + NEON + SVE | C++ wrapper, requires rewrite | Yes | Alternative for new code |
| **Manual NEON/SVE2** | N/A | Hand-optimized intrinsics | N/A | **Hot-path optimization** |

### 4.4 SIMDe Coverage Gap Analysis

SIMDe is the right starting point since OAI already uses it for ARM cross-compilation. However:

- **AVX-512F (core):** Partially complete, NOT 100%
- **AVX-512BW (byte/word):** Used heavily by LDPC -- partial coverage
- **AVX-512VNNI:** 100% coverage (used for neural-net-like operations)
- **AVX-512VPOPCNTDQ:** 100% coverage
- **AVX-512BITALG:** 100% coverage

**Gap:** Any OAI code using AVX-512F/BW intrinsics not yet in SIMDe (~67% of functions) will **fail to compile** on ARM. These require manual NEON replacements.

### 4.5 NEON vs SVE2 on Cortex-X925

On the X925 (128-bit SVE2 implementation):
- **SVE2 provides NO vector-width advantage over NEON** -- both are 128-bit
- SVE2 adds useful instructions: complex multiply-add (FCMLA), histogram, match, bit deposit/extract
- SVE2 PTRUE elimination was **removed on X925** -- predicate setup adds overhead that NEON avoids
- **Recommendation:** Use NEON for bulk porting. Use SVE2 selectively for:
  - Complex multiply-accumulate (FCMLA) -- valuable for channel estimation
  - Reduction operations (ADDV, FADDV) -- useful for signal processing
  - Future-proofing for wider SVE implementations (Graviton, A64FX)

### 4.6 Expected Performance After Porting

| Workload | AVX-512 (Xeon) | NEON (X925, estimated) | Ratio |
|---|---|---|---|
| LDPC decode (per core) | ~7.5 Gbps (deep optimization) | ~2-3 Gbps | 2.5-3.5x slower |
| FFT 2048-point | Baseline | ~2-3x more cycles | 2-3x slower |
| Complex vector multiply | Baseline | ~1.5-2x more cycles | 1.5-2x slower |
| Channel estimation | Baseline | ~1.5x more cycles | 1.5x slower |
| **Overall PHY (10 X925 cores)** | **Baseline (4 Xeon cores)** | **Comparable** | **~1:1 with parallelism** |

The 10 X925 performance cores compensate for narrower SIMD through parallelism. Achievability depends on OAI's thread scalability.

---

## 5. OAI PHY Functions Requiring SIMD Porting

### 5.1 Priority Order (by CPU time impact)

#### Priority 1: LDPC Encoder/Decoder (Critical Path)

- **Location:** `openair1/PHY/CODING/nrLDPC_decoder/`, `nrLDPC_encoder/`
- **Architecture:** Dynamically loaded as `libldpc.so` via `dlopen()` in `nrLDPC_load.c`
- **Must implement:** `nrLDPC_initcall`, `nrLDPC_decod`, `nrLDPC_encod`
- **SIMD usage:** AVX2/AVX-512 for min-sum belief propagation
  - 8-bit quantized soft values (LLRs) processed in SIMD lanes
  - Saturating add/subtract for LLR updates
  - Min operations for check-node processing
  - **Heavy shuffle/permute** for message routing between variable and check nodes
- **Porting strategy:** Create `libldpc_neon.so` as separate implementation -- cleanest entry point since OAI's dlopen architecture supports it natively
- **CPU impact without GPU:** ~710 us per slot (consumes 10 of 20 cores)
- **CPU impact with GPU offload:** ~118 us per slot (even unoptimized via Sionna/TensorFlow)

#### Priority 2: DFT/IDFT (OFDM Processing)

- **Location:** `openair1/PHY/TOOLS/`
- **SIMD usage:** Custom SSE/AVX2 butterfly operations, complex multiply, twiddle factors
- **Supported sizes:** 128, 256, 512, 1024, 1536, 2048, 4096 points
- **Porting strategy:** Consider replacing custom DFT with NEON-optimized library:
  - **Ne10** (ARM's open-source math library, has optimized FFT)
  - **FFTW 3.3.1+** (has NEON support)
  - **KissFFT** (portable, good ARM performance)
- **CPU impact:** ~40% of total PHY compute in some analyses

#### Priority 3: Channel Estimation

- **Location:** `openair1/PHY/NR_ESTIMATION/`
- **SIMD usage:** Complex multiply of received pilots with conjugate of known reference signals, frequency-domain interpolation
- **Key operations:** Vectorized complex multiply-accumulate
- **Porting advantage:** NEON has native FP16 support and FMLA; SVE2's FCMLA (fused complex MLA) is valuable here

#### Priority 4: DLSCH/ULSCH Demodulation

- **Location:** `openair1/PHY/NR_TRANSPORT/`
- **Files:** `nr_dlsch_demodulation.c`, `nr_ulsch_demodulation.c`
- **SIMD usage:** MIMO detection, equalization, soft demapping
- **Key operations:** Complex vector multiply, matrix operations

#### Priority 5: Polar Encoder/Decoder (Control Channels)

- **Location:** `openair1/PHY/CODING/nrPolar_tools/`
- **Usage:** PDCCH/PBCH decoding (control channels, lower throughput requirement)
- **SIMD usage:** Moderate -- CRC-aided successive cancellation list decoder

#### Priority 6: Common DSP Utilities

- Complex vector multiplication, conjugate multiply
- Quantization, scaling, clipping
- Bit interleaving/deinterleaving
- Rate matching operations

### 5.2 SIMD Intrinsic Hotspots in OAI

| x86 Intrinsic | Usage in OAI | NEON Replacement | Difficulty |
|---|---|---|---|
| `_mm256_adds_epi16` | LDPC LLR saturating add | `vqaddq_s16` | Easy |
| `_mm256_subs_epi16` | LDPC LLR saturating sub | `vqsubq_s16` | Easy |
| `_mm256_min_epi16` | LDPC check-node min | `vminq_s16` | Easy |
| `_mm256_shuffle_epi8` | LDPC message routing | `vqtbl1q_u8` (128-bit only) | **Hard** |
| `_mm256_permutevar8x32_epi32` | Cross-lane permute | Multi-instruction decomposition | **Hard** |
| `_mm_movemask_epi8` | Bit extraction | Multi-instruction workaround | Medium |
| `_mm256_fmadd_ps` | Channel est. FMA | `vfmaq_f32` | Easy |
| `_mm256_mullo_epi16` | General multiply | `vmulq_s16` | Easy |
| `_mm256_cmpeq_epi16` | Compare operations | `vceqq_s16` | Easy (signedness caution) |
| `_mm512_permutexvar_epi8` | AVX-512 byte permute | **No direct equivalent** | **Very Hard** |
| `_mm512_mask_blend_epi16` | Masked blend | SVE2 predicated ops or VBSL | Medium |

---

## 6. CUDA PHY Acceleration Strategy

### 6.1 Why GPU Offload is Mandatory

The LDPC-on-DGX-Spark study (arXiv 2602.04652) demonstrates:

| Metric | CPU Only (Grace) | GPU (Blackwell) | Improvement |
|---|---|---|---|
| LDPC decode latency (20 iter) | ~710 us | ~118 us | **6x faster** |
| LDPC decode latency (5 iter) | ~180 us | ~30 us | **6x faster** |
| CPU cores consumed | ~10 of 20 | GPU + ~15W additional | **Frees 10 cores** |
| Slot budget usage (30 kHz SCS) | 142% (fails) | 24% (passes) | **Real-time viable** |

These results used **unoptimized Sionna/TensorFlow** layers -- hand-tuned CUDA will be significantly faster.

### 6.2 NVIDIA Aerial Components Reusable for UE

Aerial is gNB-focused but many primitives are symmetric:

| Aerial Component | gNB Function | UE Equivalent | Reusable? |
|---|---|---|---|
| cuPHY LDPC decoder | PUSCH RX decode | PDSCH RX decode | **YES** (identical algorithm) |
| cuPHY LDPC encoder | PDSCH TX encode | PUSCH TX encode | **YES** |
| cuPHY FFT/IFFT | OFDM mod/demod | OFDM mod/demod | **YES** |
| cuPHY Polar decoder | PUCCH/PRACH decode | PDCCH/PBCH decode | **YES** |
| cuPHY Polar encoder | PDCCH encode | PUCCH encode | **YES** |
| cuPHY Channel Est. | UL channel estimation | DL channel estimation | **Partially** (different RS patterns) |
| cuPHY Equalizer | UL equalization | DL equalization | **Partially** |
| cuPHY QAM demod | UL soft demapping | DL soft demapping | **YES** |
| cuPHY Beamforming | DL precoding | N/A (UE perspective) | No |

**pyAerial** (Python bindings) exposes individual cuPHY primitives for integration.

### 6.3 CUDA Integration Architecture

```
┌─────────────────────────────────────────────────────┐
│                   OAI nr-uesoftmodem                 │
│                                                     │
│  ┌──────────────────┐    ┌───────────────────────┐  │
│  │  L2/L3 (CPU)     │    │  PHY Framework (CPU)  │  │
│  │  RRC, NAS, PDCP  │    │  Slot scheduling      │  │
│  │  RLC, MAC        │    │  Thread management    │  │
│  └────────┬─────────┘    └──────────┬────────────┘  │
│           │                         │               │
│           │              ┌──────────┴────────────┐  │
│           │              │  CUDA PHY Offload     │  │
│           │              │                       │  │
│           │              │  Stream 0: LDPC dec   │  │
│           │              │  Stream 1: FFT/IFFT   │  │
│           │              │  Stream 2: Chan Est   │  │
│           │              │  Stream 3: Polar dec  │  │
│           │              │  Stream 4: LDPC enc   │  │
│           │              │                       │  │
│           │              │  ┌─────────────────┐  │  │
│           │              │  │ Blackwell GPU   │  │  │
│           │              │  │ 6,144 CUDA cores│  │  │
│           │              │  │ Unified memory  │  │  │
│           │              │  └─────────────────┘  │  │
│           │              └───────────────────────┘  │
└───────────┴─────────────────────────────────────────┘
```

### 6.4 GPU Offload Considerations

- **Unified memory advantage:** No explicit H2D/D2H copies needed -- CPU and GPU share the same 128 GB LPDDR5x. Use `cudaMallocManaged()` or direct pointer sharing.
- **Kernel launch latency:** ~5-10 us per kernel launch. Batch operations to amortize.
- **Concurrency:** Use multiple CUDA streams for parallel PHY operations within a slot.
- **Memory bandwidth contention:** CPU PHY threads and GPU kernels compete for the shared 273 GB/s. Under GPU load, CPU DRAM latency degrades from 113 ns to **351+ ns**.
- **Mitigation:** Pipeline GPU operations to minimize simultaneous CPU+GPU memory pressure peaks.

---

## 7. Real-Time Kernel and OS Configuration

### 7.1 Current DGX OS State

| Component | Current | Required |
|---|---|---|
| Kernel | 6.17-1008.8 (standard HWE) | PREEMPT_RT enabled |
| Preemption | CONFIG_PREEMPT_VOLUNTARY (likely) | CONFIG_PREEMPT_RT |
| Timer tick | Periodic | CONFIG_NO_HZ_FULL on isolated cores |
| CPU governor | Likely ondemand/schedutil | `performance` (fixed max frequency) |

### 7.2 PREEMPT_RT Kernel Options

#### Option A: Canonical Real-Time Kernel (Easiest)

```bash
# Requires Ubuntu Pro (free for up to 5 machines)
sudo pro attach <token>
sudo pro enable realtime-kernel
```

**Downside:** Based on kernel 6.8, may lose DGX Spark HWE hardware enablement from 6.17.

#### Option B: Custom RT Kernel Build (Recommended)

```bash
# 1. Get DGX Spark kernel source (6.17)
apt-get source linux-image-$(uname -r)

# 2. PREEMPT_RT is in mainline since ~6.12, enable it:
scripts/config --enable PREEMPT_RT
scripts/config --disable DEBUG_LOCKDEP
scripts/config --disable DEBUG_PREEMPT
scripts/config --disable DEBUG_OBJECTS
scripts/config --disable SLUB_DEBUG

# 3. Build
make -j$(nproc) ARCH=arm64 bindeb-pkg

# 4. Install and rebuild NVIDIA modules
sudo dpkg -i linux-image-*.deb linux-headers-*.deb
sudo dkms autoinstall  # Rebuild GPU + ConnectX-7 modules
```

**Risk:** NVIDIA kernel modules (`nvidia.ko`, `mlnx-ofed-kernel-dkms`) may need patching. Known issue with `mlnx-ofed-kernel-dkms` failing on kernel 6.14.0-1015-nvidia.

#### Option C: Stock Kernel with Aggressive Tuning (Fallback)

If custom kernel is not feasible:

```bash
# Boot parameters
GRUB_CMDLINE_LINUX="isolcpus=10-19 nohz_full=10-19 rcu_nocbs=10-19 \
    irqaffinity=0-9 processor.max_cstate=0 idle=poll \
    intel_pstate=disable nosoftlockup tsc=reliable"
```

Less deterministic than PREEMPT_RT but avoids kernel changes.

### 7.3 Essential Kernel Tuning

```bash
# CPU frequency governor -- lock to max
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu
done

# Disable frequency scaling on performance cores
echo 0 > /sys/devices/system/cpu/cpufreq/boost  # If applicable

# Memory locking for OAI process
ulimit -l unlimited

# Real-time scheduling priority
ulimit -r 99

# Disable transparent huge pages (reduces latency variance)
echo never > /sys/kernel/mm/transparent_hugepage/enabled

# Network tuning for USRP
sysctl -w net.core.rmem_max=62500000
sysctl -w net.core.wmem_max=62500000
sysctl -w net.core.rmem_default=62500000
sysctl -w net.core.wmem_default=62500000
```

---

## 8. CPU Core Isolation and Thread Pinning

### 8.1 GB10 Core Layout

```
┌─────────────────────────────────────────────────────────┐
│                      GB10 SoC                           │
│                                                         │
│  ┌─── Cluster 0 (8 MB L3) ───┐  ┌─── Cluster 1 (16 MB L3) ───┐
│  │                            │  │                              │
│  │  A725  A725  A725  A725    │  │  A725  A725  A725  A725     │
│  │  [0]   [1]   [2]   [3]    │  │  [4]   [5]   [6]   [7]     │
│  │                            │  │                              │
│  │  A725  X925  X925  X925    │  │  A725  X925  X925  X925     │
│  │  [8]   [9]   [10]  [11]   │  │  [12]  [13]  [14]  [15]    │
│  │                            │  │                              │
│  │  X925  X925                │  │  X925  X925  X925  X925     │
│  │  [16]  [17]                │  │  [18]  [19]                 │
│  └────────────────────────────┘  └──────────────────────────────┘
│                                                         │
│  Note: Actual core numbering may differ. Verify with:   │
│  lscpu --extended or /sys/devices/system/cpu/            │
└─────────────────────────────────────────────────────────┘
```

**Important:** Verify actual core-to-cluster mapping on your hardware with:
```bash
lscpu --extended
cat /sys/devices/system/cpu/cpu*/topology/cluster_id
cat /sys/devices/system/cpu/cpu*/cpu_capacity  # X925 > A725
```

### 8.2 Kernel Boot Parameters

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX="isolcpus=managed_irq,domain,10-19 \
    nohz_full=10-19 \
    rcu_nocbs=10-19 \
    irqaffinity=0-9 \
    processor.max_cstate=0 \
    idle=poll"
```

### 8.3 cgroups v2 Configuration (Preferred)

```bash
# Create RT cpuset for OAI PHY (X925 Cluster 1 cores)
mkdir -p /sys/fs/cgroup/oai_phy
echo "13-19" > /sys/fs/cgroup/oai_phy/cpuset.cpus
echo "0" > /sys/fs/cgroup/oai_phy/cpuset.mems

# Create non-RT cpuset for housekeeping (A725 cores)
mkdir -p /sys/fs/cgroup/oai_ctrl
echo "0-9" > /sys/fs/cgroup/oai_ctrl/cpuset.cpus
echo "0" > /sys/fs/cgroup/oai_ctrl/cpuset.mems

# Launch OAI UE in RT cpuset
echo $$ > /sys/fs/cgroup/oai_phy/cgroup.procs
exec taskset -c 13-19 chrt -f 90 ./nr-uesoftmodem <args>
```

### 8.4 IRQ Affinity

```bash
# Pin all IRQs to A725 efficiency cores
for irq in /proc/irq/*/smp_affinity; do
    echo "3ff" > $irq  # Cores 0-9 (A725)
done

# Exception: Pin USRP/network IRQs to specific A725 core
# Find network IRQ: cat /proc/interrupts | grep <interface>
echo "4" > /proc/irq/<usrp_irq>/smp_affinity  # Core 2
```

### 8.5 Recommended Thread-to-Core Mapping

```
X925 Cluster 1 Cores (13-19):          A725 Cores (0-9):
  Core 13: UE_thread (sample acq.)       Cores 0-3: OS, systemd, logging
  Core 14: UE_processing[0]              Cores 4-5: Network IRQs
  Core 15: UE_processing[1]              Cores 6-7: ITTI threads (NAS/RRC)
  Core 16: UE_processing[2]              Cores 8-9: NFAPI, misc
  Core 17: LDPC_thread[0-1]
  Core 18: LDPC_thread[2-3]
  Core 19: LDPC_thread[4-5]

X925 Cluster 0 Cores (10-12):
  Core 10: LDPC_thread[6-7]
  Core 11: Tpool workers
  Core 12: CUDA management / GPU sync
```

---

## 9. Radio Fronthaul Options

### 9.1 Compatible SDR Options

| SDR | Connection | Max BW | FR | Latency | DGX Spark Port | Cost |
|---|---|---|---|---|---|---|
| **USRP B210** | USB 3.0 | 30 MHz (2x2) | FR1 only | 100-200 us | USB-C | ~$2K |
| **USRP N310** | 2x SFP+ 10GbE | 100 MHz (4x4) | FR1 | 50-100 us | 10GbE RJ-45 (GX10) | ~$8K |
| **USRP N320** | 2x SFP+ 10GbE | 100 MHz (2x2) | FR1+FR2 | 50-100 us | 10GbE RJ-45 (GX10) | ~$10K |
| **USRP X410** | 2x QSFP28 100GbE | 400 MHz (4x4) | FR1+FR2 | <50 us | ConnectX-7 QSFP28 | ~$30K |

### 9.2 Recommended Setup

**For development/testing:** USRP B210 via USB-C
- Simple, affordable, sufficient for narrow-band NR (20-30 MHz)
- USB jitter is manageable at 15/30 kHz SCS

**For production real-time:** USRP X410 via ConnectX-7 QSFP28
- Lowest latency, highest bandwidth
- Native match with DGX Spark's ConnectX-7 ports
- **Tradeoff:** Uses one QSFP port, leaving one for stacking (DGX Spark has 2, GX10 has 1)
- **GX10 limitation:** Only 1 QSFP port -- cannot stack AND use X410. Use N310 via 10GbE instead.

**For balanced setup (GX10):** USRP N310 via 10GbE RJ-45
- Good bandwidth (100 MHz FR1), acceptable latency
- Uses 10GbE port, leaving QSFP free for stacking

### 9.3 Clock Synchronization

- External 10 MHz reference and PPS required for gNB-UE synchronization
- Use Ettus OctoClock-G (with GPSDO) when both gNB and UE are SDR-based
- Configure: `--clock-source external --time-source external`

---

## 10. Dual-Node Stacking Architecture

### 10.1 Interconnect

- **Link:** ConnectX-7 QSFP28, 200 Gbps aggregate
- **Latency:** ~2-5 us per hop (Ethernet frame)
- **Cable:** Amphenol NJAAKK-N911 or Luxshare LMTQF022-SD-R (0.4-0.5m DAC)

### 10.2 Recommended Workload Split

```
┌─────────── Node 1 (PHY/Real-Time) ───────┐    QSFP     ┌──── Node 2 (Upper Layers/GPU) ────┐
│                                           │◄──200Gbps──►│                                   │
│  USRP ──► PHY Processing                 │             │  Higher Layer Processing           │
│           ├─ FFT/IFFT (NEON)             │             │  ├─ High-MAC                       │
│           ├─ Channel Estimation (NEON)    │             │  ├─ RLC / PDCP / SDAP              │
│           ├─ Equalization (NEON)          │             │  ├─ RRC / NAS                      │
│           ├─ Low-MAC (X925 cores)         │             │  └─ IP stack                       │
│           └─ LDPC/Polar (local GPU)       │             │                                   │
│                                           │             │  GPU Workloads                     │
│  10 X925 cores → RT PHY threads          │             │  ├─ AI/ML inference               │
│  10 A725 cores → OS, network, misc       │             │  ├─ Additional CUDA PHY offload    │
│                                           │             │  └─ Analytics / logging            │
│  GPU: LDPC decode, FFT (primary offload) │             │                                   │
└───────────────────────────────────────────┘             └───────────────────────────────────┘
```

### 10.3 Key Rules

1. **PHY processing MUST stay on one node** -- sub-slot timing cannot tolerate inter-node latency
2. **Do NOT split PHY across nodes** -- even 2-5 us per hop breaks sub-symbol timing
3. **High-MAC and above can run on Node 2** -- these operate at slot/subframe timescale (ms)
4. **Use NCCL or MPI** for GPU-to-GPU data transfer across nodes if needed
5. **Slurm or Kubernetes** can orchestrate the two-node cluster

### 10.4 Single-Node Feasibility

For many UE configurations, a single node may be sufficient:
- 30 kHz SCS, 100 MHz BW, 2x2 MIMO: Single node with CUDA offload likely sufficient
- 30 kHz SCS, 100 MHz BW, 4x4 MIMO: May need dual-node
- 120 kHz SCS (FR2): Extremely tight -- dual-node with aggressive GPU offload

---

## 11. Memory Bandwidth Analysis

### 11.1 DGX Spark Memory Architecture

```
┌─────────────────────────────────────────────┐
│              128 GB LPDDR5x                 │
│           273 GB/s aggregate                │
│                                             │
│    ┌──────────┐        ┌──────────┐         │
│    │ CPU      │        │ GPU      │         │
│    │ X925+A725│        │ Blackwell│         │
│    │          │        │ 6144 CUDA│         │
│    └────┬─────┘        └────┬─────┘         │
│         │    C2C NVLink     │               │
│         └───────────────────┘               │
│                                             │
│  Measured CPU bandwidth:                    │
│    A725 cluster: ~26 GB/s                   │
│    X925 cluster: ~38 GB/s                   │
│                                             │
│  DRAM latency:                              │
│    Idle:     113 ns                         │
│    GPU load: 351+ ns  ⚠️                    │
└─────────────────────────────────────────────┘
```

### 11.2 5G NR PHY Bandwidth Demands (30 kHz SCS, 100 MHz FR1)

| Component | Estimated BW | Notes |
|---|---|---|
| Sample I/O (61.44 Msps) | ~0.5 GB/s | IQ at 16-bit I + 16-bit Q |
| FFT/IFFT | ~2-4 GB/s | Complex float, 2 per slot |
| LDPC decoding | ~4-8 GB/s | Multiple code blocks, iterative |
| Channel estimation | ~1-2 GB/s | DMRS processing, interpolation |
| Equalization | ~1-2 GB/s | Per-subcarrier, per-layer |
| **Total PHY pipeline** | **~10-20 GB/s** | |

**Assessment:** 38 GB/s X925 cluster bandwidth is **sufficient** for UE PHY at 100 MHz FR1. The 273 GB/s aggregate is well above requirements.

### 11.3 GPU Contention Risk

**Critical issue:** When GPU CUDA kernels (LDPC, FFT) are actively using memory, CPU DRAM latency triples from 113 ns to 351+ ns.

**Mitigations:**
1. Pipeline GPU and CPU memory accesses (avoid simultaneous bursts)
2. Use CUDA streams with careful scheduling
3. Pre-fetch data into L2/L3 cache before GPU bursts
4. Consider pinning frequently-accessed PHY buffers to SLC (16 MB system-level cache)

---

## 12. Known Issues and Risks

### 12.1 Build and Compilation

| Issue | Severity | Mitigation |
|---|---|---|
| SIMDe AVX-512 coverage only ~33% | **High** | Manual NEON implementations for missing functions |
| OAI CMake SIMD detection is x86-centric | Medium | Patch CMakeLists.txt for ARM NEON/SVE2 detection |
| Cross-compile Dockerfiles target Ubuntu 22.04 | Low | Update to 24.04 base |
| `mlnx-ofed-kernel-dkms` build failures on recent kernels | Medium | Pin compatible OFED version or use in-tree mlx5 driver |
| OAI test vectors validated on x86 only | Medium | Run full PHY simulator test suite on ARM, compare numerical results |

### 12.2 Runtime and Performance

| Issue | Severity | Mitigation |
|---|---|---|
| No published OAI NR benchmarks on ARM | **High** | Must benchmark and characterize from scratch |
| big.LITTLE thread migration causes jitter | **High** | `isolcpus` + `taskset` to pin RT threads to X925 only |
| GPU memory contention triples CPU DRAM latency | **High** | Pipeline GPU/CPU accesses, use CUDA streams |
| USB SDR latency jitter (if using B210) | Medium | Use network-attached SDR (N310/X410) for production |
| DGX OS stock kernel is not RT | **High** | Build custom PREEMPT_RT kernel from 6.17 source |
| NVIDIA GPU driver compatibility with RT kernel | Medium | Test thoroughly; may need driver version pinning |

### 12.3 Functional Gaps

| Issue | Severity | Mitigation |
|---|---|---|
| NVIDIA Aerial is gNB-only (no UE support) | Medium | Reuse cuPHY primitives via pyAerial; build custom UE pipeline |
| No MIG (Multi-Instance GPU) on GB10 | Low | Use CUDA streams for time-slicing PHY and AI workloads |
| Timer/clock resolution differences on ARM | Low | Verify `clock_gettime(CLOCK_MONOTONIC)` accuracy |
| UHD (USRP driver) tuning docs are x86-focused | Low | Benchmark and tune on ARM; basic functionality confirmed |

---

## 13. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

- [ ] Verify OAI ARM64 build on DGX Spark (use existing Docker cross-compile support)
- [ ] Identify and catalog all SIMD compilation failures (SIMDe AVX-512 gaps)
- [ ] Build PREEMPT_RT kernel from DGX OS 6.17 sources
- [ ] Validate NVIDIA GPU driver + ConnectX-7 on custom RT kernel
- [ ] Set up USRP connectivity (B210 USB for initial testing)
- [ ] Run OAI PHY simulator tests on ARM, compare with x86 reference

### Phase 2: SIMD Porting (Weeks 4-8)

- [ ] Create `libldpc_neon.so` with hand-optimized NEON LDPC decoder
- [ ] Port remaining SIMDe-unsupported AVX-512 intrinsics to NEON
- [ ] Evaluate Ne10/FFTW as DFT replacement vs porting custom DFT
- [ ] Port channel estimation complex multiply-accumulate (consider SVE2 FCMLA)
- [ ] Port DLSCH/ULSCH demodulation SIMD paths
- [ ] Benchmark each ported function against x86 reference

### Phase 3: CUDA Offload (Weeks 6-10, overlapping with Phase 2)

- [ ] Integrate Aerial cuPHY LDPC decoder into OAI UE via `libldpc_cuda.so`
- [ ] Port/adapt cuPHY FFT for UE OFDM demodulation
- [ ] Port cuPHY Polar decoder for PDCCH/PBCH
- [ ] Implement CUDA stream management for concurrent PHY operations
- [ ] Benchmark GPU-offloaded PHY pipeline end-to-end
- [ ] Measure and mitigate CPU memory latency under GPU load

### Phase 4: Real-Time Integration (Weeks 9-12)

- [ ] Configure core isolation (isolcpus, nohz_full, rcu_nocbs)
- [ ] Pin OAI threads to X925 cores per mapping in Section 8.5
- [ ] Set IRQ affinity, disable frequency scaling
- [ ] Run cyclictest to measure RT latency jitter
- [ ] Upgrade SDR to network-attached (N310/X410) for production latency
- [ ] End-to-end test: UE attach, data transfer, handover

### Phase 5: Dual-Node Cluster (Weeks 11-14, overlapping)

- [ ] Set up Spark Stacking between two nodes
- [ ] Configure workload split (PHY on Node 1, upper layers on Node 2)
- [ ] Benchmark inter-node latency and throughput
- [ ] Test distributed operation with increasing traffic load
- [ ] Optimize for target bandwidth and MIMO configuration

### Phase 6: Validation and Optimization (Weeks 13-16)

- [ ] Full 3GPP conformance test subset
- [ ] Stress testing: maximum throughput, sustained operation
- [ ] Profile and optimize remaining bottlenecks
- [ ] Document final configuration and operational procedures
- [ ] Characterize maximum supported bandwidth/MIMO/SCS configuration

---

## 14. References

### DGX Spark / GB10 Platform
- [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [DGX Spark Stacking Guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [Inside GB10 Memory Subsystem (Chips and Cheese)](https://chipsandcheese.com/p/inside-nvidia-gb10s-memory-subsystem)
- [ASUS Ascent GX10 Specifications](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/)
- [DGX Spark LDPC Acceleration Study](https://arxiv.org/html/2602.04652v1)

### OpenAirInterface
- [OAI 5G NR RAN](https://openairinterface.org/ran/)
- [OAI Build Documentation](https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/doc/BUILD.md)
- [OAI Feature Set](https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/doc/FEATURE_SET.md)
- [OAI Reference Architecture with USRP (Ettus)](https://kb.ettus.com/OAI_Reference_Architecture_for_5G_and_6G_Research_with_USRP)
- [OAI GitLab MR !1636: SIMDe Integration](https://gitlab.eurecom.fr/oai/openairinterface5g/-/merge_requests/1636)

### SIMD Porting
- [SIMDe (SIMD Everywhere)](https://github.com/simd-everywhere/simde)
- [SIMDe AVX-512 Implementation Status](https://github.com/simd-everywhere/implementation-status/blob/main/avx512.md)
- [sse2neon Translation Library](https://github.com/DLTcollab/sse2neon)
- [ARM: Port Intel Intrinsics with SIMDe](https://developer.arm.com/documentation/102581/latest/Port-with-SSE2Neon-and-SIMDe)
- [ARM SVE2 Critical Analysis](https://gist.github.com/zingaburga/805669eb891c820bd220418ee3f0d6bd)
- [5x Faster Set Intersections: SVE2 vs AVX-512 Benchmarks](https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/)
- [DGX Spark SIMD Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/simd.html)

### NVIDIA Aerial / CUDA RAN
- [NVIDIA Aerial CUDA-Accelerated RAN SDK](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- [Aerial cuPHY Documentation](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/cubb/index.html)
- [pyAerial LDPC API](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/pyaerial/api_reference/aerial.phy5g.ldpc.html)

### ARM Architecture
- [ARM Cortex-X925 Technical Reference](https://developer.arm.com/documentation/102807/latest/Technical-overview)
- [Cortex-X925 Microarchitecture (Chips and Cheese)](https://chipsandcheese.com/p/arms-cortex-x925-reaching-desktop)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [ARM SVE Programming Guide](https://developer.arm.com/documentation/102476/latest/)

### Real-Time Linux
- [Real-time Ubuntu 24.04 LTS](https://ubuntu.com/blog/real-time-24-04)
- [Building PREEMPT_RT Kernel (acontis)](https://www.acontis.com/en/building-a-real-time-linux-kernel-in-ubuntu-preemptrt.html)
- [CPU Core Isolation Guide](https://manuel.bernhardt.io/posts/2023-11-16-core-pinning/)
- [USRP Host Performance Tuning](https://kb.ettus.com/USRP_Host_Performance_Tuning_Tips_and_Tricks)
