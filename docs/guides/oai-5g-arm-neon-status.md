# OpenAirInterface 5G ARM NEON Support Status

## Current State (March 2026, develop branch)

OAI uses a **hybrid ARM optimization strategy** — not just SIMDe translation, but an increasing number of hand-written native NEON code paths for performance-critical PHY functions.

---

## Optimization Layers

| Layer | Role | Coverage |
|---|---|---|
| **SIMDe (SIMD Everywhere)** | Baseline portability: translates all SSE/AVX2 intrinsics to NEON transparently | All PHY code compiles on ARM |
| **Native NEON Intrinsics** | Hand-optimized hot paths where SIMDe performance or correctness is insufficient | DFT, precoding, scrambling, turbo decoder, LDPC rate-matching |
| **Arm RAL (RAN Acceleration Library)** | Vendor-optimized ARM functions for FHI 7.2 fronthaul builds | `armral-25.01` integrated in Docker builds |
| **128-bit Forced Paths** | `USE_128BIT` defined on aarch64 to avoid poorly-emulated 256-bit AVX2 code | LDPC decoder, UL LLR computation, PHY common |

---

## Native NEON Implementation Status by PHY Function

### Fully Native NEON (No SIMDe Dependency)

| Function | File(s) | NEON Features Used | Notes |
|---|---|---|---|
| **DFT/IDFT (OFDM)** | `openair1/PHY/TOOLS/oai_dfts_neon.c` | `vmulq_s16`, `vmull_s16`, `vqaddq_s32`, `vcombine_s32`, `vpadd_s32`, `vrev32q_s16` | Complete dedicated NEON file, compiled as `libdfts.so` alongside x86 `oai_dfts.c` |
| **Precoding (up to 4-layer MIMO)** | `openair1/PHY/MODULATION/nr_modulation.c` | `vuzp1q_s16`, `vuzp2q_s16`, `vqdmulhq_s16`, `vqrdmlshq_s16` (ARMv8.1-A), `vmull_s16`/`vmlsl_s16` (ARMv8.0-A fallback) | Comment: *"SIMDe doesn't handle this properly, gcc up to 14.2 neither"* |
| **Complex Vector Rotation** | `openair1/PHY/TOOLS/tools_defs.h` | `vqdmulhq_s16`, `vqrdmlahq_s16`, `vqrdmlshq_s16` (ARMv8.1-A QRDMX) + ARMv8.0-A fallback | Exploits ARMv8.1-A rounding doubling multiply-accumulate |
| **Scrambling** | `openair1/PHY/NR_TRANSPORT/nr_scrambling.c` | `vld1q_u32`, `veorq_u32` | Vectorized XOR operations |
| **Turbo Decoder (LTE)** | `openair1/PHY/CODING/3gpplte_turbo_decoder_sse_8bit.c` | `int8x16_t`, `vhaddq_s8`, `vhsubq_s8`, `vdupq_n_s8`, `vsetq_lane_s8` | Full `#elif defined(__aarch64__)` path |
| **LDPC Rate-Matching** | `openair1/PHY/CODING/nrLDPC_coding/.../nrLDPC_coding_segment_encoder.c` | `vld1q_s8`, `vld1q_u8`, `vshlq_u8`, `vandq_u8`, `vaddv_u8` | Native `#elif defined(__aarch64__)` path |
| **O-RAN FHI Compression** | `cmake_targets/tools/oran_fhi_integration_patches/` | Direct `#include <arm_neon.h>` | Fixed in commit `d7512bdc8b` (2026-02-27) |

### ARM-Optimized via SIMDe (Hybrid)

| Function | File(s) | Optimization | Notes |
|---|---|---|---|
| **LDPC Encoder** | `openair1/PHY/CODING/nrLDPC_encoder/ldpc_encode_parity_check.c` | Defines `USE_ALIGNR` on aarch64, selects `ldpc384_alignr_byte_128.c` | `simde_mm_alignr_epi8` maps efficiently to NEON `vextq` |
| **LDPC Decoder** | Via `USE_128BIT` forced path | 128-bit SIMDe (avoids 256-bit emulation overhead) | **Main remaining optimization gap** — no native NEON decoder |
| **UL LLR Computation** | `openair1/PHY/NR_TRANSPORT/nr_ulsch_llr_computation.c` | `USE_128BIT` on aarch64 | SIMDe 128-bit path |
| **PHY Common** | `openair1/PHY/nr_phy_common/src/nr_phy_common.c` | `USE_128BIT` on aarch64 | SIMDe 128-bit path |

### Not Yet ARM-Optimized

| Function | Current State | Impact | Priority |
|---|---|---|---|
| **LDPC Decoder** | SIMDe 128-bit translation only | **Critical** — dominates slot processing time (~710 us on CPU) | HIGH — best candidate for native NEON or CUDA offload |
| **Channel Estimation** | SIMDe translation | Moderate — complex multiply-accumulate would benefit from SVE2 FCMLA | MEDIUM |
| **MIMO Detection/Equalization** | SIMDe translation | Moderate | MEDIUM |
| **Polar Decoder** | SIMDe translation | Lower — control channel, less throughput demand | LOW |

---

## Build System ARM Support

### CMakeLists.txt CPU Detection

OAI detects ARM CPU part numbers from `/proc/cpuinfo` and sets specific compiler flags:

| CPU Part | Identification | Compiler Flags |
|---|---|---|
| Neoverse-V1 | `0xd40` | `-mcpu=neoverse-v1` |
| Neoverse-V2 | `0xd4f` | `-mcpu=neoverse-v2 -ftree-vectorize` |
| Neoverse-N1 | `0xd0c` | `-mcpu=neoverse-n1` |
| Neoverse-N2 | `0xd49` | `-mcpu=neoverse-n2 -ftree-vectorize` |
| Cortex-A53 | `0xd03` | `-mcpu=cortex-a53 -march=armv8-a+simd` |
| Default (native) | Any other | `-mcpu=native` |

**Note:** Cortex-X925 (GB10's performance core, part `0xd85`) is not explicitly listed — will fall through to `-mcpu=native`. This should work correctly but may miss X925-specific scheduling optimizations. A patch to add explicit X925 detection would be beneficial.

### Cross-Compilation

- Toolchain: `cross-arm.cmake` with `gcc-aarch64-linux-gnu`
- Architecture: `-march=armv8.2-a`
- SIMDe version pinned: commit `389f360a66d4a3bec62b7d71ad8be877487809ba`
- Requires building LDPC code generators on x86 host first, then cross-compiling with `-DNATIVE_DIR`

### Docker ARM64 Build Targets

| Dockerfile | Purpose |
|---|---|
| `Dockerfile.base.ubuntu.cross-arm64` | Base image for cross-compiling to ARM64 (Ubuntu 24.04) |
| `Dockerfile.build.ubuntu.cross-arm64` | Cross-compiled ARM64 build targets |
| `Dockerfile.build.fhi72.native_arm.ubuntu` | Native ARM build with DPDK, O-RAN FHI, and Arm RAL |

---

## Recent ARM/NEON Development Activity (2025-2026)

| Date | Reference | Description |
|---|---|---|
| 2026-02-27 | Commit `d7512bdc8b` | Fix O-RAN compression for ARM |
| 2026-02-10 | MR !3896 (`armv8-a-compatibility`) | ARMv8.0-A compatibility merged into `integration_2026_w07` |
| 2026-01-30 | Commit `d7ea319b59` | ARMv8.0-A fallback for `cmac0_prec128()`/`rotate_cpx_vector()` |
| 2026-01-30 | Commit `15ee75980b` | Cortex-A53 specific compile options |
| 2025 (w36) | Integration merge | Speedup complex rotate for aarch64 |
| 2025-01 | MR !3187 | ARM NEON and LDPC coding improvements |
| 2023-01 | MR !1909 (178 files) | Major rework for aarch64 after SIMDe integration, tested on Neoverse N1 |
| 2022-08 | MR !1636 (114 files) | Foundational SIMDe integration |

---

## SVE / SVE2 Support

**No SVE or SVE2 support exists** in the OAI codebase. No references to `__ARM_FEATURE_SVE`, `svld1`, or any SVE intrinsics were found.

The `-ftree-vectorize` flag set for Neoverse-V1/V2/N2 may allow the compiler to auto-vectorize using SVE, but there is no explicit SVE code.

**Opportunity for GB10 (Cortex-X925, 128-bit SVE2):**
- SVE2 `FCMLA` (fused complex multiply-add) would benefit channel estimation
- SVE2 reduction operations (`ADDV`, `FADDV`) useful for signal processing
- At 128-bit width, SVE2 provides no throughput advantage over NEON — benefit is instruction set richness

---

## Arm RAN Acceleration Library (RAL)

Arm RAL (`armral-25.01`) is integrated in the FHI 7.2 ARM Docker build and provides hardware-optimized:
- LDPC encode/decode
- FFT/IFFT
- Rate matching
- Channel coding primitives

This is cloned from `git.gitlab.arm.com/networking/ral.git` during the Docker build process. RAL functions could potentially replace SIMDe-translated paths for additional performance on ARM.

---

## Implications for DGX Spark / GX10 Deployment

### What's Already Done (Less Work Than Expected)

| Component | Status | Remaining Work |
|---|---|---|
| DFT/IDFT | Native NEON | None — verify performance on X925 |
| Precoding | Native NEON (ARMv8.1-A) | X925 supports QRDMX — should work optimally |
| Scrambling | Native NEON | None |
| Turbo Decoder (LTE) | Native NEON | None (LTE only) |
| LDPC Encoder | ARM-optimized | Minor — already selects efficient 128-bit path |
| LDPC Rate-Matching | Native NEON | None |
| Build System | ARM64 supported | Add Cortex-X925 (`0xd85`) detection to CMakeLists.txt |

### What Still Needs Work

| Component | Current State | Recommended Action |
|---|---|---|
| **LDPC Decoder** | SIMDe 128-bit | **Priority 1:** CUDA offload (proven 6x speedup on GB10) or native NEON rewrite |
| **Channel Estimation** | SIMDe | Native NEON rewrite, explore SVE2 FCMLA |
| **MIMO Detection/Equalization** | SIMDe | Native NEON rewrite for hot paths |
| **Polar Decoder** | SIMDe | CUDA offload or native NEON (lower priority) |
| **PREEMPT_RT Kernel** | Not available on DGX OS | Must build custom RT kernel from 6.17 sources |
| **Core Isolation** | Not configured | Pin PHY threads to X925 cores, OS to A725 |

### Revised Effort Estimate

The original estimate of 4-5 weeks for SIMD porting (Phase 2 in the implementation roadmap) can be **reduced to 2-3 weeks**, focused primarily on:
1. LDPC decoder optimization (native NEON or CUDA — 1-2 weeks)
2. Channel estimation native NEON (1 week)
3. CMake patch for X925 detection (hours)
4. Verification and benchmarking (1 week)

---

## References

- [OAI GitLab Repository](https://gitlab.eurecom.fr/oai/openairinterface5g)
- [SIMDe Integration MR !1636](https://gitlab.eurecom.fr/oai/openairinterface5g/-/merge_requests/1636)
- [aarch64 Rework MR !1909](https://gitlab.eurecom.fr/oai/openairinterface5g/-/merge_requests/1909)
- [ARMv8-A Compatibility MR !3896](https://gitlab.eurecom.fr/oai/openairinterface5g/-/merge_requests/3896)
- [OAI LDPC Implementation Documentation](https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/openair1/PHY/CODING/DOC/LDPCImplementation.md)
- [OAI Cross-Compile Documentation](https://github.com/OPENAIRINTERFACE/openairinterface5g/blob/develop/doc/cross-compile.md)
- [Arm RAL Open Source Announcement](https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/arm-ral-is-now-open-source)
- [Arm RAL Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/ran/armral/)
- [LDPC Acceleration on DGX Spark (arXiv 2602.04652)](https://arxiv.org/html/2602.04652v1)
- [SIMDe GitHub Repository](https://github.com/simd-everywhere/simde)
