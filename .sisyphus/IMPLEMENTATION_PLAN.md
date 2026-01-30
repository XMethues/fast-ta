# TA-Lib Rust 重写项目 - 完整实施计划

**版本**: v2.0 (Multi-ISSE SIMD Revision)  
**创建日期**: 2026-01-29  
**最后更新**: 2026-01-30  
**状态**: 修订中 (Phase 1: 核心架构重新设计)  

---

## 重要说明

**本计划为 v2.0 修订版，关键变更**：

1. **SIMD架构重大改进**: 从 `wide` 库的编译时单一目标，升级为手动实现的多指令集运行时动态分派（不使用 `multiversion` crate，详见 `simd-implementation.md`）
2. **跨平台支持**: 新增对 x86_64 (AVX2/AVX-512)、ARM64 (NEON)、WASM (SIMD128) 的全面支持
3. **向后兼容**: 保持现有的 `f32`/`f64` 特性系统不变
4. **新增任务**: 任务 1.6 (构建配置) 和 1.7 (多平台测试基础设施)

**影响范围**: Phase 1 的核心架构设计，特别是任务 1.3、1.4、1.5

---

## 目录

1. [项目概述](#项目概述)
2. [架构设计](#架构设计)
3. [实施阶段总览](#实施阶段总览)
4. [Phase 1: 核心基础设施](#phase-1-核心基础设施)
5. [Phase 2: 重叠研究指标](#phase-2-重叠研究指标)
6. [Phase 3: 动量指标](#phase-3-动量指标)
7. [Phase 4: 成交量与波动率](#phase-4-成交量与波动率)
8. [Phase 5: 绑定层](#phase-5-绑定层)
9. [Phase 6: 验证与优化](#phase-6-验证与优化)
10. [风险管理](#风险管理)
11. [质量保证](#质量保证)
12. [附录](#附录)

---

## 项目概述

### 项目目标
将 TA-Lib（技术分析库）从 C/C++ 移植到 Rust，实现：
- 150+ 技术指标
- 零拷贝 API
- **多指令集 SIMD 优化**（手动调度 + `wide`）
- 批量与流式双模式
- Python 和 WASM 绑定

### 成功标准
- 100% 数值精度匹配 TA-Lib C（ε < 1e-10）
- 所有 150+ 指标实现
- **SIMD 优化在 x86_64 (AVX2/AVX-512)、ARM64 (NEON)、WASM (SIMD128) 上有效**
- 流式 API 延迟 < 1ms
- **跨平台测试覆盖率 > 90%**

### 关键技术决策 (v2.0 更新)

**SIMD 架构 (重大变更)**:
- **运行时多指令集分派**: 手动实现运行时 CPU 特性检测和最优代码路径选择（使用 `std::is_x86_feature_detected!` 和函数指针调度，不使用外部 crate）
- **底层 SIMD 实现**: 继续使用 `wide` crate 提供稳定的跨平台 SIMD 原语
- **平台支持矩阵**:
| 平台 | 指令集 | 实现方式 |
|------|--------|----------|
| x86_64 | AVX2 | `wide::f64x4` / `f32x8` |
| x86_64 | AVX-512 | `wide::f64x8` / `f32x16` |
| ARM64 | NEON | `wide::f64x2` / `f32x4` |
| WASM | SIMD128 | `wide::f64x2` / `f32x4` |
| 通用 | Scalar | 纯 Rust 标量实现 (fallback) |

- **向后兼容**: 完全保持现有的 `f32`/`f64` 特性系统，零破坏性变更

**架构**: 4-crate workspace（core, py, wasm, benchmarks）
**API**: 零拷贝（`&[Float]` / `&mut [Float]`），`Float` 类型通过条件编译确定
**指标接口**: 统一 `Indicator` trait，采用混合方案（性能 + 易用性）
  - `compute(&self, inputs, outputs)` - **零拷贝批量计算**（性能优先，无内存分配）
  - `compute_to_vec(&self, inputs)` - **便捷批量计算**（易用性优先，返回 Vec）
  - `next(&mut self, input)` - 获取最新值（实时流式）
  - `stream(&mut self, inputs)` - 流式处理多个输入
**浮点精度**: 通过 Cargo features 在编译时选择 `f32` 或 `f64`（默认 `f64`），使用条件编译实现零成本抽象

---

## 架构设计

### SIMD 抽象层架构 (v2.0)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Application Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │    SMA       │  │    EMA       │  │   RSI        │  ... 150+ indicators│
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                  │                  │                        │
│         └──────────────────┴──────────────────┘                        │
│                            │                                           │
└────────────────────────────┼───────────────────────────────────────────┘
                             │
┌────────────────────────────┼───────────────────────────────────────────┐
│                   SIMD Abstraction Layer                              │
│                            │                                           │
│  ┌─────────────────────────▼──────────────────────────────────────┐   │
│  │  simd/mod.rs - Platform-agnostic SIMD API                        │   │
│  │                                                                  │   │
│  │  trait SimdFloat {                                               │   │
│  │      fn sum(data: &[Float]) -> Float;                            │   │
│  │      fn dot_product(a: &[Float], b: &[Float]) -> Float;          │   │
│  │      fn add(a: Self, b: Self) -> Self;                           │   │
│  │      ...                                                        │   │
│  │  }                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            │                                           │
│  ┌─────────────────────────▼──────────────────────────────────────┐   │
│  │  Runtime Dispatch (Manual)                                       │   │
│  │                                                                  │   │
│  │  Targets:                                                          │   │
│  │      "x86_64+avx512f+avx512vl",  // AVX-512 (if available)    │   │
│  │      "x86_64+avx2",              // AVX2 (baseline modern)  │   │
│  │      "aarch64+neon",             // ARM64 NEON              │   │
│  │      "wasm32+simd128"            // WASM SIMD128            │   │
│  │  ))]                                                             │   │
│  │  fn sum_simd(data: &[Float]) -> Float {                         │   │
│  │      // Implementation uses wide crate for each target        │   │
│  │  }                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            │                                           │
└────────────────────────────┼───────────────────────────────────────────┘
                             │
┌────────────────────────────┼───────────────────────────────────────────┐
│                   Platform-Specific Implementations                    │
│                            │                                           │
│  ┌───────────────────┐    │    ┌───────────────────┐                   │
│  │ x86_64/AVX512      │    │    │ x86_64/AVX2       │                   │
│  │ (wide::f64x4)     │    │    │ (wide::f64x2)     │                   │
│  └─────────┬─────────┘    │    └─────────┬─────────┘                   │
│            │              │              │                              │
│  ┌─────────▼─────────┐    │    ┌─────────▼─────────┐                   │
│  │ ARM64/NEON        │    │    │ WASM/SIMD128      │                   │
│  │ (wide::f64x2)     │    │    │ (wide::f64x2)     │                   │
│  └─────────┬─────────┘    │    └─────────┬─────────┘                   │
│            │              │              │                              │
│            └──────────────┴──────────────┘                              │
│                           │                                            │
│  ┌────────────────────────▼────────────────────────┐                   │
│  │              Scalar Fallback                     │                   │
│  │         (Pure Rust, always available)            │                   │
│  └─────────────────────────────────────────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Decision Records

#### ADR-001: SIMD Implementation Strategy

**Status**: Accepted (v2.0 Revision)

**Context**:
The original plan used `wide` crate with compile-time target selection. This works for single-target builds but fails for:
- Distribution builds that need to run on multiple CPU generations
- Apple Silicon (ARM64) support
- WASM targets
- Runtime CPU feature detection

**Decision**:
Use manual runtime dispatch with `std::is_x86_feature_detected!` and function pointers, combined with `wide` crate for SIMD primitives.

**Rationale**:
1. **Runtime Dispatch**: Manual dispatch using `std::is_x86_feature_detected!` to check CPU capabilities at startup, then selecting the optimal implementation via function pointers
2. **Performance**: No branch prediction overhead - direct function pointer dispatch
3. **Maintainability**: Single codebase, no manual feature detection
4. **Portability**: Supports all target platforms (x86_64, ARM64, WASM)
5. **Fallback**: Automatic scalar fallback when no SIMD available

**Consequences**:
- **Positive**: 
  - Single binary runs optimally on all CPUs
  - No feature flags needed for SIMD selection
  - Easy cross-compilation
  - Future CPU support automatic
- **Negative**:
  - Binary size increases (multiple implementations)
  - Compile time increases
  - More complex testing (must test all paths)

**Alternatives Considered**:
1. **std::simd**: Nightly only, not production ready
2. **Manual feature detection**: Complex, error-prone
3. **Compile-time selection only**: Poor distribution support
4. **portable-simd**: Still experimental

---

#### ADR-002: Platform Support Matrix

**Status**: Accepted

**Supported Platforms**:

| Platform | Minimum Version | SIMD Levels | Priority |
|----------|----------------|-------------|----------|
| x86_64 | AVX2 (2011) | AVX2, AVX-512 | P0 |
| x86_64-v3 | AVX2 (2013) | AVX2, AVX-512 | P0 |
| aarch64 | ARMv8 (2013) | NEON | P0 |
| wasm32 | SIMD128 | SIMD128 | P1 |
 
**Notes**:
- AVX2 is baseline for modern x86_64 CPUs (Sandy Bridge, 2011+)
- AVX2 provides 4x lanes over scalar (f64x4 vs scalar)
- AVX-512 provides 2x lanes over AVX2 but available on Skylake-X and newer
- NEON is standard on all ARM64 (Apple Silicon, AWS Graviton, etc.)

---

## 实施阶段总览

| 阶段 | 名称 | 持续时间 | 主要交付物 | 依赖 |
|------|------|----------|------------|------|
| **Phase 1** | 核心基础设施 | 6 周 | Workspace, traits, **Multi-ISSE SIMD层**, 测试框架 | 无 |
| **Phase 2** | 重叠研究指标 | 4 周 | 16 个移动平均指标 | Phase 1 |
| **Phase 3** | 动量指标 | 4 周 | 33 个动量指标 | Phase 1 |
| **Phase 4** | 成交量与波动率 | 3 周 | 6 个指标 | Phase 1 |
| **Phase 5** | 绑定层 | 3 周 | Python 和 WASM 绑定 | Phase 1-4 |
| **Phase 6** | 验证与优化 | 4 周 | 完整测试套件，性能优化 | Phase 1-5 |

**总工期**: 28 周（约 7 个月）- **增加 2 周用于 Multi-ISSE SIMD 架构**

**关键路径**: Phase 1 → Phase 2 → Phase 6

**可并行**: Phase 2, 3, 4 可以并行执行

---

## Phase 1: 核心基础设施

**阶段名称**: Core Infrastructure  
**持续时间**: 6 周（Weeks 1-6）- **增加 2 周**  
**目标**: 建立项目基础架构，实现核心 traits 和 **Multi-ISSE SIMD 抽象层**  
**依赖**: 无  

**阶段进度**:

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 任务 1.1: 创建 Workspace 结构 | ✅ 已完成 | 2026-01-29 | 基础结构已就绪 |
| 任务 1.2: 实现错误类型系统 | ✅ 已完成 | 2026-01-29 | 19 个测试通过 |
| 任务 1.3: 实现核心 Traits | ⬜ 修订中 | - | **需更新以支持 Multi-ISSE** |
| 任务 1.4: 实现 SIMD 抽象层 | ⬜ 待开始 | - | **完全重写 - Multiversion 架构** |
| 任务 1.5: 设置测试基础设施 | ⬜ 待开始 | - | **需添加多平台测试** |
| 任务 1.6: 构建配置与平台支持 | ⬜ 新增 | - | **新增任务 - build.rs, CI矩阵** |
| 任务 1.7: 跨平台测试基础设施 | ⬜ 新增 | - | **新增任务 - Docker, QEMU, 模拟器** |

**总体进度**: 2/7 任务完成 (29%) - **任务数从5增加到7**

---

### 任务 1.1: 创建 Workspace 结构 ✅

*(保持原有内容不变 - 已完成)*

---

### 任务 1.2: 实现错误类型系统 ✅

*(保持原有内容不变 - 已完成)*

---

### 任务 1.3: 实现核心 Traits (修订版)

**任务 ID**: 1.3  
**任务名称**: 实现统一 Indicator trait（支持批量、流式和单值查询）+ **Multi-ISSE SIMD 集成**  
**优先级**: P0 (最高)  
**预估工时**: 16 小时 **(+4 小时 用于 SIMD 集成)**  
**负责人**: TBD  
**状态**: ⬜ 修订中  

**描述**:  
定义统一的 `Indicator` trait，同时支持批量计算、获取最新值和流式处理，建立指标实现的标准接口。**新增：集成 Multi-ISSE SIMD 抽象层，确保所有指标自动获得跨平台 SIMD 加速。**

**关键更新（v2.0）**:
1. **SIMD 集成**: 指标实现通过 `SimdFloat` trait 自动使用最优 SIMD 路径
2. **透明加速**: 指标开发者无需关心底层平台，自动获得加速
3. **向后兼容**: 现有 trait 设计不变，SIMD 层作为实现细节

**子任务**:

- [ ] 1.3.1 定义 `Indicator` trait  
  *(保持原有详细内容不变)*

- [ ] 1.3.2 定义辅助 trait  
  *(保持原有详细内容不变)*  
  **新增子任务**: 
  - [ ] 1.3.2.1 定义 `SimdCompute` trait  
    ```rust
    /// 内部 trait，用于指标实现中的 SIMD 加速计算
    /// 由 ta-core 内部实现，指标开发者通常不需要直接实现
    pub trait SimdCompute {
        /// 使用最优 SIMD 路径计算数组和
        fn simd_sum(data: &[Float]) -> Float;
        
        /// 使用最优 SIMD 路径计算点积
        fn simd_dot_product(a: &[Float], b: &[Float]) -> Float;
        
        /// 批量滑动窗口计算（SMA 优化）
        fn simd_rolling_sum(data: &[Float], window: usize) -> Vec<Float>;
    }
    ```

- [ ] 1.3.3 实现示例指标（SMA）**[重大更新]**  
  **关键变更**: SMA 不再直接使用 `wide::f64x4`，而是通过 `SimdCompute` trait 使用多平台 SIMD  
  
  - 创建 `crates/ta-core/src/overlap/sma.rs`  
  - 实现 `Sma` 结构体：  
    ```rust
    pub struct Sma {
        period: usize,
        buffer: Vec<Float>,  // 用于流式模式
        sum: Float,
        index: usize,
        count: usize,
    }
    ```
  - 实现 `Indicator` trait：  
    - **核心方法 `compute()`**:  
      ```rust
      fn compute(&self, inputs: &[Float], outputs: &mut [Float]) -> Result<usize> {
          // 小窗口优化：period < SIMD_LANES * 2 时使用标量实现
          if self.period < MIN_SIMD_PERIOD {
              return self.compute_scalar(inputs, outputs);
          }
          
          // 使用 SIMD 加速计算
          // 通过 SimdCompute trait 自动选择最优 SIMD 路径
          let result = simd::rolling_average(inputs, self.period, outputs)?;
          Ok(result.len())
      }
      ```
    - `compute_to_vec()`: 直接使用 trait 默认实现（复用 `compute()`）
    - `next()`: 滚动和算法，支持流式处理
    - `stream()`: 直接使用 trait 默认实现
  - 实现 `Resettable` trait  
  - **性能要求**: SIMD 加速比 > 2x（数据量 > 1000，period >= MIN_SIMD_PERIOD）
   - **平台支持**: 自动适应 x86_64 (AVX2/AVX-512)、ARM64 (NEON)、WASM (SIMD128)
  - 添加完整文档和示例，说明SIMD优化策略和多平台支持

   - [ ] 1.3.4 编写单元测试
   *(保持原有内容不变)*
   - **新增**: 
   - 在 x86_64 上测试 AVX2、AVX-512 路径
   - 在 ARM64 上测试 NEON 路径
   - 测试标量 fallback 路径
   - 验证所有路径结果一致性

- [ ] 1.3.5 添加示例代码和文档  
  *(保持原有内容不变)*  
  **新增**:  
  - 添加多平台构建示例  
  - 添加运行时 CPU 特性检测示例

**验收标准**:
- [ ] `Indicator` trait 定义完成且有完整文档
- [ ] `Resettable` trait 定义完成（可选）
- [ ] `SimdCompute` trait 定义完成并在内部使用
- [ ] SMA 示例指标实现并测试通过
   - [ ] **SMA 在所有支持平台（x86_64 AVX2/AVX-512、ARM64 NEON、WASM SIMD128）上通过测试**
- [ ] **所有 SIMD 路径结果一致性验证通过**
- [ ] 零拷贝 `compute()` 接口正常工作（性能验证）
- [ ] 便捷 `compute_to_vec()` 接口正常工作（易用性验证）
- [ ] 流式 `next()` 和 `stream()` 接口正常工作
- [ ] 三种模式（批量、流式、单值查询）都能正常工作
- [ ] 所有单元测试通过
- [ ] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/traits.rs`
- `crates/ta-core/src/simd/mod.rs` (内部模块)
- `crates/ta-core/src/overlap/sma.rs`
- 单元测试文件
- 文档和示例

**依赖**: 任务 1.2 (错误类型系统), **任务 1.4 (SIMD 抽象层 - 需先完成)**  
**可并行**: 部分子任务可并行  
**风险**: **高** (Multi-ISSE SIMD 架构复杂度高，需要充分测试)  

---

### 任务 1.4: 实现 SIMD 抽象层 (完全重写 - Multi-ISSE 架构)

**任务 ID**: 1.4  
**任务名称**: 实现 Multi-ISSE SIMD 抽象层（手动调度 + wide）  
**优先级**: P0 (最高)  
**预估工时**: 24 小时 **(原10小时增加到24小时 - 架构复杂度显著提升)**  
**负责人**: TBD  
**状态**: ⬜ 待开始  

**描述**:  
实现基于手动调度的 Multi-ISSE SIMD 抽象层，结合 `wide` crate 提供稳定的跨平台 SIMD 原语。**这是 v2.0 的核心架构变更，从编译时单一目标升级为运行时多指令集动态分派。不使用 `multiversion` crate，改为手动实现函数指针调度（详见 `simd-implementation.md`）。**

**关键设计目标**:
1. **运行时自动选择最优路径**: 无需编译时指定，运行时自动检测 CPU 特性
2. **单二进制多平台**: 一个二进制文件在所有平台最优运行
3. **零成本抽象**: 无运行时开销（直接函数指针分派）
4. **向后兼容**: 完全保持 `f32`/`f64` 特性系统

**依赖添加**:

在 workspace 根目录的 Cargo.toml 的 `[workspace.dependencies]` 中添加:
```toml
[workspace.dependencies]
# SIMD 核心库
wide = "0.7"
multiversion = "0.4"

# 特性检测（用于非 multiversion 场景）
cfg-if = "1.0"

# 编译时 CPU 特性检测（build.rs 中使用）
rustc_version = "0.4"
```

在 `ta-core` 的 Cargo.toml 的 `[dependencies]` 中添加:
```toml
[dependencies]
# SIMD 依赖
wide.workspace = true
multiversion.workspace = true
cfg-if.workspace = true

# 内部特性用于测试特定 SIMD 路径
[features]
default = ["f64"]
f32 = []
f64 = []
# 内部测试特性 - 强制使用特定 SIMD 路径
test-avx2 = []
test-avx512 = []
test-neon = []
test-scalar = []
```

**子任务**:

- [ ] 1.4.1 创建 `simd` 模块结构
  - 创建 `crates/ta-core/src/simd/mod.rs`
  - 创建 `crates/ta-core/src/simd/arch/` 子目录结构:
    ```
    simd/
    ├── mod.rs          # 公共 API 和 multiversion 分派
    ├── types.rs        # SIMD 类型别名和常量
    ├── scalar.rs       # 标量 fallback 实现
    └── arch/
        ├── x86_64.rs   # x86_64 通用代码
          ├── avx2.rs     # AVX2 实现
        ├── avx512.rs   # AVX-512 实现
        ├── aarch64.rs  # ARM64 NEON 实现
        └── wasm32.rs   # WASM SIMD128 实现
    ```
  - 添加模块文档，说明 Multi-ISSE 架构设计

- [ ] 1.4.2 定义 SIMD 类型和常量 (`simd/types.rs`)
  ```rust
  //! SIMD 类型定义和平台常量
  //! 
  //! 这些类型和常量在不同平台有不同的值，
  //! 但 API 保持一致。

  use crate::types::Float;

   /// 当前平台最优 SIMD  lanes 数量
   /// 
   /// - x86_64/AVX2: 4 (f64) / 8 (f32)
   /// - x86_64/AVX-512: 8 (f64) / 16 (f32)
   /// - ARM64/NEON: 2 (f64) / 4 (f32)
   /// - WASM: 2 (f64) / 4 (f32)
   pub const LANES: usize = 4; // 默认值，实际值由 multiversion 处理

  /// SIMD 向量类型别名（在 multiversion 函数中使用）
  #[cfg(feature = "f32")]
  pub type SimdVec = wide::f32x8; // AVX2 版本，其他平台自动适配

  #[cfg(feature = "f64")]
  pub type SimdVec = wide::f64x4; // AVX2 版本，其他平台自动适配

   /// 平台特性枚举
   #[derive(Debug, Clone, Copy, PartialEq, Eq)]
   pub enum SimdTarget {
       Scalar,      // 标量 fallback
       Avx2,        // x86_64 AVX2
       Avx512,      // x86_64 AVX-512
       Neon,        // ARM64 NEON
       Simd128,     // WASM SIMD128
   }

   /// 获取当前运行时的 SIMD 目标
   /// 
   /// 注意：这只是一个信息函数，实际分派由 multiversion 自动处理
   pub fn current_simd_target() -> SimdTarget {
       // 使用 target_feature 检测当前编译目标
       #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
       return SimdTarget::Avx512;
       
       #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
       return SimdTarget::Avx2;
       
       #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
       return SimdTarget::Neon;
       
       #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
       return SimdTarget::Simd128;
       
       #[cfg(not(any(
           all(target_arch = "x86_64", target_feature = "avx2"),
           all(target_arch = "aarch64", target_feature = "neon"),
           all(target_arch = "wasm32", target_feature = "simd128")
       )))]
       return SimdTarget::Scalar;
   }
  ```

- [ ] 1.4.3 实现多平台 SIMD 核心函数 (`simd/mod.rs`)
  
  这是 multiversion 架构的核心。每个函数使用 `#[multiversion]` 属性生成多个实现：
  
  ```rust
  //! SIMD 抽象层 - 公共 API
  //!
  //! 本模块提供平台无关的 SIMD 操作 API。
  //! 实际实现通过 multiversion 自动选择最优路径。

  use crate::types::Float;
  use crate::Result;

  pub mod types;
  mod scalar;
  
   #[cfg(target_arch = "x86_64")]
   mod arch {
       pub mod x86_64;
       pub mod avx2;
       #[cfg(feature = "test-avx512")]
       pub mod avx512;
   }
  
  #[cfg(target_arch = "aarch64")]
  mod arch {
      pub mod aarch64;
      pub mod neon;
  }
  
  #[cfg(target_arch = "wasm32")]
  mod arch {
      pub mod wasm32;
      pub mod simd128;
  }

   /// SIMD 加速数组求和
   ///
   /// 自动选择最优 SIMD 路径（AVX2, AVX-512, NEON, 或标量）
   #[multiversion(targets(
       "x86_64+avx2",
       "x86_64+avx512f",
       "aarch64+neon",
       "wasm32+simd128"
   ))]
   pub fn sum(data: &[Float]) -> Float {
      // 小数组优化：使用标量
      if data.len() < 16 {
          return scalar::sum(data);
      }
      
       // 使用 target-specific 实现
       #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
       return arch::avx2::sum(data);
       
       #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
       return arch::avx512::sum(data);

      
      #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
      return arch::neon::sum(data);
      
      #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
      return arch::simd128::sum(data);
      
      // Fallback
      scalar::sum(data)
  }

   /// SIMD 加速点积计算
   #[multiversion(targets(
       "x86_64+avx2",
       "x86_64+avx512f",
       "aarch64+neon",
       "wasm32+simd128"
   ))]
   pub fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
      if a.len() != b.len() {
          return Err(TalibError::InvalidInput("Arrays must have same length".into()));
      }
      
      // 实现类似 sum() 的多版本分派...
      // (详细实现省略，遵循相同模式)
      todo!()
  }

   /// SIMD 加速滚动窗口平均 (SMA 核心)
   #[multiversion(targets(
       "x86_64+avx2",
       "x86_64+avx512f",
       "aarch64+neon",
       "wasm32+simd128"
   ))]
   pub fn rolling_average(
      data: &[Float], 
      window: usize,
      output: &mut [Float]
  ) -> Result<usize> {
      if window == 0 || window > data.len() {
          return Err(TalibError::InvalidPeriod(window));
      }
      
      let output_len = data.len() - window + 1;
      if output.len() < output_len {
          return Err(TalibError::InvalidInput("Output buffer too small".into()));
      }
      
      // 小窗口使用标量实现
      if window < 8 || output_len < 16 {
          return scalar::rolling_average(data, window, output);
      }
      
      // 多版本 SIMD 实现...
      // (详细实现省略)
      todo!()
  }

  // 其他 SIMD 辅助函数...
  // - mul_add (乘加)
  // - horizontal operations (水平操作)
  // - load/store with alignment hints (对齐提示的加载/存储)
  ```

- [ ] 1.4.4 实现平台特定 SIMD 模块

   为每个目标架构创建具体实现：
   
   **x86_64/AVX2** (`simd/arch/avx2.rs`):
  ```rust
  //! AVX2 实现
  use crate::types::Float;
  use wide::f64x4;
  
  pub const LANES: usize = 4; // f64
  
  pub fn sum(data: &[Float]) -> Float {
      let chunks = data.chunks_exact(LANES);
      let remainder = chunks.remainder();
      
      let mut sum_vec = f64x4::ZERO;
      for chunk in chunks {
          let vec = f64x4::from_slice(chunk);
          sum_vec += vec;
      }
      
      let mut sum = sum_vec.horizontal_add();
      for &x in remainder {
          sum += x;
      }
      sum
  }
  
  // AVX2 可以处理更多 lanes，性能更好
  ```
  
  **ARM64/NEON** (`simd/arch/neon.rs`):
  ```rust
  //! ARM64 NEON 实现
  use crate::types::Float;
  use wide::f64x2;
  
  pub const LANES: usize = 2; // f64
  
  pub fn sum(data: &[Float]) -> Float {
       // NEON 实现（2 lanes，与 AVX2/AVX-512 适配）
      // wide crate 会自动处理平台差异
      let chunks = data.chunks_exact(LANES);
      let remainder = chunks.remainder();
      
      let mut sum_vec = f64x2::ZERO;
      for chunk in chunks {
          let vec = f64x2::from_slice(chunk);
          sum_vec += vec;
      }
      
      let mut sum = sum_vec.horizontal_add();
      for &x in remainder {
          sum += x;
      }
      sum
  }
  ```
  
  **WASM/SIMD128** (`simd/arch/simd128.rs`):
  ```rust
  //! WASM SIMD128 实现
  use crate::types::Float;
  use wide::f64x2;
  
  pub const LANES: usize = 2; // f64
  
   // 实现与 AVX2/AVX-512 适配
  // wide crate 会自动使用 WASM SIMD128 指令
  ```
  
  **标量 Fallback** (`simd/scalar.rs`):
  ```rust
  //! 标量 fallback 实现（纯 Rust，无 SIMD）
  //! 
  //! 当目标平台不支持任何 SIMD 指令集时使用
  
  use crate::types::Float;
  
  pub const LANES: usize = 1;
  
  pub fn sum(data: &[Float]) -> Float {
      data.iter().sum()
  }
  
  pub fn dot_product(a: &[Float], b: &[Float]) -> crate::Result<Float> {
      if a.len() != b.len() {
          return Err(crate::TalibError::InvalidInput(
              "Arrays must have same length".into()
          ));
      }
      Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
  }
  
  pub fn rolling_average(
      data: &[Float],
      window: usize,
      output: &mut [Float],
  ) -> crate::Result<usize> {
      if window == 0 || window > data.len() {
          return Err(crate::TalibError::InvalidPeriod(window));
      }
      
      let output_len = data.len() - window + 1;
      if output.len() < output_len {
          return Err(crate::TalibError::InvalidInput(
              "Output buffer too small".into()
          ));
      }
      
      // 计算第一个窗口的和
      let mut window_sum: Float = data[..window].iter().sum();
      output[0] = window_sum / window as Float;
      
      // 滚动计算
      for i in 1..output_len {
          window_sum += data[i + window - 1] - data[i - 1];
          output[i] = window_sum / window as Float;
      }
      
      Ok(output_len)
  }
  ```

- [ ] 1.4.5 实现 `build.rs` 构建脚本
  - 创建 `crates/ta-core/build.rs`
  - 检测编译时目标平台特性
  - 设置条件编译标志
  - 生成平台信息常量
  ```rust
  // build.rs
  use std::env;

  fn main() {
      let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
      let target_feature = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();

      println!("cargo:rustc-cfg=arch_{}", target_arch);

       // 根据目标特性设置条件编译标志
       if target_feature.contains("avx512f") {
           println!("cargo:rustc-cfg=simd_avx512");
       }
       if target_feature.contains("avx2") {
           println!("cargo:rustc-cfg=simd_avx2");
       }
       if target_feature.contains("neon") {
           println!("cargo:rustc-cfg=simd_neon");
       }
       if target_feature.contains("simd128") {
           println!("cargo:rustc-cfg=simd_simd128");
       }

      // 生成版本信息
      println!("cargo:rustc-env=BUILD_DATE={}", chrono::Utc::now().format("%Y-%m-%d"));
  }
  ```

- [ ] 1.4.6 编写单元测试（每个 SIMD 路径）
  - 为每个 SIMD 路径创建单元测试
  - 使用条件编译和内部特性强制测试特定路径
  - 验证所有路径结果一致性
  ```rust
  // tests/simd_tests.rs
  #![cfg(test)]

  use ta_core::simd;

  #[test]
  fn test_simd_sum_basic() {
      let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
      let result = simd::sum(&data);
      assert!((result - 15.0).abs() < 1e-10);
  }

  #[test]
  fn test_simd_sum_large_array() {
      let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
      let result = simd::sum(&data);
      let expected: f64 = (0..10000).map(|i| i as f64).sum();
      assert!((result - expected).abs() < 1e-6);
  }

  #[test]
  fn test_simd_dot_product() {
      let a = vec![1.0, 2.0, 3.0];
      let b = vec![4.0, 5.0, 6.0];
      let result = simd::dot_product(&a, &b).unwrap();
      assert!((result - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
  }

  #[test]
  fn test_simd_rolling_average() {
      let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
      let mut output = vec![0.0; data.len() - 3 + 1];
      
      let count = simd::rolling_average(&data, 3, &mut output).unwrap();
      
      assert_eq!(count, 6);
      assert!((output[0] - 2.0).abs() < 1e-10);  // (1+2+3)/3 = 2
      assert!((output[1] - 3.0).abs() < 1e-10);  // (2+3+4)/3 = 3
      assert!((output[5] - 7.0).abs() < 1e-10);  // (6+7+8)/3 = 7
  }

  // 特定 SIMD 路径测试（使用内部特性强制路径）
  #[cfg(feature = "test-scalar")]
  mod scalar_tests {
      use super::*;
      
      #[test]
      fn test_scalar_path() {
          // 强制使用标量路径
          let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
          let result = simd::scalar::sum(&data);
          assert!((result - 15.0).abs() < 1e-10);
      }
  }

  // 跨路径一致性测试
  #[test]
  fn test_cross_path_consistency() {
      let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
      
      // 所有 SIMD 路径应该产生相同结果（忽略浮点舍入误差）
      let result_simd = simd::sum(&data);
      let result_scalar = simd::scalar::sum(&data);
      
      // 允许小的浮点误差
      assert!((result_simd - result_scalar).abs() < 1e-6,
          "SIMD and scalar results differ: simd={}, scalar={}",
          result_simd, result_scalar
      );
  }
  ```

- [ ] 1.4.7 性能基准测试
  - 为每个 SIMD 路径创建 Criterion 基准测试
  - 比较不同 SIMD 路径的性能
  - 生成性能报告
  ```rust
  // benches/simd_benchmarks.rs
  use criterion::{black_box, criterion_group, criterion_main, Criterion};
  use ta_core::simd;

  fn benchmark_simd_sum(c: &mut Criterion) {
      let sizes = [100, 1000, 10000, 100000];
      
      for size in sizes {
          let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
          
          c.bench_function(&format!("simd_sum_{}", size), |b| {
              b.iter(|| {
                  let result = simd::sum(black_box(&data));
                  black_box(result);
              })
          });
          
          c.bench_function(&format!("scalar_sum_{}", size), |b| {
              b.iter(|| {
                  let result = simd::scalar::sum(black_box(&data));
                  black_box(result);
              })
          });
      }
  }

  criterion_group!(benches, benchmark_simd_sum);
  criterion_main!(benches);
  ```

**验收标准**:
- [ ] `multiversion` 和 `wide` 依赖正确配置
- [ ] `simd/mod.rs` 公共 API 实现完成
- [ ] `simd/types.rs` 类型定义完成
   - [ ] **所有平台特定模块实现完成**:
   - [ ] `simd/arch/avx2.rs` (x86_64 AVX2)
   - [ ] `simd/arch/avx512.rs` (x86_64 AVX-512) - 可选
   - [ ] `simd/arch/neon.rs` (ARM64 NEON)
   - [ ] `simd/arch/simd128.rs` (WASM SIMD128)
   - [ ] `simd/scalar.rs` (标量 fallback)
   - [ ] `build.rs` 构建脚本实现
   - [ ] **所有 SIMD 路径通过单元测试**:
   - [ ] AVX2 路径测试
   - [ ] AVX-512 路径测试（如可用）
   - [ ] NEON 路径测试
   - [ ] SIMD128 路径测试
   - [ ] 标量 fallback 测试
- [ ] **跨路径一致性验证通过**（所有路径产生相同结果）
- [ ] 性能基准测试完成
- [ ] **性能要求**: 相比标量实现加速比 > 2x（数据量 > 1000）
- [ ] 代码覆盖率 > 95%

**交付物**:
- `crates/ta-core/build.rs`
- `crates/ta-core/src/simd/mod.rs`
   - `crates/ta-core/src/simd/types.rs`
   - `crates/ta-core/src/simd/scalar.rs`
   - `crates/ta-core/src/simd/arch/x86_64.rs`
   - `crates/ta-core/src/simd/arch/avx2.rs`
   - `crates/ta-core/src/simd/arch/avx512.rs` (可选)
   - `crates/ta-core/src/simd/arch/aarch64.rs`
   - `crates/ta-core/src/simd/arch/neon.rs`
   - `crates/ta-core/src/simd/arch/wasm32.rs`
   - `crates/ta-core/src/simd/arch/simd128.rs`
- 单元测试文件
- 性能基准测试 (`benches/simd_benchmarks.rs`)
- 使用文档

**依赖**: 任务 1.1 (workspace 结构) - ✅ 已完成  
**被依赖**: 任务 1.3 (核心 Traits - 需要 SIMD 层), 任务 2.1+ (所有指标实现)  
**可并行**: 无（这是基础依赖）  
**风险**: **高**  
**风险缓解**: 
 - 分阶段实现：先完成 AVX2/AVX-512，再添加 NEON/SIMD128
- 充分的单元测试确保各路径正确性
- 性能基准验证加速效果

---

### 任务 1.5: 设置测试基础设施 (修订版)

**任务 ID**: 1.5  
**任务名称**: 设置测试基础设施和工具  
**优先级**: P1 (高)  
**预估工时**: 8 小时 **(+2 小时 用于多平台测试配置)**  
**负责人**: TBD  
**状态**: ⬜ 待开始  

**描述**:  
配置测试框架、工具和 CI/CD 管道，确保代码质量和测试覆盖率。**新增：配置多平台测试，确保所有 SIMD 路径得到测试。**

**子任务更新**:

- [ ] 1.5.1 配置测试框架  
  *(保持原有内容)*

- [ ] 1.5.2 配置代码质量工具  
  *(保持原有内容)*

- [ ] 1.5.3 创建 CI/CD 配置 **(重大更新)**
  - 创建 `.github/workflows/ci.yml`（GitHub Actions）
  - **配置测试矩阵，覆盖所有 SIMD 路径**：
    ```yaml
    strategy:
      matrix:
        include:
          # x86_64 platforms
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            rust: stable
             flags: ""  # Default (AVX2 baseline)
            
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            rust: stable
            flags: "-C target-feature=+avx2"
            test_feature: "test-avx2"
            
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            rust: nightly  # AVX-512 requires nightly sometimes
            flags: "-C target-feature=+avx512f,+avx512vl"
            test_feature: "test-avx512"
            
          # ARM64 platforms
          - target: aarch64-unknown-linux-gnu
            os: ubuntu-latest
            rust: stable
            flags: ""
            test_feature: "test-neon"
            
          - target: aarch64-apple-darwin
            os: macos-latest  # Apple Silicon
            rust: stable
            flags: ""
            test_feature: "test-neon"
            
          # WASM
          - target: wasm32-unknown-unknown
            os: ubuntu-latest
            rust: stable
            flags: ""
            test_feature: "test-simd128"
            
          # Scalar fallback testing
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            rust: stable
            flags: "-C target-feature=-sse2"  # Disable SSE2
            test_feature: "test-scalar"
    ```
  - 添加步骤：
    - 格式检查（`rustfmt -- --check`）
    - Clippy 检查
    - **针对每种 SIMD 路径的编译检查**（`RUSTFLAGS="..." cargo build`）
    - **针对每种 SIMD 路径的单元测试**（`RUSTFLAGS="..." cargo test`）
    - 代码覆盖率（`cargo tarpaulin`）
    - **跨平台一致性测试**（验证不同路径结果一致）

- [ ] 1.5.4 配置性能基准测试  
  *(保持原有内容)*  
  **新增**: 
  - 创建 SIMD 路径性能对比基准
  - 生成跨平台性能报告

- [ ] 1.5.5 创建测试辅助模块  
  *(保持原有内容)*  
  **新增**:
  - 创建 SIMD 路径测试辅助函数
  - 创建跨平台结果一致性验证函数

**验收标准**:
- [ ] 测试框架配置完成
- [ ] 代码覆盖率工具可以正常运行
- [ ] CI/CD 管道配置完成 **（包含多平台矩阵测试）**
- [ ] 所有 lint 规则配置完成
- [ ] 基准测试可以运行 **（包含 SIMD 路径对比）**
- [ ] 测试辅助模块完成 **（包含 SIMD 测试辅助）**
- [ ] **所有 SIMD 路径在 CI 中通过测试**

**交付物**:
- `.github/workflows/ci.yml` **（更新 - 多平台矩阵）**
- `.rustfmt.toml`
- `Cargo.toml`（更新测试依赖和 lint 配置）
- `crates/ta-core/src/test_utils.rs`
- **CI/CD 配置文档（包含多平台测试说明）**

**依赖**: 任务 1.1 (workspace 结构), 任务 1.2 (错误类型系统), **任务 1.4 (Multi-ISSE SIMD - 需要其测试)**  
**可并行**: 可以与任务 1.6 和 1.7 并行  
**风险**: **中**（多平台 CI 配置复杂）  
**风险缓解**: 
- 使用 GitHub Actions 的矩阵功能简化配置
- 分阶段启用平台：先 x86_64，再 ARM64，最后 WASM
- 使用 Docker 标准化测试环境

---

### 任务 1.6: 构建配置与平台支持 (新增)

**任务 ID**: 1.6  
**任务名称**: 配置构建系统与平台支持（build.rs, Cargo features）  
**优先级**: P1 (高)  
**预估工时**: 8 小时  
**负责人**: TBD  
**状态**: ⬜ 新增  

**描述**:  
配置构建脚本和 Cargo 特性，实现编译时平台检测、条件编译和多平台支持。

**子任务**:

- [ ] 1.6.1 实现 `build.rs` 构建脚本
  - 检测目标平台架构
  - 检测可用的 SIMD 特性
  - 设置条件编译标志
  - 生成平台信息常量
  - 打印构建信息
  - 详细实现见任务 1.4.5

- [ ] 1.6.2 配置 Cargo.toml features
  - 更新 `ta-core/Cargo.toml`:
    ```toml
    [features]
    default = ["f64"]
    
    # 浮点精度选择（保持向后兼容）
    f32 = []
    f64 = []
    
    # 内部测试特性 - 强制使用特定 SIMD 路径
    # 这些特性仅用于测试，不应在发布版本中使用
    _test_avx2 = []
    _test_avx512 = []
    _test_neon = []
    _test_simd128 = []
    _test_scalar = []
    
    # 文档生成特性
    docs = []
    ```

- [ ] 1.6.3 创建平台检测模块
  - 创建 `crates/ta-core/src/platform.rs`
  - 实现运行时平台信息查询
  - 提供 CPU 特性检测（使用 `std::is_x86_feature_detected!` 等）
  ```rust
  //! 平台检测和运行时信息
  
  /// 运行时平台信息
  #[derive(Debug, Clone)]
  pub struct PlatformInfo {
      pub target_arch: &'static str,
      pub target_os: &'static str,
      pub simd_target: SimdTarget,
      pub available_features: Vec<String>,
  }
  
  /// SIMD 目标枚举
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum SimdTarget {
       Scalar,
       Avx2,
       Avx512,
       Neon,
       Simd128,
  }
  
  impl PlatformInfo {
      /// 获取当前运行时平台信息
      pub fn current() -> Self {
          Self {
              target_arch: env!("CARGO_CFG_TARGET_ARCH"),
              target_os: env!("CARGO_CFG_TARGET_OS"),
              simd_target: detect_simd_target(),
              available_features: detect_available_features(),
          }
      }
  }
  
  fn detect_simd_target() -> SimdTarget {
      #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
      return SimdTarget::Avx512;
      
      #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
      return SimdTarget::Avx2;
      

      #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
      return SimdTarget::Neon;
      
      #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
      return SimdTarget::Simd128;
      
      SimdTarget::Scalar
  }
  
  fn detect_available_features() -> Vec<String> {
      let mut features = vec![];
      
       #[cfg(target_arch = "x86_64")]
       {
           if std::is_x86_feature_detected!("avx512f") { features.push("avx512f".into()); }
           if std::is_x86_feature_detected!("avx2") { features.push("avx2".into()); }
       }
      
      #[cfg(target_arch = "aarch64")]
      {
          if std::arch::is_aarch64_feature_detected!("neon") {
              features.push("neon".into());
          }
      }
      
      features
  }
  ```

- [ ] 1.6.4 创建构建配置文档
  - 编写 `docs/build-configuration.md`
  - 说明 Cargo features 使用方法
  - 说明跨平台编译配置
  - 说明性能调优选项

**验收标准**:
- [ ] `build.rs` 构建脚本实现完成
- [ ] `Cargo.toml` features 配置完成
- [ ] 平台检测模块实现完成
- [ ] 构建配置文档完成
- [ ] **所有目标平台可以成功编译**:
   - [ ] x86_64-unknown-linux-gnu (AVX2/AVX-512)
  - [ ] x86_64-pc-windows-msvc
  - [ ] x86_64-apple-darwin
  - [ ] aarch64-unknown-linux-gnu
  - [ ] aarch64-apple-darwin (Apple Silicon)
  - [ ] wasm32-unknown-unknown
- [ ] **运行时平台检测工作正常**

**交付物**:
- `crates/ta-core/build.rs`
- `crates/ta-core/Cargo.toml` (更新 features)
- `crates/ta-core/src/platform.rs`
- `docs/build-configuration.md`

**依赖**: 任务 1.1 (workspace 结构)  
**被依赖**: 任务 1.4 (SIMD 抽象层), 任务 1.7 (跨平台测试)  
**可并行**: 可以与任务 1.5 部分并行  
**风险**: 中（跨平台配置复杂）  
**风险缓解**:
- 使用 cross 工具简化交叉编译
- 使用 Docker 标准化环境
- 优先支持主要平台（x86_64 Linux/macOS/Windows）

---

### 任务 1.7: 跨平台测试基础设施 (新增)

**任务 ID**: 1.7  
**任务名称**: 设置跨平台测试基础设施（Docker, QEMU, 模拟器）  
**优先级**: P1 (高)  
**预估工时**: 12 小时  
**负责人**: TBD  
**状态**: ⬜ 新增  

**描述**:  
配置跨平台测试环境，使用 Docker、QEMU 和模拟器在开发机器上测试所有目标平台，无需物理硬件。

**子任务**:

- [ ] 1.7.1 配置 Docker 跨平台测试环境
  - 创建 `docker/` 目录
  - 创建 `docker/Dockerfile.test-x86_64` (x86_64 测试)
  - 创建 `docker/Dockerfile.test-aarch64` (ARM64 交叉测试)
  - 创建 `docker/docker-compose.test.yml`
  - 创建 `docker/test-all-platforms.sh` 脚本
  ```dockerfile
  # docker/Dockerfile.test-aarch64
  FROM rust:1.78-bookworm
  
  # Install cross-compilation tools
  RUN apt-get update && apt-get install -y \
      qemu-user \
      qemu-system-arm \
      gcc-aarch64-linux-gnu \
      libc6-dev-arm64-cross \
      && rm -rf /var/lib/apt/lists/*
  
  # Install cross
  RUN cargo install cross --version 0.2.5
  
  # Add ARM64 target
  RUN rustup target add aarch64-unknown-linux-gnu
  
  WORKDIR /workspace
  ```

- [ ] 1.7.2 配置 QEMU 模拟器测试
  - 创建 `scripts/test-with-qemu.sh`
  - 配置 QEMU 用户模式模拟
  - 测试 ARM64 和 WASM 目标
  ```bash
  #!/bin/bash
  # scripts/test-with-qemu.sh
  
  set -e
  
  echo "Testing with QEMU emulation..."
  
  # Test ARM64
  echo "Testing aarch64-unknown-linux-gnu..."
  cargo test --target aarch64-unknown-linux-gnu \
      --config 'target.aarch64-unknown-linux-gnu.runner="qemu-aarch64"'
  
  # Test WASM (using wasmtime)
  echo "Testing wasm32-unknown-unknown..."
  cargo test --target wasm32-wasi
  
  echo "QEMU tests complete!"
  ```

- [ ] 1.7.3 配置 WASM 测试环境
  - 安装 `wasmtime` 或 `wasmer` WASM 运行时
  - 配置 `cargo wasi` 工具
  - 创建 WASM 特定测试

- [ ] 1.7.4 创建跨平台测试脚本
  - 创建 `scripts/test-all-platforms.sh` (主测试脚本)
  - 创建 `scripts/test-x86_64.sh` (x86_64 专用)
  - 创建 `scripts/test-aarch64.sh` (ARM64 专用)
  - 创建 `scripts/test-wasm.sh` (WASM 专用)
  ```bash
  #!/bin/bash
  # scripts/test-all-platforms.sh
  
  set -e
  
  echo "================================"
  echo "Running Cross-Platform Tests"
  echo "================================"
  
  # Native platform (current host)
  echo ""
  echo "1. Testing native platform..."
  cargo test --all-features
  
  # x86_64 with different SIMD levels
  if command -v docker &> /dev/null; then
      echo ""
      echo "2. Testing x86_64 (Docker)..."
      docker-compose -f docker/docker-compose.test.yml run test-x86_64
      
      echo ""
      echo "3. Testing aarch64 (Docker+QEMU)..."
      docker-compose -f docker/docker-compose.test.yml run test-aarch64
  else
      echo "Docker not available, skipping containerized tests"
  fi
  
  # WASM
  if command -v wasmtime &> /dev/null; then
      echo ""
      echo "4. Testing WASM..."
      ./scripts/test-wasm.sh
  else
      echo "wasmtime not available, skipping WASM tests"
  fi
  
  echo ""
  echo "================================"
  echo "Cross-Platform Tests Complete!"
  echo "================================"
  ```

- [ ] 1.7.5 创建跨平台测试文档
  - 编写 `docs/cross-platform-testing.md`
  - 说明如何在本地运行跨平台测试
  - 说明如何在 CI 中运行
  - 说明如何添加新的目标平台

**验收标准**:
- [ ] Docker 测试环境配置完成（x86_64, ARM64）
- [ ] QEMU 配置完成并可以运行 ARM64 测试
- [ ] WASM 测试环境配置完成
- [ ] 所有跨平台测试脚本完成
- [ ] **至少可以在以下平台运行测试**:
  - [ ] x86_64 Linux (原生)
  - [ ] x86_64 Linux (Docker)
  - [ ] ARM64 Linux (Docker + QEMU)
  - [ ] WASM32 (wasmtime)
- [ ] 跨平台测试文档完成

**交付物**:
- `docker/Dockerfile.test-x86_64`
- `docker/Dockerfile.test-aarch64`
- `docker/docker-compose.test.yml`
- `scripts/test-all-platforms.sh`
- `scripts/test-x86_64.sh`
- `scripts/test-aarch64.sh`
- `scripts/test-wasm.sh`
- `scripts/test-with-qemu.sh`
- `docs/cross-platform-testing.md`

**依赖**: 任务 1.1 (workspace 结构), 任务 1.4 (SIMD 抽象层 - 需要其完成才能测试)  
**被依赖**: 任务 6.1 (全面测试 - 依赖跨平台测试基础设施)  
**可并行**: 可以与任务 1.5 和 1.6 部分并行  
**风险**: 中（QEMU 和 Docker 配置复杂）  
**风险缓解**:
- 使用成熟工具（cross, cargo-bisect-rustc）
- 提供详细的故障排除指南
- 允许部分平台测试失败（非阻塞）

---

## 附录

### A. 修订记录

#### v2.0 (2026-01-30) - Multi-ISSE SIMD 重大修订

**变更摘要**:
1. **架构升级**: 从 `wide` 编译时单一目标 → `multiversion` + `wide` 运行时多指令集动态分派
2. **平台支持**: 新增 ARM64 (NEON)、WASM (SIMD128) 全面支持
3. **任务结构调整**:
   - Phase 1 持续时间：4周 → 6周 (+2周)
   - 总工期：26周 → 28周 (+2周)
   - 任务数：5 → 7 个任务
4. **新增任务**:
   - 任务 1.6: 构建配置与平台支持
   - 任务 1.7: 跨平台测试基础设施
5. **修订任务**:
   - 任务 1.3: 添加 `SimdCompute` trait 集成
   - 任务 1.4: 完全重写 - Multi-ISSE 架构
   - 任务 1.5: 添加多平台 CI 矩阵
6. **新增架构文档**:
   - ADR-001: SIMD Implementation Strategy
   - ADR-002: Platform Support Matrix
   - Multi-ISSE SIMD Architecture 图表

**影响评估**:
- **风险**: 中等偏高（架构复杂度高）
- **收益**: 极高（单二进制多平台最优性能）
- **工作量**: +2周，主要集中在任务 1.4、1.6、1.7
- **兼容性**: 100% 向后兼容，无破坏性变更

**迁移指南**:
- 现有代码（任务 1.1、1.2）无需变更
- 任务 1.3、1.4、1.5 需要按修订版实施
- 新增任务 1.6、1.7 需要全新实施

---

### B. 术语表

- **Multi-ISSE**: Multiple Instruction Set Extensions - 多指令集扩展
- **SIMD**: Single Instruction Multiple Data - 单指令多数据
- **Runtime Dispatch**: 运行时动态分派 - 根据 CPU 特性选择最优代码路径
- **Target Feature**: 目标特性 - 编译目标支持的 CPU 特性
- **Feature Detection**: 特性检测 - 运行时检测 CPU 支持的特性
- **Fallback**: 回退 - 当高级特性不可用时使用的基础实现
- **Cross-compilation**: 交叉编译 - 在一种平台上编译另一种平台的代码
- **QEMU**: 快速模拟器 - 用于在非原生硬件上运行程序
- **NEON**: ARM 的高级 SIMD 架构
- **AVX-512**: Intel 的 512-bit SIMD 指令集

---

**文档结束**
