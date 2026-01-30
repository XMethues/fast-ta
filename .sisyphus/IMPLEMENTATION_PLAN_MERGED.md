# TA-Lib Rust 重写项目 - 完整实施计划 (合并版)

**版本**: v3.0 (Unified Implementation Plan)
**创建日期**: 2026-01-30
**最后更新**: 2026-01-30
**状态**: 准备就绪

---

## 重要说明

**本计划为 v3.0 合并版**，整合了：
1. **完整的项目范围** (来自 IMPLEMENTATION_PLAN.md v2.0)
2. **详细的 SIMD 手动调度实施步骤** (来自 simd-implementation.md)

**关键架构决策（统一）**：
- ❌ **不使用** multiversion crate
- ✅ **手动实现**函数指针调度系统（OnceLock + std::is_x86_feature_detected!）
- ✅ **底层使用** wide crate 提供稳定的 SIMD 原语
- ✅ **支持多平台**：x86_64 (AVX2/AVX-512), ARM64 (NEON), WASM (SIMD128)
- ✅ **单二进制**：运行时动态选择最优 SIMD 路径

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
- **多指令集 SIMD 优化**（手动调度 + wide）
- 批量与流式双模式
- Python 和 WASM 绑定

### 成功标准

- 100% 数值精度匹配 TA-Lib C（ε < 1e-10）
- 所有 150+ 指标实现
- **SIMD 优化在 x86_64 (AVX2/AVX-512)、ARM64 (NEON)、WASM (SIMD128) 上有效**
- 流式 API 延迟 < 1ms
- **跨平台测试覆盖率 > 90%**

### 关键技术决策

**SIMD 架构（手动调度）**:
- **运行时多指令集分派**: 手动实现运行时 CPU 特性检测和最优代码路径选择
  - 使用 `std::is_x86_feature_detected!` 检测 x86_64 特性
  - 使用 `std::arch::is_aarch64_feature_detected!` 检测 ARM64 特性
  - 使用 `OnceLock` 缓存检测结果和函数指针
  - 启动后零运行时开销（直接函数指针调用）
- **底层 SIMD 实现**: **统一使用 `wide` crate 提供稳定的跨平台 SIMD 原语**
  - `wide` 已支持 AVX-512 (`wide::f64x8` / `wide::f32x16`)，无需使用 `std::arch` intrinsics
  - **关键优势**: 代码更简洁，类型更安全，编译器自动优化
- **平台支持矩阵**:
- **平台支持矩阵**:

| 平台 | 指令集 | 实现方式 | Lanes (f64) | Lanes (f32) |
|------|--------|----------|-------------|-------------|
| x86_64 | AVX2 | `wide::f64x4` / `f32x8` | 4 | 8 |
| x86_64 | AVX-512 | `wide::f64x8` / `f32x16` (wide 已支持) | 8 | 16 |
| ARM64 | NEON | `wide::f64x2` / `f32x4` | 2 | 4 |
| WASM | SIMD128 | `wide::f64x2` / `f32x4` | 2 | 4 |
| 通用 | Scalar | 纯 Rust 标量实现 | 1 | 1 |

- **向后兼容**: 完全保持现有的 `f32`/`f64` 特性系统

**架构**: 4-crate workspace（core, py, wasm, benchmarks）
**API**: 零拷贝（`&[Float]` / `&mut [Float]`），`Float` 类型通过条件编译确定
**指标接口**: 统一 `Indicator` trait，采用混合方案（性能 + 易用性）
- `compute(&self, inputs, outputs)` - **零拷贝批量计算**（性能优先，无内存分配）
- `compute_to_vec(&self, inputs)` - **便捷批量计算**（易用性优先，返回 Vec）
- `next(&mut self, input)` - 获取最新值（实时流式）
- `stream(&mut self, inputs)` - 流式处理多个输入

---

## 架构设计

### SIMD 抽象层架构（手动调度）

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
│  │  pub fn sum(data: &[Float]) -> Float {                           │   │
│  │      // Runtime dispatch via function pointer                      │   │
│  │      dispatch::sum(data)                                          │   │
│  │  }                                                              │   │
│  │  pub fn dot_product(a: &[Float], b: &[Float]) -> Float {          │   │
│  │      dispatch::dot_product(a, b)                                  │   │
│  │  }                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            │                                           │
│  ┌─────────────────────────▼──────────────────────────────────────┐   │
│  │  simd/dispatch.rs - Runtime Dispatch System                       │   │
│  │                                                                  │   │
│  │  static DISPATCH_TABLE: OnceLock<DispatchTable> = OnceLock::new();│   │
│  │                                                                  │   │
│  │  fn init_dispatch() {                                             │   │
│  │      #[cfg(target_arch = "x86_64")] {                            │   │
│  │          if is_x86_feature_detected!("avx512f") {                 │   │
│  │              set_avx512_functions();                               │   │
│  │          } else if is_x86_feature_detected!("avx2") {              │   │
│  │              set_avx2_functions();                                │   │
│  │          } else {                                                │   │
│  │              set_scalar_functions();                               │   │
│  │          }                                                       │   │
│  │      }                                                           │   │
│  │      #[cfg(target_arch = "aarch64")] {                           │   │
│  │          set_neon_functions();                                    │   │
│  │      }                                                           │   │
│  │      #[cfg(target_arch = "wasm32")] {                            │   │
│  │          set_simd128_functions();                                 │   │
│  │      }                                                           │   │
│  │  }                                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                            │                                           │
└────────────────────────────┼───────────────────────────────────────────┘
                              │
┌────────────────────────────┼───────────────────────────────────────────┐
│              Platform-Specific Implementations                          │
│                            │                                           │
│  ┌───────────────────┐    │    ┌───────────────────┐                   │
│  │ x86_64/AVX512    │    │    │ x86_64/AVX2       │                   │
│  │ std::arch        │    │    │ wide::f64x4      │                   │
│  └─────────┬─────────┘    │    └─────────┬─────────┘                   │
│            │              │              │                              │
│  ┌─────────▼─────────┐    │    ┌─────────▼─────────┐                   │
│  │ ARM64/NEON        │    │    │ WASM/SIMD128      │                   │
│  │ wide::f64x2      │    │    │ wide::f64x2      │                   │
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

### 文件结构

```
crates/ta-core/src/
├── lib.rs              # 库入口
├── error.rs            # 错误类型 (已完成 ✅)
├── types.rs            # Float 类型别名 (已完成 ✅)
├── traits.rs           # Indicator trait (待实现 ⬜)
├── overlap/
│   ├── mod.rs         # 重叠指标模块
│   └── sma.rs        # SMA 实现 (需更新 ⬜)
└── simd/
    ├── mod.rs         # 公共 API
    ├── dispatch.rs    # 运行时调度系统
    ├── types.rs       # SIMD 类型和常量
    ├── scalar.rs      # 标量 fallback
    └── arch/
        ├── x86_64/
        │   ├── mod.rs
        │   ├── avx2.rs     # AVX2 实现 (wide::f64x4/f32x8)
        │   └── avx512.rs    # AVX-512 实现 (wide::f64x8/f32x16)
        ├── aarch64/
        │   ├── mod.rs
        │   └── neon.rs     # ARM64 NEON 实现 (wide::f64x2/f32x4)
        └── wasm32/
            ├── mod.rs
            └── simd128.rs    # WASM SIMD128 实现 (wide::f64x2/f32x4)
```

### Architecture Decision Records

#### ADR-001: SIMD Implementation Strategy

**Status**: Accepted (v3.0 Unified)

**Context**:
需要高性能的 SIMD 加速，同时支持多平台和运行时动态选择最优指令集。

**Decision**:
**统一使用 `wide` crate** 进行所有 SIMD 实现（包括 AVX-512）。`wide` crate 已经支持 AVX-512 (f64x8/f32x16)，提供统一的 API。结合 `std::is_x86_feature_detected!` 运行时特性检测和 `OnceLock` 函数指针调度。

**关键优势**:
1. **wide 已支持 AVX-512**: `wide::f64x8` (8 lanes for f64) 和 `wide::f32x16` (16 lanes for f32) 可直接使用
2. **代码更简洁**: 无需条件编译的 `std::arch` intrinsics，统一使用 wide API
3. **类型更安全**: 通过条件编译的类型别名 `SimdVec` 自动选择正确的 wide 类型
4. **编译器优化**: wide crate 内部已优化，编译器可以更好地优化代码

**Rationale**:
1. **统一 API**: 所有平台（AVX2, AVX-512, NEON, WASM）都使用相同的 wide API
2. **条件编译**: 通过 `#[cfg(feature = "f32")]` 和 `#[cfg(feature = "f64")]` 选择正确的 wide 类型别名
3. **运行时调度**: 仍然使用 `std::is_x86_feature_detected!` 检测 CPU 特性并选择最优实现
4. **维护性提升**: 代码更少，逻辑更清晰，更容易理解和扩展

**Rationale**:
1. **Runtime Dispatch**: 手动调度确保最大灵活性和性能
2. **跨平台**: 支持 x86_64, ARM64, WASM 而无需编译时标志
3. **零开销**: `OnceLock` 缓存函数指针，启动后无调度开销
4. **可维护性**: 清晰的架构分离，易于添加新平台

**Consequences**:
- **Positive**:
  - 单二进制在所有平台最优运行
  - 不需要特性标志或编译时配置
  - 易于测试和调试
- **Negative**:
  - 代码量增加（多实现）
  - 编译时间增长
  - 需要更全面的测试

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

**总工期**: 28 周（约 7 个月）

**关键路径**: Phase 1 → Phase 2 → Phase 6

**可并行**: Phase 2, 3, 4 可以并行执行

---

## Phase 1: 核心基础设施

**阶段名称**: Core Infrastructure
**持续时间**: 6 周（Weeks 1-6）
**目标**: 建立项目基础架构，实现核心 traits 和 **Multi-ISSE SIMD 抽象层**
**依赖**: 无

**阶段进度**:

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 任务 1.1: 创建 Workspace 结构 | ✅ 已完成 | 2026-01-29 | 基础结构已就绪 |
| 任务 1.2: 实现错误类型系统 | ✅ 已完成 | 2026-01-29 | 19 个测试通过 |
| 任务 1.3: SIMD 模块基础框架 | ✅ 已完成 | 2026-01-30 | 44 个测试通过 (包括 SIMD traits, types, scalar 实现) |
| 任务 1.4: SIMD 运行时调度系统 | ✅ 已完成 | 2026-01-30 | dispatch.rs 完成, OnceLock 调度系统实现 |
<<<<<<< HEAD
| 任务 1.5: x86_64 SIMD 实现 | ✅ 已完成 | 2026-01-30 | AVX2/AVX-512 实现, 调度系统集成 |
| 任务 1.6: ARM64 SIMD 实现 | ⬜ 待开始 | - | 8 小时预估 |
=======
| 任务 1.5: x86_64 SIMD 实现 | ⬜ 待开始 | - | 16 小时预估 |
| 任务 1.6: ARM64 SIMD 实现 | ✅ 已完成 | 2026-01-30 | NEON 路径实现完成, 单元测试通过 |
>>>>>>> vk/4dd6-1-6-arm64-simd
| 任务 1.7: WASM SIMD 实现 | ⬜ 待开始 | - | 6 小时预估 |
| 任务 1.8: 实现 Core Traits | ⬜ 待开始 | - | 12 小时预估 |
| 任务 1.9: 测试基础设施 | ⬜ 待开始 | - | 8 小时预估 |
| 任务 1.10: 构建配置与平台支持 | ⬜ 待开始 | - | 8 小时预估 |
| 任务 1.11: 跨平台测试基础设施 | ⬜ 待开始 | - | 12 小时预估 |

**总体进度**: 5/11 任务完成 (45%)

---

### 任务 1.1: 创建 Workspace 结构 ✅

**已完成** - 保持现有结构不变

---

### 任务 1.2: 实现错误类型系统 ✅

**已完成** - error.rs 已完整实现并通过 19 个测试

---

### 任务 1.3: SIMD 模块基础框架

**任务 ID**: 1.3
**任务名称**: 创建 SIMD 模块基础结构（mod.rs, types.rs, scalar.rs）
**优先级**: P0 (最高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
建立 SIMD 模块的基础结构，包括公共 API、类型定义和标量回退实现。

**子任务**:

#### 1.3.1 创建 `simd/mod.rs`

- [ ] 创建 `crates/ta-core/src/simd/mod.rs`
- [ ] 定义 `SimdLevel` 枚举（Scalar, Avx2, Avx512, Neon, Simd128）
- [ ] 实现 `detect_simd_level()` 函数（使用条件编译）
- [ ] 定义公共 API 函数签名（sum, dot_product 等）
- [ ] 添加模块级文档

```rust
//! SIMD 抽象层 - 运行时手动调度
//!
//! 本模块提供平台无关的 SIMD 操作 API。
//! 实际实现通过运行时函数指针调度自动选择最优路径。

use crate::types::Float;
use crate::Result;

pub mod dispatch;
pub mod scalar;
pub mod types;

#[cfg(target_arch = "x86_64")]
pub mod arch {
    pub mod x86_64;
}

#[cfg(target_arch = "aarch64")]
pub mod arch {
    pub mod aarch64;
}

#[cfg(target_arch = "wasm32")]
pub mod arch {
    pub mod wasm32;
}

/// SIMD 加速数组求和
///
/// 自动选择最优 SIMD 路径（AVX2, AVX-512, NEON, SIMD128 或标量）
///
/// # Example
///
/// ```rust
/// use ta_core::simd;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sum = simd::sum(&data);
/// assert_eq!(sum, 15.0);
/// ```
pub fn sum(data: &[Float]) -> Float {
    dispatch::sum(data)
}

/// SIMD 加速点积计算
///
/// 自动选择最优 SIMD 路径
pub fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    dispatch::dot_product(a, b)
}
```

#### 1.3.2 创建 `simd/types.rs`

- [ ] 定义 SIMD 相关的公共类型和常量
- [ ] 定义平台特定的 lane 数量常量
- [ ] 添加文档说明各平台的 lane 数量

```rust
//! SIMD 类型定义和平台常量
//!
//! 这些类型和常量在不同平台有不同的值，
//! 但 API 保持一致。
//!
//! **关键设计**: 所有 SIMD 实现必须使用 `SimdVec` 类型别名，
//! 而不是直接使用 `f64x4` 或 `f32x8` 等硬编码类型。
//! 这样可以确保通过 Cargo feature 正确切换 f32/f64。

use crate::types::Float;

/// SIMD 目标级别枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// 标量回退（无 SIMD）
    Scalar,
    /// x86_64 AVX2 (256-bit)
    Avx2,
    /// x86_64 AVX-512 (512-bit)
    Avx512,
    /// ARM64 NEON (128-bit)
    Neon,
    /// WASM SIMD128 (128-bit)
    Simd128,
}

/// 获取当前平台的 SIMD 级别（编译时）
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub const fn current_simd_level() -> SimdLevel {
    SimdLevel::Avx512
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
pub const fn current_simd_level() -> SimdLevel {
    SimdLevel::Avx2
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub const fn current_simd_level() -> SimdLevel {
    SimdLevel::Neon
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub const fn current_simd_level() -> SimdLevel {
    SimdLevel::Simd128
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub const fn current_simd_level() -> SimdLevel {
    SimdLevel::Scalar
}

// ============================================================================
// 条件编译的 SIMD 向量类型别名（根据 Float 配置自动选择）
// ============================================================================
//
// **重要**: wide crate 已经支持 AVX-512，无需使用 std::arch intrinsics。
// 统一使用 wide crate 提供的类型。

/// SIMD 向量类型别名（根据 Float 配置自动选择 f32/f64 对应的类型）
///
/// **使用规则**: 所有 SIMD 实现必须使用 `SimdVec` 类型别名，
/// 绝对不要直接使用 `wide::f64x4` 或 `wide::f32x8` 等硬编码类型。
///
/// | Float | AVX2 | AVX-512 | NEON | WASM | 标量 |
/// |-------|------|---------|------|------|--------|
/// | f64   | f64x4 (4 lanes) | f64x8 (8 lanes) | f64x2 (2 lanes) | f64x2 (2 lanes) | N/A |
/// | f32   | f32x8 (8 lanes) | f32x16 (16 lanes) | f32x4 (4 lanes) | f32x4 (4 lanes) | N/A |
#[cfg(feature = "f32")]
pub type SimdVec = wide::f32x8;

#[cfg(feature = "f32")]
pub type SimdVecDouble = wide::f32x16;

#[cfg(all(feature = "f64", not(feature = "f32")))]
pub type SimdVec = wide::f64x4;

#[cfg(all(feature = "f64", not(feature = "f32")))]
pub type SimdVecDouble = wide::f64x8;

/// SIMD lanes 数量（根据 Float 和平台配置自动选择）
///
/// | Float | AVX2 | AVX-512 | NEON | WASM |
/// |-------|------|---------|------|------|
/// | f64   | 4 | 8 | 2 | 2 |
/// | f32   | 8 | 16 | 4 | 4 |
#[cfg(all(
    feature = "f32",
    any(
        all(target_arch = "x86_64", target_feature = "avx512f")
    )
))]
pub const SIMD_LANES: usize = 16;

#[cfg(all(
    feature = "f32",
    any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        not(target_feature = "avx512f")
    ),
    any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
pub const SIMD_LANES: usize = 8;

#[cfg(all(
    feature = "f64",
    any(
        all(target_arch = "x86_64", target_feature = "avx512f")
    )
))]
pub const SIMD_LANES: usize = 8;

#[cfg(all(
    feature = "f64",
    any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        not(target_feature = "avx512f")
    ),
    any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
pub const SIMD_LANES: usize = 4;
```

#### 1.3.3 创建 `simd/scalar.rs`

- [ ] 实现标量回退版本的 `sum` 函数
- [ ] 实现标量回退版本的 `dot_product` 函数
- [ ] 实现标量回退版本的 `rolling_sum` 函数
- [ ] 添加单元测试

```rust
//! 标量回退实现（纯 Rust，无 SIMD）
//!
//! 当目标平台不支持任何 SIMD 指令集时使用
//! 也作为其他 SIMD 路径的参考实现

use crate::types::Float;
use crate::Result;

/// 标量版本的数组求和
pub fn sum(data: &[Float]) -> Float {
    data.iter().sum()
}

/// 标量版本的点积计算
pub fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Arrays must have same length".into(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// 标量版本的滚动窗口求和（用于 SMA 等指标）
pub fn rolling_sum(data: &[Float], window: usize) -> Result<Vec<Float>> {
    if window == 0 || window > data.len() {
        return Err(crate::TalibError::InvalidPeriod {
            period: window,
            reason: "must be > 0 and <= data.len()".into(),
        });
    }

    let mut result = Vec::with_capacity(data.len() - window + 1);

    // 计算第一个窗口的和
    let mut window_sum: Float = data[..window].iter().sum();
    result.push(window_sum);

    // 滚动计算
    for i in 1..=(data.len() - window) {
        window_sum += data[i + window - 1] - data[i - 1];
        result.push(window_sum);
    }

    Ok(result)
}
```

**验收标准**:
- [ ] `simd/mod.rs` 模块结构创建完成
- [ ] `SimdLevel` 枚举定义完成
- [ ] `simd/types.rs` 类型定义完成
- [ ] `simd/scalar.rs` 标量实现完成
- [ ] 所有函数有完整文档
- [ ] 标量实现单元测试通过

**交付物**:
- `crates/ta-core/src/simd/mod.rs`
- `crates/ta-core/src/simd/types.rs`
- `crates/ta-core/src/simd/scalar.rs`

**依赖**: 任务 1.2 (错误类型系统)
**可并行**: 无
**风险**: 低

---

### 任务 1.4: SIMD 运行时调度系统

**任务 ID**: 1.4
**任务名称**: 实现运行时函数指针调度系统（dispatch.rs）
**优先级**: P0 (最高)
**预估工时**: 12 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
实现基于 `OnceLock` 和函数指针的运行时调度系统，在启动时检测 CPU 特性并选择最优实现。

**子任务**:

#### 1.4.1 创建 `simd/dispatch.rs` 结构

- [ ] 定义 `ComputeFn` 类型别名（函数指针类型）
- [ ] 实现 `DispatchTable` 结构体（包含所有函数指针）
- [ ] 实现 `OnceLock` 全局静态变量
- [ ] 实现 `init_dispatch()` 初始化函数

```rust
//! 运行时调度系统
//!
//! 使用 OnceLock 缓存函数指针，实现零开销的运行时调度

use crate::types::Float;
use crate::Result;
use std::sync::OnceLock;

/// 函数指针类型定义
type SumFn = fn(data: &[Float]) -> Float;
type DotProductFn = fn(a: &[Float], b: &[Float]) -> Result<Float>;

/// 调度表结构
struct DispatchTable {
    sum: SumFn,
    dot_product: DotProductFn,
}

/// 全局调度表（OnceLock 确保只初始化一次）
static DISPATCH_TABLE: OnceLock<DispatchTable> = OnceLock::new();

/// 初始化调度表
fn init_dispatch() {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            DISPATCH_TABLE.get_or_init(|| DispatchTable {
                sum: crate::simd::arch::x86_64::avx512::sum,
                dot_product: crate::simd::arch::x86_64::avx512::dot_product,
            });
        } else if is_x86_feature_detected!("avx2") {
            DISPATCH_TABLE.get_or_init(|| DispatchTable {
                sum: crate::simd::arch::x86_64::avx2::sum,
                dot_product: crate::simd::arch::x86_64::avx2::dot_product,
            });
        } else {
            DISPATCH_TABLE.get_or_init(|| DispatchTable {
                sum: crate::simd::scalar::sum,
                dot_product: crate::simd::scalar::dot_product,
            });
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        DISPATCH_TABLE.get_or_init(|| DispatchTable {
            sum: crate::simd::arch::aarch64::neon::sum,
            dot_product: crate::simd::arch::aarch64::neon::dot_product,
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        DISPATCH_TABLE.get_or_init(|| DispatchTable {
            sum: crate::simd::arch::wasm32::simd128::sum,
            dot_product: crate::simd::arch::wasm32::simd128::dot_product,
        });
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        DISPATCH_TABLE.get_or_init(|| DispatchTable {
            sum: crate::simd::scalar::sum,
            dot_product: crate::simd::scalar::dot_product,
        });
    }
}

/// 获取调度表（确保已初始化）
fn get_dispatch_table() -> &'static DispatchTable {
    init_dispatch();
    DISPATCH_TABLE.get().expect("Dispatch table should be initialized")
}
```

#### 1.4.2 实现公共 API 函数

- [ ] 实现 `dispatch::sum()` 函数
- [ ] 实现 `dispatch::dot_product()` 函数
- [ ] 添加性能测试（验证调度开销）

```rust
/// 调度后的 sum 函数
pub fn sum(data: &[Float]) -> Float {
    let table = get_dispatch_table();
    (table.sum)(data)
}

/// 调度后的 dot_product 函数
pub fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    let table = get_dispatch_table();
    (table.dot_product)(a, b)
}
```

#### 1.4.3 添加单元测试

- [ ] 测试调度初始化
- [ ] 测试多次调用使用同一函数指针
- [ ] 性能基准测试（调度开销 < 10ns）

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_initialization() {
        // 第一次调用触发初始化
        let data = vec![1.0, 2.0, 3.0];
        let _ = sum(&data);

        // 验证调度表已初始化
        assert!(DISPATCH_TABLE.get().is_some());
    }

    #[test]
    fn test_dispatch_consistency() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // 多次调用应返回相同结果
        let result1 = sum(&data);
        let result2 = sum(&data);
        let result3 = sum(&data);

        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
    }
}
```

**验收标准**:
- [ ] `simd/dispatch.rs` 模块实现完成
- [ ] `DispatchTable` 结构体定义正确
- [ ] `OnceLock` 全局变量实现正确
- [ ] x86_64 特性检测逻辑实现
- [ ] ARM64 特性检测逻辑实现
- [ ] WASM 特性检测逻辑实现
- [ ] 公共 API 函数实现完成
- [ ] 所有单元测试通过
- [ ] 性能基准测试：调度开销 < 10ns

**交付物**:
- `crates/ta-core/src/simd/dispatch.rs`

**依赖**: 任务 1.3 (SIMD 模块基础框架)
**可并行**: 无
**风险**: 中（特性检测和调度逻辑需要仔细测试）

---

### 任务 1.5: x86_64 SIMD 实现

**任务 ID**: 1.5
**任务名称**: 实现 x86_64 平台的 AVX2 和 AVX-512 SIMD 路径
**优先级**: P0 (最高)
**预估工时**: 16 小时
**负责人**: TBD
**状态**: ✅ 已完成

**描述**:
实现 x86_64 平台的 AVX2 和 AVX-512 SIMD 路径，使用 wide crate 和 std::arch intrinsics。

**子任务**:

#### 1.5.1 创建 x86_64 模块结构

- [ ] 创建 `simd/arch/x86_64/mod.rs`
- [ ] 创建 `simd/arch/x86_64/avx2.rs`
- [ ] 创建 `simd/arch/x86_64/avx512.rs`

```rust
//! x86_64 SIMD 实现

#[cfg(target_feature = "avx2")]
pub mod avx2;

#[cfg(target_feature = "avx512f")]
pub mod avx512;
```

#### 1.5.2 实现 AVX2 路径

- [ ] 实现 `avx2::sum()` 函数（使用 `wide::f64x4`）
- [ ] 实现 `avx2::dot_product()` 函数
- [ ] 处理余数元素
- [ ] 添加单元测试

```rust
//! AVX2 实现 (256-bit SIMD)
//!
//! 使用 wide crate 进行 SIMD 计算。
//! **重要**: 所有实现必须使用 `SimdVec` 类型别名，
//! 以确保根据 Float 配置正确选择 f32/f64。

use crate::types::Float;
use crate::simd::types::{SimdVec, SIMD_LANES};
use crate::Result;

/// AVX2 SIMD 数组求和
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn sum(data: &[Float]) -> Float {
    let chunks = data.chunks_exact(SIMD_LANES);
    let remainder = chunks.remainder();

    let mut sum_vec = SimdVec::ZERO;
    for chunk in chunks {
        let vec = SimdVec::from_slice_unaligned(chunk);
        sum_vec += vec;
    }

    let mut sum = sum_vec.horizontal_sum();
    for &x in remainder {
        sum += x;
    }
    sum
}

/// AVX2 SIMD 点积计算
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Arrays must have same length".into(),
        });
    }

    let mut sum = Float::from(0.0);
    let chunks = a.chunks_exact(SIMD_LANES).zip(b.chunks_exact(SIMD_LANES));
    let remainder_a = a.chunks_exact(SIMD_LANES).remainder();
    let remainder_b = b.chunks_exact(SIMD_LANES).remainder();

    for (chunk_a, chunk_b) in chunks {
        let vec_a = SimdVec::from_slice_unaligned(chunk_a);
        let vec_b = SimdVec::from_slice_unaligned(chunk_b);
        sum += (vec_a * vec_b).horizontal_sum();
    }

    for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
        sum += x * y;
    }

    Ok(sum)
}
```

#### 1.5.3 实现 AVX-512 路径

- [ ] 实现 `avx512::sum()` 函数（使用 `std::arch::x86_64::_mm512_*`）
- [ ] 实现 `avx512::dot_product()` 函数
- [ ] 添加条件编译（`#[cfg(target_feature = "avx512f")]`）
- [ ] 添加单元测试（标记为 `#[ignore]` 除非有 AVX-512 硬件）

```rust
//! AVX-512 实现 (512-bit SIMD)
//!
//! 使用 wide crate 进行 SIMD 计算。
//! **重要**: wide crate 已经支持 AVX-512，无需使用 std::arch intrinsics。
//!
//! | Float | AVX-512 类型 | Lanes |
//! |-------|--------------|-------|
//! | f64   | wide::f64x8 | 8      |
//! | f32   | wide::f32x16 | 16     |

use crate::types::Float;
use crate::simd::types::{SimdVec, SIMD_LANES};
use crate::Result;

/// AVX-512 SIMD 数组求和
#[inline(never)]
#[target_feature(enable = "avx512f")]
pub unsafe fn sum(data: &[Float]) -> Float {
    let chunks = data.chunks_exact(SIMD_LANES);
    let remainder = chunks.remainder();

    let mut sum_vec = SimdVec::ZERO;
    for chunk in chunks {
        let vec = SimdVec::from_slice_unaligned(chunk);
        sum_vec += vec;
    }

    let mut sum = sum_vec.horizontal_sum();
    for &x in remainder {
        sum += x;
    }
    sum
}

/// AVX-512 SIMD 点积计算
#[inline(never)]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Arrays must have same length".into(),
        });
    }

    let mut sum = Float::from(0.0);
    let chunks = a.chunks_exact(SIMD_LANES).zip(b.chunks_exact(SIMD_LANES));
    let remainder_a = a.chunks_exact(SIMD_LANES).remainder();
    let remainder_b = b.chunks_exact(SIMD_LANES).remainder();

    for (chunk_a, chunk_b) in chunks {
        let vec_a = SimdVec::from_slice_unaligned(chunk_a);
        let vec_b = SimdVec::from_slice_unaligned(chunk_b);
        sum += (vec_a * vec_b).horizontal_sum();
    }

    for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
        sum += x * y;
    }

    Ok(sum)
}
```

#### 1.5.4 创建集成测试

- [ ] 创建 `tests/x86_64_simd.rs`
- [ ] 测试 AVX2 路径
- [ ] 测试 AVX-512 路径（条件编译）
- [ ] 测试跨路径一致性
- [ ] 性能基准测试

```rust
// tests/x86_64_simd.rs
#![cfg(test)]

use ta_core::simd;

#[test]
fn test_avx2_path() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let result = simd::sum(&data);
    let expected: f64 = data.iter().sum();
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_cross_path_consistency() {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();

    let result_simd = simd::sum(&data);
    let result_scalar = simd::scalar::sum(&data);

    // 允许小的浮点误差
    assert!((result_simd - result_scalar).abs() < 1e-9);
}
```

**验收标准**:
- [ ] `simd/arch/x86_64/mod.rs` 模块结构创建完成
- [ ] AVX2 实现 (`avx2.rs`) 完成
  - [ ] `sum()` 函数实现并测试通过
  - [ ] `dot_product()` 函数实现并测试通过
- [ ] AVX-512 实现 (`avx512.rs`) 完成
  - [ ] 代码编译通过（`#[cfg(target_feature = "avx512f")]`）
  - [ ] `sum()` 函数实现
  - [ ] `dot_product()` 函数实现
- [ ] 所有单元测试通过
- [ ] 跨路径一致性验证通过
- [ ] 性能基准：AVX2 比标量快 2x 以上

**交付物**:
- `crates/ta-core/src/simd/arch/x86_64/mod.rs`
- `crates/ta-core/src/simd/arch/x86_64/avx2.rs`
- `crates/ta-core/src/simd/arch/x86_64/avx512.rs`
- `tests/x86_64_simd.rs`

**依赖**: 任务 1.4 (SIMD 运行时调度系统)
**可并行**: 部分子任务可并行
**风险**: 中（AVX-512 硬件可用性有限）

---

### 任务 1.6: ARM64 SIMD 实现

**任务 ID**: 1.6
**任务名称**: 实现 ARM64 平台的 NEON SIMD 路径
**优先级**: P0 (最高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ✅ 已完成

**描述**:
实现 ARM64 平台的 NEON SIMD 路径，使用 wide crate。

**子任务**:

#### 1.6.1 创建 ARM64 模块结构

- [x] 创建 `simd/arch/aarch64/mod.rs`
- [x] 创建 `simd/arch/aarch64/neon.rs`

```rust
//! ARM64 SIMD 实现

#[cfg(target_arch = "aarch64")]
pub mod neon;
```

#### 1.6.2 实现 NEON 路径

- [x] 实现 `neon::sum()` 函数（使用 `wide::f64x2`）
- [x] 实现 `neon::dot_product()` 函数
- [x] 处理余数元素
- [x] 添加单元测试

```rust
//! ARM64 NEON 实现 (128-bit SIMD)
//!
//! 使用 wide crate 进行 SIMD 计算。
//! **重要**: 所有实现必须使用 `SimdVec` 类型别名，
//! 以确保根据 Float 配置正确选择 f32/f64。

use crate::types::Float;
use crate::simd::types::{SimdVec, SIMD_LANES};
use crate::Result;

/// NEON SIMD 数组求和
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn sum(data: &[Float]) -> Float {
    let chunks = data.chunks_exact(SIMD_LANES);
    let remainder = chunks.remainder();

    let mut sum_vec = SimdVec::ZERO;
    for chunk in chunks {
        let vec = SimdVec::from_slice_unaligned(chunk);
        sum_vec += vec;
    }

    let mut sum = sum_vec.horizontal_sum();
    for &x in remainder {
        sum += x;
    }
    sum
}

/// NEON SIMD 点积计算
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Arrays must have same length".into(),
        });
    }

    let mut sum = Float::from(0.0);
    let chunks = a.chunks_exact(SIMD_LANES).zip(b.chunks_exact(SIMD_LANES));
    let remainder_a = a.chunks_exact(SIMD_LANES).remainder();
    let remainder_b = b.chunks_exact(SIMD_LANES).remainder();

    for (chunk_a, chunk_b) in chunks {
        let vec_a = SimdVec::from_slice_unaligned(chunk_a);
        let vec_b = SimdVec::from_slice_unaligned(chunk_b);
        sum += (vec_a * vec_b).horizontal_sum();
    }

    for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
        sum += x * y;
    }

    Ok(sum)
}
```

#### 1.6.3 创建集成测试

- [x] 创建 `tests/aarch64_simd.rs`
- [x] 测试 NEON 路径
- [x] 条件编译确保只在 ARM64 运行
- [x] 跨路径一致性测试

```rust
// tests/aarch64_simd.rs
#![cfg(test)]
#![cfg(target_arch = "aarch64")]

use ta_core::simd;

#[test]
fn test_neon_path() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let result = simd::sum(&data);
    let expected: f64 = data.iter().sum();
    assert!((result - expected).abs() < 1e-10);
}
```

**验收标准**:
- [x] `simd/arch/aarch64/mod.rs` 模块结构创建完成
- [x] NEON 实现 (`neon.rs`) 完成
  - [x] `sum()` 函数实现并测试通过
  - [x] `dot_product()` 函数实现并测试通过
- [x] 所有单元测试通过
- [x] 跨路径一致性验证通过
- [ ] 性能基准：NEON 比标量快 2x 以上 (待 ARM64 硬件测试)

**交付物**:
- `crates/ta-core/src/simd/arch/aarch64/mod.rs`
- `crates/ta-core/src/simd/arch/aarch64/neon.rs`
- `tests/aarch64_simd.rs`

**依赖**: 任务 1.4 (SIMD 运行时调度系统)
**可并行**: 可以与任务 1.5 并行
**风险**: 中（需要 ARM64 硬件或模拟器测试）

---

### 任务 1.7: WASM SIMD 实现

**任务 ID**: 1.7
**任务名称**: 实现 WASM 平台的 SIMD128 SIMD 路径
**优先级**: P1 (高)
**预估工时**: 6 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
实现 WASM 平台的 SIMD128 SIMD 路径，使用 wide crate。

**子任务**:

#### 1.7.1 创建 WASM 模块结构

- [ ] 创建 `simd/arch/wasm32/mod.rs`
- [ ] 创建 `simd/arch/wasm32/simd128.rs`

```rust
//! WASM SIMD 实现

#[cfg(target_arch = "wasm32")]
pub mod simd128;
```

#### 1.7.2 实现 SIMD128 路径

- [ ] 实现 `simd128::sum()` 函数（使用 `wide::f64x2`）
- [ ] 实现 `simd128::dot_product()` 函数
- [ ] 处理 WASM 特定的限制（128-bit 最大）
- [ ] 添加单元测试

```rust
//! WASM SIMD128 实现 (128-bit SIMD)
//!
//! 使用 wide crate 进行 SIMD 计算。
//! **重要**: 所有实现必须使用 `SimdVec` 类型别名，
//! 以确保根据 Float 配置正确选择 f32/f64。

use crate::types::Float;
use crate::simd::types::{SimdVec, SIMD_LANES};
use crate::Result;

/// SIMD128 SIMD 数组求和
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn sum(data: &[Float]) -> Float {
    let chunks = data.chunks_exact(SIMD_LANES);
    let remainder = chunks.remainder();

    let mut sum_vec = SimdVec::ZERO;
    for chunk in chunks {
        let vec = SimdVec::from_slice_unaligned(chunk);
        sum_vec += vec;
    }

    let mut sum = sum_vec.horizontal_sum();
    for &x in remainder {
        sum += x;
    }
    sum
}

/// SIMD128 SIMD 点积计算
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Arrays must have same length".into(),
        });
    }

    let mut sum = Float::from(0.0);
    let chunks = a.chunks_exact(SIMD_LANES).zip(b.chunks_exact(SIMD_LANES));
    let remainder_a = a.chunks_exact(SIMD_LANES).remainder();
    let remainder_b = b.chunks_exact(SIMD_LANES).remainder();

    for (chunk_a, chunk_b) in chunks {
        let vec_a = SimdVec::from_slice_unaligned(chunk_a);
        let vec_b = SimdVec::from_slice_unaligned(chunk_b);
        sum += (vec_a * vec_b).horizontal_sum();
    }

    for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
        sum += x * y;
    }

    Ok(sum)
}
```

#### 1.7.2 创建 WASM 构建配置

- [ ] 更新 `ta-wasm/Cargo.toml` 添加 WASM 特性
- [ ] 创建 `wasm-pack` 配置
- [ ] 添加 WASM 示例

```toml
# crates/ta-wasm/Cargo.toml
[package]
name = "ta-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
ta-core = { path = "../ta-core" }
wasm-bindgen = "0.2"
console_error_panic_hook = "0.1"

[dev-dependencies]
wasm-bindgen-test = "0.3"
```

#### 1.7.3 创建集成测试

- [ ] 创建 `tests/wasm_simd.rs`
- [ ] 条件编译确保只在 WASM 运行
- [ ] 测试 SIMD128 路径

**验收标准**:
- [ ] `simd/arch/wasm32/mod.rs` 模块结构创建完成
- [ ] SIMD128 实现 (`simd128.rs`) 完成
  - [ ] `sum()` 函数实现并测试通过
  - [ ] `dot_product()` 函数实现并测试通过
- [ ] WASM 构建配置完成
- [ ] `wasm-pack build` 成功
- [ ] 所有单元测试通过

**交付物**:
- `crates/ta-core/src/simd/arch/wasm32/mod.rs`
- `crates/ta-core/src/simd/arch/wasm32/simd128.rs`
- `crates/ta-wasm/Cargo.toml` (更新)
- `tests/wasm_simd.rs`

**依赖**: 任务 1.4 (SIMD 运行时调度系统)
**可并行**: 可以与任务 1.5 和 1.6 并行
**风险**: 中（WASM 测试需要特定工具）

---

### 任务 1.8: 实现 Core Traits

**任务 ID**: 1.8
**任务名称**: 实现统一 Indicator trait（集成 SIMD 支持）
**优先级**: P0 (最高)
**预估工时**: 12 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
定义统一的 `Indicator` trait，同时支持批量计算、获取最新值和流式处理，并集成 SIMD 加速。

**子任务**:

#### 1.8.1 更新 `traits.rs`

- [ ] 定义 `Indicator` trait（保持现有设计）
- [ ] 定义 `Resettable` trait（可选）
- [ ] 添加 `SimdCompute` trait（内部使用）

```rust
//! 核心 Traits 定义
//!
//! 定义指标的统一接口，支持批量、流式和单值查询。

use crate::types::Float;
use crate::Result;

/// 指标 trait：所有技术指标的统一接口
///
/// 提供三种计算模式：
/// - `compute()`: 零拷贝批量计算（性能优先）
/// - `compute_to_vec()`: 便捷批量计算（易用性优先）
/// - `next()`: 获取最新值（实时流式）
/// - `stream()`: 流式处理多个输入
pub trait Indicator {
    /// 零拷贝批量计算
    ///
    /// 计算指标值，将结果写入预分配的输出缓冲区。
    ///
    /// # Arguments
    ///
    /// * `inputs` - 输入数据切片
    /// * `outputs` - 输出缓冲区（必须足够大）
    ///
    /// # Returns
    ///
    /// 返回实际写入的输出值数量
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::{Sma, Indicator};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let mut output = vec![0.0; data.len()];
    /// let sma = Sma::new(3);
    ///
    /// let count = sma.compute(&data, &mut output).unwrap();
    /// assert_eq!(count, 3);  // 5 - 3 + 1 = 3
    /// ```
    fn compute(&self, inputs: &[Float], outputs: &mut [Float]) -> Result<usize>;

    /// 便捷批量计算
    ///
    /// 自动分配输出缓冲区并返回 Vec。使用性优先，性能次要。
    ///
    /// # Returns
    ///
    /// 返回包含所有计算结果的 Vec
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::{Sma, Indicator};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let sma = Sma::new(3);
    ///
    /// let result = sma.compute_to_vec(&data).unwrap();
    /// assert_eq!(result.len(), 3);
    /// ```
    fn compute_to_vec(&self, inputs: &[Float]) -> Result<Vec<Float>> {
        // 计算输出长度（由具体指标实现）
        let output_len = self.required_output_len(inputs.len());
        let mut outputs = vec![Float::from(0.0); output_len];
        self.compute(inputs, &mut outputs)?;
        Ok(outputs)
    }

    /// 获取下一个值（实时流式）
    ///
    /// 处理单个输入值并返回最新指标值。
    /// 适用于实时数据流处理。
    ///
    /// # Arguments
    ///
    /// * `input` - 单个输入值
    ///
    /// # Returns
    ///
    /// 返回最新指标值
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::{Sma, Indicator};
    ///
    /// let mut sma = Sma::new(3);
    ///
    /// assert!(sma.next(1.0).is_none());  // 数据不足
    /// assert!(sma.next(2.0).is_none());
    /// assert_eq!(sma.next(3.0), Some(2.0));  // (1+2+3)/3 = 2
    /// assert_eq!(sma.next(4.0), Some(3.0));  // (2+3+4)/3 = 3
    /// ```
    fn next(&mut self, input: Float) -> Option<Float>;

    /// 流式处理多个输入
    ///
    /// 批量处理输入数据，适用于高频数据流。
    ///
    /// # Returns
    ///
    /// 返回所有可用的输出值
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::{Sma, Indicator};
    ///
    /// let mut sma = Sma::new(3);
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    ///
    /// let result = sma.stream(&data);
    /// assert_eq!(result.len(), 3);
    /// assert_eq!(result[0], 2.0);
    /// assert_eq!(result[2], 4.0);
    /// ```
    fn stream(&mut self, inputs: &[Float]) -> Vec<Float> {
        let mut results = Vec::new();
        for &input in inputs {
            if let Some(value) = self.next(input) {
                results.push(value);
            }
        }
        results
    }

    /// 计算给定输入长度所需的输出长度
    ///
    /// 默认实现假设输出长度等于输入长度。
    /// 指标可以覆盖此方法以提供更精确的计算。
    fn required_output_len(&self, input_len: usize) -> usize {
        input_len
    }
}

/// 可重置 trait：允许指标重置内部状态
///
/// 适用于需要多次独立计算同一指标的场景。
pub trait Resettable {
    /// 重置指标到初始状态
    ///
    /// 清除所有缓冲区和内部状态。
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::{Sma, Indicator, Resettable};
    ///
    /// let mut sma = Sma::new(3);
    ///
    /// // 第一次计算
    /// let data1 = vec![1.0, 2.0, 3.0];
    /// sma.stream(&data1);
    ///
    /// // 重置
    /// sma.reset();
    ///
    /// // 第二次计算（从干净状态开始）
    /// let data2 = vec![4.0, 5.0, 6.0];
    /// let result = sma.stream(&data2);
    /// ```
    fn reset(&mut self);
}

/// SIMD 计算辅助 trait（内部使用）
///
/// 为指标实现提供 SIMD 加速的计算原语。
/// 指标开发者通常不需要直接实现此 trait。
///
/// # Example
///
/// ```rust,ignore
/// use ta_core::simd;
///
/// // 在指标实现中使用 SIMD 加速
/// let sum = simd::sum(&data);
/// ```
pub trait SimdCompute {
    /// 使用最优 SIMD 路径计算数组和
    fn simd_sum(data: &[Float]) -> Float;

    /// 使用最优 SIMD 路径计算点积
    fn simd_dot_product(a: &[Float], b: &[Float]) -> Result<Float>;

    /// 批量滑动窗口计算（SMA 优化）
    fn simd_rolling_sum(data: &[Float], window: usize) -> Result<Vec<Float>>;
}

// 为 Float 类型实现 SimdCompute（使用全局 SIMD API）
impl SimdCompute for Float {
    fn simd_sum(data: &[Float]) -> Float {
        crate::simd::sum(data)
    }

    fn simd_dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
        crate::simd::dot_product(a, b)
    }

    fn simd_rolling_sum(data: &[Float], window: usize) -> Result<Vec<Float>> {
        crate::simd::scalar::rolling_sum(data, window)
    }
}
```

#### 1.8.2 更新 SMA 实现

- [ ] 修改 `overlap/sma.rs` 使用 SIMD 加速
- [ ] 实现 `Indicator` trait
- [ ] 实现 `Resettable` trait
- [ ] 添加完整文档和示例

```rust
//! Simple Moving Average (SMA) - 简单移动平均
//!
//! SMA 是最基本的移动平均线，计算过去 N 个价格点的平均值。

use crate::types::Float;
use crate::{Indicator, Resettable, Result, SimdCompute, TalibError};

/// Simple Moving Average
///
/// 计算指定周期内的简单算术平均值。
///
/// # Example
///
/// ```rust
/// use ta_core::{Sma, Indicator};
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mut sma = Sma::new(3);
///
/// // 批量计算
/// let result = sma.compute_to_vec(&data).unwrap();
/// assert_eq!(result, vec![2.0, 3.0, 4.0]);
///
/// // 流式计算
/// let mut sma = Sma::new(3);
/// assert_eq!(sma.next(1.0), None);
/// assert_eq!(sma.next(2.0), None);
/// assert_eq!(sma.next(3.0), Some(2.0));
/// ```
pub struct Sma {
    /// 移动平均周期
    period: usize,
    /// 缓冲区（用于流式模式）
    buffer: Vec<Float>,
    /// 当前窗口和
    sum: Float,
    /// 缓冲区索引
    index: usize,
    /// 已累计的数据点数量
    count: usize,
}

impl Sma {
    /// 创建新的 SMA 指标
    ///
    /// # Arguments
    ///
    /// * `period` - 移动平均周期（必须 > 0）
    ///
    /// # Panics
    ///
    /// 如果 period 为 0，将 panic
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::Sma;
    ///
    /// let sma = Sma::new(10);
    /// ```
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "SMA period must be greater than 0");

        Self {
            period,
            buffer: vec![Float::from(0.0); period],
            sum: Float::from(0.0),
            index: 0,
            count: 0,
        }
    }

    /// 获取移动平均周期
    pub fn period(&self) -> usize {
        self.period
    }
}

impl Indicator for Sma {
    fn compute(&self, inputs: &[Float], outputs: &mut [Float]) -> Result<usize> {
        if inputs.len() < self.period {
            return Err(TalibError::InsufficientData {
                required: self.period,
                actual: inputs.len(),
            });
        }

        let output_len = inputs.len() - self.period + 1;
        if outputs.len() < output_len {
            return Err(TalibError::InvalidInput {
                message: format!("Output buffer too small: need {}, got {}", output_len, outputs.len()),
            });
        }

        // 使用 SIMD 加速的滚动求和
        let rolling_sums = Float::simd_rolling_sum(inputs, self.period)?;

        // 计算平均值
        let period_float = Float::from(self.period);
        for (i, &sum) in rolling_sums.iter().enumerate() {
            outputs[i] = sum / period_float;
        }

        Ok(output_len)
    }

    fn next(&mut self, input: Float) -> Option<Float> {
        // 更新缓冲区
        self.buffer[self.index] = input;
        self.sum += input;
        self.index = (self.index + 1) % self.period;
        self.count += 1;

        // 只有在数据足够时才返回值
        if self.count >= self.period {
            Some(self.sum / Float::from(self.period))
        } else {
            None
        }
    }

    fn required_output_len(&self, input_len: usize) -> usize {
        if input_len < self.period {
            0
        } else {
            input_len - self.period + 1
        }
    }
}

impl Resettable for Sma {
    fn reset(&mut self) {
        self.buffer.fill(Float::from(0.0));
        self.sum = Float::from(0.0);
        self.index = 0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_new() {
        let sma = Sma::new(5);
        assert_eq!(sma.period(), 5);
    }

    #[test]
    fn test_sma_compute() {
        let sma = Sma::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; 3];

        let count = sma.compute(&data, &mut output).unwrap();
        assert_eq!(count, 3);
        assert_eq!(output, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sma_compute_to_vec() {
        let sma = Sma::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = sma.compute_to_vec(&data).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sma_next() {
        let mut sma = Sma::new(3);

        assert_eq!(sma.next(1.0), None);
        assert_eq!(sma.next(2.0), None);
        assert_eq!(sma.next(3.0), Some(2.0));
        assert_eq!(sma.next(4.0), Some(3.0));
        assert_eq!(sma.next(5.0), Some(4.0));
    }

    #[test]
    fn test_sma_stream() {
        let mut sma = Sma::new(3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = sma.stream(&data);
        assert_eq!(result, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sma_reset() {
        let mut sma = Sma::new(3);

        sma.next(1.0);
        sma.next(2.0);
        sma.next(3.0);

        sma.reset();

        // 重置后应该从头开始
        assert_eq!(sma.next(4.0), None);
        assert_eq!(sma.next(5.0), None);
        assert_eq!(sma.next(6.0), Some(5.0));
    }
}
```

#### 1.8.3 添加单元测试

- [ ] 测试 `Indicator` trait 所有方法
- [ ] 测试 `Resettable` trait
- [ ] 测试 SIMD 加速效果
- [ ] 跨平台一致性测试

**验收标准**:
- [ ] `Indicator` trait 定义完成且有完整文档
- [ ] `Resettable` trait 定义完成
- [ ] `SimdCompute` trait 定义完成并在内部使用
- [ ] SMA 示例指标实现并测试通过
  - [ ] 在所有支持平台（x86_64 AVX2/AVX-512、ARM64 NEON、WASM SIMD128）上通过测试
- [ ] 所有 SIMD 路径结果一致性验证通过
- [ ] **所有 SIMD 实现正确使用 `Float` 类型别名**：
  - [ ] 不直接使用 `wide::f64x4` 或 `wide::f32x8` 等硬编码类型
  - [ ] 不直接使用 `std::arch::x86_64::_mm512_*` 等硬编码 intrinsics（使用条件编译包装）
  - [ ] 所有函数签名使用 `Float` 类型：`fn sum(data: &[Float]) -> Float`
  - [ ] 所有内部 SIMD 操作使用 `SimdVec` 类型别名
- [ ] 零拷贝 `compute()` 接口正常工作（性能验证）
- [ ] 便捷 `compute_to_vec()` 接口正常工作（易用性验证）
- [ ] 流式 `next()` 和 `stream()` 接口正常工作
- [ ] 三种模式（批量、流式、单值查询）都能正常工作
- [ ] 所有单元测试通过
- [ ] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/traits.rs` (更新)
- `crates/ta-core/src/overlap/sma.rs` (更新)
- 单元测试文件

**依赖**: 任务 1.5-1.7 (平台 SIMD 实现)
**可并行**: 部分子任务可并行
**风险**: 中

---

### 任务 1.9: 测试基础设施

**任务 ID**: 1.9
**任务名称**: 设置测试基础设施和工具
**优先级**: P1 (高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
配置测试框架、工具和 CI/CD 管道，确保代码质量和测试覆盖率。**配置多平台测试，确保所有 SIMD 路径得到测试。**

**子任务**:

#### 1.9.1 配置测试框架

- [ ] 添加测试依赖到 `Cargo.toml`
- [ ] 配置测试输出格式
- [ ] 创建测试辅助模块

```toml
# crates/ta-core/Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
quickcheck = "1.0"
```

#### 1.9.2 配置代码质量工具

- [ ] 创建 `.rustfmt.toml` 配置
- [ ] 创建 `clippy.toml` 配置
- [ ] 添加 pre-commit hook

```toml
# .rustfmt.toml
max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Unix"
use_small_heuristics = "Default"
```

#### 1.9.3 创建 CI/CD 配置

- [ ] 创建 `.github/workflows/ci.yml`（GitHub Actions）
- [ ] 配置测试矩阵，覆盖所有 SIMD 路径

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    name: Test (${{ matrix.target }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        include:
          # x86_64 Linux
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            rust: stable
            flags: ""
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
            rust: stable
            flags: "-C target-feature=+avx2"

          # ARM64 Linux
          - target: aarch64-unknown-linux-gnu
            os: ubuntu-latest
            rust: stable
            flags: ""

          # ARM64 macOS (Apple Silicon)
          - target: aarch64-apple-darwin
            os: macos-latest
            rust: stable
            flags: ""

          # WASM
          - target: wasm32-unknown-unknown
            os: ubuntu-latest
            rust: stable
            flags: ""

    steps:
    - uses: actions/checkout@v4

    - uses: actions-rust-lang/setup-rust-toolchain@v1
      with:
        toolchain: ${{ matrix.rust }}
        target: ${{ matrix.target }}
        override: true

    - name: Run tests
      run: cargo test --target ${{ matrix.target }} --all-features

    - name: Run clippy
      run: cargo clippy --target ${{ matrix.target }} --all-targets --all-features -- -D warnings

    - name: Check formatting
      run: cargo fmt -- --check
```

#### 1.9.4 配置性能基准测试

- [ ] 更新 `benches/sma.rs` 添加 SIMD 对比
- [ ] 生成跨平台性能报告

```rust
// benches/sma_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ta_core::{Sma, Indicator, simd};

fn benchmark_sma_compute(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let period = 20;

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        let sma = Sma::new(period);

        c.bench_function(&format!("sma_compute_{}", size), |b| {
            b.iter(|| {
                let mut output = vec![0.0; size - period + 1];
                sma.compute(black_box(&data), black_box(&mut output)).unwrap()
            })
        });
    }
}

fn benchmark_simd_sum(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        c.bench_function(&format!("simd_sum_{}", size), |b| {
            b.iter(|| {
                simd::sum(black_box(&data))
            })
        });
    }
}

criterion_group!(benches, benchmark_sma_compute, benchmark_simd_sum);
criterion_main!(benches);
```

#### 1.9.5 创建测试辅助模块

- [ ] 创建 `crates/ta-core/src/test_utils.rs`
- [ ] 添加 SIMD 路径测试辅助函数
- [ ] 添加跨平台结果一致性验证函数

```rust
//! 测试辅助工具

use crate::types::Float;

/// 断言两个数组在给定误差范围内相等
pub fn assert_close(a: &[Float], b: &[Float], epsilon: Float) {
    assert_eq!(a.len(), b.len(), "Arrays have different lengths");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < epsilon,
            "Arrays differ at index {}: {} vs {} (epsilon = {})",
            i, x, y, epsilon
        );
    }
}

/// 生成随机测试数据
pub fn generate_random_data(len: usize) -> Vec<Float> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..len).map(|_| rng.gen::<Float>() * 100.0).collect()
}

/// 验证 SIMD 和标量实现结果一致
pub fn verify_simd_scalar_consistency(data: &[Float], simd_result: Float) {
    let scalar_result: Float = data.iter().sum();
    assert!(
        (simd_result - scalar_result).abs() < 1e-9,
        "SIMD and scalar results differ: {} vs {}",
        simd_result,
        scalar_result
    );
}
```

**验收标准**:
- [ ] 测试框架配置完成
- [ ] 代码覆盖率工具可以正常运行
- [ ] CI/CD 管道配置完成（包含多平台矩阵测试）
- [ ] 所有 lint 规则配置完成
- [ ] 基准测试可以运行（包含 SIMD 路径对比）
- [ ] 测试辅助模块完成（包含 SIMD 测试辅助）
- [ ] 所有 SIMD 路径在 CI 中通过测试

**交付物**:
- `.github/workflows/ci.yml`
- `.rustfmt.toml`
- `clippy.toml`
- `crates/ta-core/src/test_utils.rs`
- CI/CD 配置文档

**依赖**: 任务 1.8 (Core Traits)
**可并行**: 可以与任务 1.10 并行
**风险**: 中（多平台 CI 配置复杂）

---

### 任务 1.10: 构建配置与平台支持

**任务 ID**: 1.10
**任务名称**: 配置构建系统与平台支持（build.rs, Cargo features）
**优先级**: P1 (高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
配置构建脚本和 Cargo 特性，实现编译时平台检测、条件编译和多平台支持。

**子任务**:

#### 1.10.1 更新 workspace Cargo.toml

- [ ] 添加 SIMD 相关依赖
- [ ] 添加特性标志

```toml
[workspace]
resolver = "3"
members = [
    "crates/ta-core",
    "crates/ta-py",
    "crates/ta-wasm",
    "crates/ta-benchmarks",
]
default-members = [
    "crates/ta-core",
    "crates/ta-wasm",
    "crates/ta-benchmarks",
]

[workspace.dependencies]
wide = "0.7"
```

#### 1.10.2 更新 ta-core Cargo.toml

- [ ] 添加 features 配置

```toml
# crates/ta-core/Cargo.toml
[package]
name = "ta-core"
version = "0.1.0"
edition = "2021"

[features]
default = ["std", "f64"]

# 浮点精度选择
f32 = []
f64 = []

# 标准库支持（用于测试和 I/O）
std = ["alloc", "dep:std"]
core_error = []

# 文档生成特性
docs = []

[dependencies]
wide.workspace = true

[dev-dependencies]
criterion.workspace = true
```

**验收标准**:
- [ ] workspace Cargo.toml 配置完成
- [ ] ta-core Cargo.toml features 配置完成
- [ ] 所有目标平台可以成功编译
- [ ] 运行时平台检测工作正常

**交付物**:
- `Cargo.toml` (workspace 根目录 - 更新)
- `crates/ta-core/Cargo.toml` (更新)
- 构建配置文档

**依赖**: 任务 1.9 (测试基础设施)
**可并行**: 可以与任务 1.11 部分并行
**风险**: 中（跨平台配置复杂）

---

### 任务 1.11: 跨平台测试基础设施

**任务 ID**: 1.11
**任务名称**: 设置跨平台测试基础设施（Docker, QEMU, 模拟器）
**优先级**: P1 (高)
**预估工时**: 12 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
配置跨平台测试环境，使用 Docker、QEMU 和模拟器在开发机器上测试所有目标平台。

**子任务**:

#### 1.11.1 配置 Docker 跨平台测试环境

- [ ] 创建 `docker/` 目录
- [ ] 创建 `docker/Dockerfile.test-x86_64`
- [ ] 创建 `docker/Dockerfile.test-aarch64`

```dockerfile
# docker/Dockerfile.test-aarch64
FROM rust:1.78-bookworm

# Install cross-compilation tools
RUN apt-get update && apt-get install -y \
    qemu-user \
    gcc-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    && rm -rf /var/lib/apt/lists/*

# Install cross
RUN cargo install cross --version 0.2.5

# Add ARM64 target
RUN rustup target add aarch64-unknown-linux-gnu

WORKDIR /workspace
```

#### 1.11.2 创建跨平台测试脚本

- [ ] 创建 `scripts/test-all-platforms.sh`
- [ ] 创建 `scripts/test-x86_64.sh`
- [ ] 创建 `scripts/test-aarch64.sh`
- [ ] 创建 `scripts/test-wasm.sh`

```bash
#!/bin/bash
# scripts/test-all-platforms.sh

set -e

echo "================================"
echo "Running Cross-Platform Tests"
echo "================================"

# Native platform
echo ""
echo "1. Testing native platform..."
cargo test --all-features

# ARM64
if command -v cross &> /dev/null; then
    echo ""
    echo "2. Testing aarch64..."
    cross test --target aarch64-unknown-linux-gnu --all-features
else
    echo "Cross not available, skipping aarch64 tests"
fi

echo ""
echo "================================"
echo "Cross-Platform Tests Complete!"
echo "================================"
```

#### 1.11.3 创建跨平台测试文档

- [ ] 编写 `docs/cross-platform-testing.md`
- [ ] 说明如何在本地运行跨平台测试
- [ ] 说明如何添加新的目标平台

**验收标准**:
- [ ] Docker 测试环境配置完成（x86_64, ARM64）
- [ ] 所有跨平台测试脚本完成
- [ ] 至少可以在以下平台运行测试:
  - [ ] x86_64 Linux (原生)
  - [ ] ARM64 Linux (cross)
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

**依赖**: 任务 1.10 (构建配置)
**可并行**: 无
**风险**: 中（QEMU 和 Docker 配置复杂）

---

## Phase 2: 重叠研究指标

**阶段名称**: Overlap Studies
**持续时间**: 4 周（Weeks 7-10）
**目标**: 实现 16 个移动平均类型指标
**依赖**: Phase 1

**阶段进度**:

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 任务 2.1: 实现移动平均指标系列 | ⬜ 待开始 | - | 16 个指标 |
| 任务 2.2: 创建性能基准测试 | ⬜ 待开始 | - | 对比测试 |
| 任务 2.3: 编写单元测试 | ⬜ 待开始 | - | 覆盖率 > 90% |

**总体进度**: 0/3 任务完成 (0%)

---

### 任务 2.1: 实现移动平均指标系列

**任务 ID**: 2.1
**任务名称**: 实现 16 个移动平均类型指标
**优先级**: P0 (最高)
**预估工时**: 24 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
实现 TA-Lib 重叠研究指标类中的所有移动平均类型指标，包括 SMA、EMA、WMA 等共 16 个指标。

**指标列表**:

| 指标名称 | 全称 | TA-Lib 函数 | 复杂度 |
|---------|--------|--------------|---------|
| SMA | Simple Moving Average | TA_MA | 中 |
| EMA | Exponential Moving Average | TA_EMA | 中 |
| WMA | Weighted Moving Average | TA_WMA | 中 |
| DEMA | Double Exponential Moving Average | TA_DEMA | 中 |
| TEMA | Triple Exponential Moving Average | TA_TEMA | 中 |
| T3 | T3 Moving Average | TA_T3 | 高 |
| KAMA | Kaufman Adaptive Moving Average | TA_KAMA | 高 |
| MAMA | MESA Adaptive Moving Average | TA_MAMA | 高 |
| TRIMA | Triangular Moving Average | TA_TRIMA | 低 |
| WWMA | Welles Wilder Moving Average | TA_WWMA | 中 |
| HMA | Hull Moving Average | TA_HMA | 中 |
| VWAP | Volume Weighted Average Price | TA_VWAP | 高 |
| VAR | Variable Moving Average | TA_VAR | 中 |
| AVGPRICE | Average Price | TA_AVGPRICE | 低 |
| MEDPRICE | Median Price | TA_MEDPRICE | 低 |
| TYPPRICE | Typical Price | TA_TYPPRICE | 低 |

**子任务**:

#### 2.1.1 创建指标模块结构

- [ ] 创建 `crates/ta-core/src/overlap/mod.rs`
- [ ] 为每个指标创建独立文件

```
overlap/
├── mod.rs
├── sma.rs      # ✅ 已完成 (Phase 1)
├── ema.rs      # 新增
├── wma.rs      # 新增
├── dema.rs     # 新增
├── tema.rs     # 新增
├── t3.rs       # 新增
├── kama.rs     # 新增
├── mama.rs     # 新增
├── trima.rs    # 新增
├── wwma.rs     # 新增
├── hma.rs      # 新增
├── vwap.rs     # 新增
├── var.rs      # 新增
├── avgprice.rs # 新增
├── medprice.rs # 新增
└── typprice.rs # 新增
```

#### 2.1.2 实现每个指标

**以 EMA 为例**:

```rust
//! Exponential Moving Average (EMA) - 指数移动平均

use crate::types::Float;
use crate::{Indicator, Resettable, Result, TalibError};

/// Exponential Moving Average
///
/// 指数移动平均给予最近的价格更大的权重。
///
/// # Example
///
/// ```rust
/// use ta_core::{Ema, Indicator};
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mut ema = Ema::new(3);
///
/// let result = ema.compute_to_vec(&data).unwrap();
/// ```
pub struct Ema {
    period: usize,
    alpha: Float,
    prev_ema: Float,
    count: usize,
}

impl Ema {
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "EMA period must be greater than 0");

        // alpha = 2 / (period + 1)
        let alpha = Float::from(2.0) / Float::from(period + 1);

        Self {
            period,
            alpha,
            prev_ema: Float::from(0.0),
            count: 0,
        }
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

impl Indicator for Ema {
    fn compute(&self, inputs: &[Float], outputs: &mut [Float]) -> Result<usize> {
        if inputs.is_empty() {
            return Ok(0);
        }

        let output_len = inputs.len();
        if outputs.len() < output_len {
            return Err(TalibError::InvalidInput {
                message: format!("Output buffer too small: need {}, got {}", output_len, outputs.len()),
            });
        }

        // EMA 批量计算
        let mut ema = inputs[0];
        outputs[0] = ema;

        for i in 1..inputs.len() {
            ema = self.alpha * inputs[i] + (Float::from(1.0) - self.alpha) * ema;
            outputs[i] = ema;
        }

        Ok(output_len)
    }

    fn next(&mut self, input: Float) -> Option<Float> {
        self.prev_ema = self.alpha * input + (Float::from(1.0) - self.alpha) * self.prev_ema;
        self.count += 1;

        Some(self.prev_ema)
    }

    fn required_output_len(&self, input_len: usize) -> usize {
        input_len
    }
}

impl Resettable for Ema {
    fn reset(&mut self) {
        self.prev_ema = Float::from(0.0);
        self.count = 0;
    }
}
```

**其他指标实现要点**:

| 指标 | 实现要点 | SIMD 优化机会 |
|-------|----------|--------------|
| WMA | 线性权重，需要权重数组 | 求和可用 SIMD |
| DEMA/TEMA | EMA 的组合 | 多次 EMA 计算 |
| T3 | 基于 DEMA 的三次平滑 | 复杂度高 |
| KAMA | 自适应平滑系数 | 条件分支多 |
| MAMA | MESA 周期检测 | 使用三角函数 |
| TRIMA | 三角加权移动平均 | 滚动窗口可用 SIMD |
| WWMA | Wilder 的 EMA 变体 | 同 EMA |
| HMA | WMA 的再平滑 | 多层计算 |
| VWAP | 成交量加权 | 需要成交量数据 |
| VAR | 自适应平滑周期 | 条件分支 |
| AVGPRICE | (High+Low+Close)/3 | 简单计算 |
| MEDPRICE | (High+Low)/2 | 简单计算 |
| TYPPRICE | (High+Low+Close)/3 | 与 AVGPRICE 相同 |

#### 2.1.3 更新 lib.rs 导出

- [ ] 在 `crates/ta-core/src/lib.rs` 中添加所有新指标

```rust
pub mod overlap;

pub use overlap::{
    Sma, Ema, Wma, Dema, Tema, T3, Kama, Mama, Trima,
    Wwma, Hma, Vwap, Var, AvgPrice, MedPrice, Typprice,
};
```

**验收标准**:
- [ ] 所有 16 个移动平均指标实现完成
- [ ] 每个指标实现 `Indicator` trait
- [ ] 每个指标实现 `Resettable` trait
- [ ] 所有指标使用 SIMD 加速（适用时）
- [ ] 所有指标有完整文档和示例
- [ ] 代码覆盖率 > 90%
- [ ] 性能基准测试通过

**交付物**:
- `crates/ta-core/src/overlap/mod.rs` (更新)
- 16 个指标实现文件
- 单元测试文件

**依赖**: Phase 1 (所有基础设施)
**可并行**: 不同指标可以并行实现
**风险**: 中（KAMA/MAMA 复杂度高）

---

### 任务 2.2: 创建性能基准测试

**任务 ID**: 2.2
**任务名称**: 为所有移动平均指标创建性能基准测试
**优先级**: P1 (高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
使用 Criterion 创建所有移动平均指标的基准测试，比较不同实现的性能。

**子任务**:

- [ ] 创建 `benches/overlap_benchmarks.rs`
- [ ] 为每个指标创建基准测试
- [ ] 测试不同数据量（100, 1000, 10000, 100000）
- [ ] 对比 SIMD vs 标量实现性能

```rust
// benches/overlap_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ta_core::{Sma, Ema, Wma, Indicator};

fn benchmark_sma(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let period = 20;

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        let sma = Sma::new(period);

        c.bench_function(&format!("sma_compute_{}", size), |b| {
            let mut output = vec![0.0; size - period + 1];
            b.iter(|| {
                sma.compute(black_box(&data), black_box(&mut output)).unwrap()
            })
        });
    }
}

fn benchmark_ema(c: &mut Criterion) {
    let sizes = [100, 1000, 10000, 100000];
    let period = 20;

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
        let ema = Ema::new(period);

        c.bench_function(&format!("ema_compute_{}", size), |b| {
            let mut output = vec![0.0; size];
            b.iter(|| {
                ema.compute(black_box(&data), black_box(&mut output)).unwrap()
            })
        });
    }
}

criterion_group!(benches, benchmark_sma, benchmark_ema);
criterion_main!(benches);
```

**验收标准**:
- [ ] 所有 16 个指标都有基准测试
- [ ] 测试多个数据量级别
- [ ] 性能报告生成
- [ ] SIMD 加速比验证通过

**交付物**:
- `benches/overlap_benchmarks.rs`
- 性能报告文档

**依赖**: 任务 2.1 (所有指标实现)
**可并行**: 可以与任务 2.3 并行
**风险**: 低

---

### 任务 2.3: 编写单元测试

**任务 ID**: 2.3
**任务名称**: 为所有移动平均指标编写单元测试
**优先级**: P1 (高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
为所有 16 个移动平均指标编写全面的单元测试，包括：
- 基本功能测试
- 边界条件测试
- 精度验证（与 TA-Lib C 对比）
- SIMD 路径一致性测试

**子任务**:

- [ ] 创建 `tests/overlap_tests.rs`
- [ ] 为每个指标编写测试
- [ ] 添加边界条件测试
- [ ] 添加精度验证测试

```rust
// tests/overlap_tests.rs
use ta_core::{Sma, Ema, Wma, Indicator};

#[test]
fn test_sma_basic() {
    let sma = Sma::new(3);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = sma.compute_to_vec(&data).unwrap();
    assert_eq!(result, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_sma_insufficient_data() {
    let sma = Sma::new(5);
    let data = vec![1.0, 2.0, 3.0];

    assert!(sma.compute_to_vec(&data).is_err());
}

#[test]
fn test_ema_basic() {
    let ema = Ema::new(3);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = ema.compute_to_vec(&data).unwrap();
    // EMA 结果需要与 TA-Lib C 对比验证
    assert!(!result.is_empty());
}
```

**验收标准**:
- [ ] 所有 16 个指标有完整单元测试
- [ ] 基本功能测试覆盖
- [ ] 边界条件测试覆盖
- [ ] 精度验证完成
- [ ] 代码覆盖率 > 90%

**交付物**:
- `tests/overlap_tests.rs`
- 测试覆盖率报告

**依赖**: 任务 2.1 (所有指标实现)
**可并行**: 可以与任务 2.2 并行
**风险**: 低

---

## Phase 3: 动量指标

**阶段名称**: Momentum Indicators
**持续时间**: 4 周（Weeks 11-14）
**目标**: 实现 33 个动量类指标
**依赖**: Phase 1

**阶段进度**:

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 任务 3.1: 实现动量指标系列 | ⬜ 待开始 | - | 33 个指标 |
| 任务 3.2: 创建性能基准测试 | ⬜ 待开始 | - | 对比测试 |
| 任务 3.3: 编写单元测试 | ⬜ 待开始 | - | 覆盖率 > 90% |

**总体进度**: 0/3 任务完成 (0%)

---

### 任务 3.1: 实现动量指标系列

**任务 ID**: 3.1
**任务名称**: 实现 33 个动量类指标
**优先级**: P0 (最高)
**预估工时**: 32 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
实现 TA-Lib 动量指标类中的所有指标，包括 RSI、MACD、ADX 等共 33 个指标。

**指标列表**:

| 分类 | 指标 | 全称 | TA-Lib 函数 |
|------|-------|--------|--------------|
| 趋势 | ADX | Average Directional Index | TA_ADX |
| 趋势 | ADXR | Average Directional Index Rating | TA_ADXR |
| 趋势 | APO | Absolute Price Oscillator | TA_APO |
| 趋势 | AROON | Aroon | TA_AROON |
| 趋势 | AROONOSC | Aroon Oscillator | TA_AROONOSC |
| 趋势 | BOP | Balance Of Power | TA_BOP |
| 趋势 | CCI | Commodity Channel Index | TA_CCI |
| 趋势 | CMO | Chande Momentum Oscillator | TA_CMO |
| 趋势 | DX | Directional Movement Index | TA_DX |
| 趋势 | MACD | Moving Average Convergence Divergence | TA_MACD |
| 趋势 | MACDEXT | MACD with controllable MA type | TA_MACDEXT |
| 趋势 | MACDFIX | Moving Average Convergence Divergence Fix | TA_MACDFIX |
| 趋势 | MFI | Money Flow Index | TA_MFI |
| 趋势 | MINUS_DI | Minus Directional Indicator | TA_MINUS_DI |
| 趋势 | MINUS_DM | Minus Directional Movement | TA_MINUS_DM |
| 趋势 | MOM | Momentum | TA_MOM |
| 趋势 | PLUS_DI | Plus Directional Indicator | TA_PLUS_DI |
| 趋势 | PLUS_DM | Plus Directional Movement | TA_PLUS_DM |
| 趋势 | PPO | Percentage Price Oscillator | TA_PPO |
| 趋势 | ROC | Rate Of Change | TA_ROC |
| 趋势 | ROCP | Rate Of Change Percentage | TA_ROCP |
| 趋势 | ROCR | Rate Of Change Ratio | TA_ROCR |
| 趋势 | RSI | Relative Strength Index | TA_RSI |
| 趋势 | STOCH | Stochastic Oscillator | TA_STOCH |
| 趋势 | STOCHF | Stochastic Fast | TA_STOCHF |
| 趋势 | STOCHRSI | Stochastic RSI | TA_STOCHRSI |
| 趋势 | TRIX | TRIX | TA_TRIX |
| 趋势 | ULTOSC | Ultimate Oscillator | TA_ULTOSC |
| 趋势 | WILLR | Williams' %R | TA_WILLR |
| 波动率 | ATR | Average True Range | TA_ATR |
| 波动率 | NATR | Normalized Average True Range | TA_NATR |
| 波动率 | TRANGE | True Range | TA_TRANGE |

**子任务**:

#### 3.1.1 创建动量指标模块结构

- [ ] 创建 `crates/ta-core/src/momentum/mod.rs`
- [ ] 为每个指标创建独立文件

```
momentum/
├── mod.rs
├── adx.rs
├── macd.rs
├── rsi.rs
├── stoch.rs
├── atr.rs
└── ... (其他 27 个指标)
```

#### 3.1.2 实现关键指标（示例）

**RSI (Relative Strength Index)**:

```rust
//! Relative Strength Index (RSI) - 相对强弱指标

use crate::types::Float;
use crate::{Indicator, Resettable, Result, TalibError};

/// Relative Strength Index
///
/// RSI 测量价格变动的速度和幅度。
///
/// # Example
///
/// ```rust
/// use ta_core::{Rsi, Indicator};
///
/// let data = vec![44.0, 44.25, 43.5, 44.0, 44.5];
/// let mut rsi = Rsi::new(14);
///
/// let result = rsi.compute_to_vec(&data).unwrap();
/// ```
pub struct Rsi {
    period: usize,
    prev_gain: Float,
    prev_loss: Float,
    count: usize,
}

impl Rsi {
    pub fn new(period: usize) -> Self {
        assert!(period > 1, "RSI period must be greater than 1");

        Self {
            period,
            prev_gain: Float::from(0.0),
            prev_loss: Float::from(0.0),
            count: 0,
        }
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

impl Indicator for Rsi {
    fn compute(&self, inputs: &[Float], outputs: &mut [Float]) -> Result<usize> {
        if inputs.len() < self.period {
            return Err(TalibError::InsufficientData {
                required: self.period,
                actual: inputs.len(),
            });
        }

        let output_len = inputs.len() - self.period + 1;
        if outputs.len() < output_len {
            return Err(TalibError::InvalidInput {
                message: format!("Output buffer too small: need {}, got {}", output_len, outputs.len()),
            });
        }

        // RSI 批量计算
        let mut gains = [Float::from(0.0); inputs.len()];
        let mut losses = [Float::from(0.0); inputs.len()];

        for i in 1..inputs.len() {
            let change = inputs[i] - inputs[i - 1];

            if change > Float::from(0.0) {
                gains[i] = change;
                losses[i] = Float::from(0.0);
            } else {
                gains[i] = Float::from(0.0);
                losses[i] = -change;
            }
        }

        // 初始平均增益和损失
        let mut avg_gain = gains[1..=self.period].iter().sum::<Float>() / Float::from(self.period);
        let mut avg_loss = losses[1..=self.period].iter().sum::<Float>() / Float::from(self.period);

        outputs[0] = Float::from(100.0) - (Float::from(100.0) / (Float::from(1.0) + avg_gain / avg_loss));

        for i in (self.period + 1)..inputs.len() {
            avg_gain = (avg_gain * Float::from(self.period - 1) + gains[i]) / Float::from(self.period);
            avg_loss = (avg_loss * Float::from(self.period - 1) + losses[i]) / Float::from(self.period);

            let rs = avg_gain / avg_loss;
            let rsi = Float::from(100.0) - (Float::from(100.0) / (Float::from(1.0) + rs));
            outputs[i - self.period] = rsi;
        }

        Ok(output_len)
    }

    fn next(&mut self, input: Float) -> Option<Float> {
        // RSI 需要历史数据，流式实现更复杂
        // 简化实现：批量计算后取最后一个值
        None
    }

    fn required_output_len(&self, input_len: usize) -> usize {
        if input_len < self.period {
            0
        } else {
            input_len - self.period
        }
    }
}

impl Resettable for Rsi {
    fn reset(&mut self) {
        self.prev_gain = Float::from(0.0);
        self.prev_loss = Float::from(0.0);
        self.count = 0;
    }
}
```

**MACD (Moving Average Convergence Divergence)**:

```rust
//! Moving Average Convergence Divergence (MACD)

use crate::types::Float;
use crate::{Indicator, Resettable, Result, TalibError};

/// MACD 参数
#[derive(Debug, Clone)]
pub struct MacdParams {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
}

impl Default for MacdParams {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}

/// MACD 结果
#[derive(Debug, Clone)]
pub struct MacdResult {
    pub macd: Float,
    pub signal: Float,
    pub histogram: Float,
}

/// Moving Average Convergence Divergence
pub struct Macd {
    params: MacdParams,
    prev_fast_ema: Float,
    prev_slow_ema: Float,
    prev_signal_ema: Float,
}

impl Macd {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            params: MacdParams {
                fast_period,
                slow_period,
                signal_period,
            },
            prev_fast_ema: Float::from(0.0),
            prev_slow_ema: Float::from(0.0),
            prev_signal_ema: Float::from(0.0),
        }
    }

    pub fn with_params(params: MacdParams) -> Self {
        Self {
            params: params.clone(),
            prev_fast_ema: Float::from(0.0),
            prev_slow_ema: Float::from(0.0),
            prev_signal_ema: Float::from(0.0),
        }
    }
}

impl Indicator for Macd {
    fn compute(&self, inputs: &[Float], outputs: &mut [Float]) -> Result<usize> {
        if inputs.len() < self.params.slow_period {
            return Err(TalibError::InsufficientData {
                required: self.params.slow_period,
                actual: inputs.len(),
            });
        }

        // MACD 实现：快线 EMA - 慢线 EMA
        // 信号线：MACD 的 EMA
        // 柱状图：MACD - 信号线

        // 简化实现：使用 EMA 计算
        let mut fast_ema = inputs[0];
        let mut slow_ema = inputs[0];

        for i in 1..inputs.len() {
            let alpha_fast = Float::from(2.0) / Float::from(self.params.fast_period + 1);
            let alpha_slow = Float::from(2.0) / Float::from(self.params.slow_period + 1);

            fast_ema = alpha_fast * inputs[i] + (Float::from(1.0) - alpha_fast) * fast_ema;
            slow_ema = alpha_slow * inputs[i] + (Float::from(1.0) - alpha_slow) * slow_ema;
        }

        // 输出格式：[macd, signal, histogram] × len
        // 实际实现需要计算信号线
        Ok(0)
    }

    fn next(&mut self, input: Float) -> Option<Float> {
        None
    }

    fn required_output_len(&self, input_len: usize) -> usize {
        0
    }
}

impl Resettable for Macd {
    fn reset(&mut self) {
        self.prev_fast_ema = Float::from(0.0);
        self.prev_slow_ema = Float::from(0.0);
        self.prev_signal_ema = Float::from(0.0);
    }
}
```

**验收标准**:
- [ ] 所有 33 个动量指标实现完成
- [ ] 每个指标实现 `Indicator` trait
- [ ] 每个指标实现 `Resettable` trait（适用时）
- [ ] 所有指标有完整文档和示例
- [ ] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/momentum/mod.rs`
- 33 个指标实现文件
- 单元测试文件

**依赖**: Phase 1
**可并行**: 不同指标可以并行实现
**风险**: 中（部分指标复杂度高，如 STOCHRSI、ULTOSC）

---

### 任务 3.2: 创建性能基准测试

**任务 ID**: 3.2
**任务名称**: 为所有动量指标创建性能基准测试
**优先级**: P1 (高)
**预估工时**: 10 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
类似 Phase 2，为所有动量指标创建 Criterion 基准测试。

**子任务**:
- [ ] 创建 `benches/momentum_benchmarks.rs`
- [ ] 为每个指标创建基准测试
- [ ] 测试不同数据量

**验收标准**:
- [ ] 所有 33 个指标都有基准测试
- [ ] 性能报告生成

**交付物**:
- `benches/momentum_benchmarks.rs`
- 性能报告

**依赖**: 任务 3.1
**可并行**: 可以与任务 3.3 并行
**风险**: 低

---

### 任务 3.3: 编写单元测试

**任务 ID**: 3.3
**任务名称**: 为所有动量指标编写单元测试
**优先级**: P1 (高)
**预估工时**: 10 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
为所有 33 个动量指标编写全面的单元测试。

**子任务**:
- [ ] 创建 `tests/momentum_tests.rs`
- [ ] 为每个指标编写测试
- [ ] 添加精度验证

**验收标准**:
- [ ] 所有 33 个指标有完整单元测试
- [ ] 代码覆盖率 > 90%

**交付物**:
- `tests/momentum_tests.rs`
- 测试覆盖率报告

**依赖**: 任务 3.1
**可并行**: 可以与任务 3.2 并行
**风险**: 低

---

## Phase 4: 成交量与波动率

**阶段名称**: Volume & Volatility
**持续时间**: 3 周（Weeks 15-17）
**目标**: 实现 6 个成交量和波动率指标
**依赖**: Phase 1

**阶段进度**:

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 任务 4.1: 实现成交量与波动率指标 | ⬜ 待开始 | - | 6 个指标 |
| 任务 4.2: 创建性能基准测试 | ⬜ 待开始 | - | 对比测试 |
| 任务 4.3: 编写单元测试 | ⬜ 待开始 | - | 覆盖率 > 90% |

**总体进度**: 0/3 任务完成 (0%)

---

### 任务 4.1: 实现成交量与波动率指标

**任务 ID**: 4.1
**任务名称**: 实现 6 个成交量和波动率指标
**优先级**: P0 (最高)
**预估工时**: 12 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
实现 TA-Lib 成交量和波动率指标类中的所有指标。

**指标列表**:

| 分类 | 指标 | 全称 | TA-Lib 函数 |
|------|-------|--------|--------------|
| 成交量 | OBV | On Balance Volume | TA_OBV |
| 成交量 | AD | Chaikin A/D Line | TA_AD |
| 波动率 | NATR | Normalized Average True Range | TA_NATR |
| 波动率 | ATR | Average True Range | TA_ATR |
| 波动率 | TRANGE | True Range | TA_TRANGE |
| 波动率 | SAR | Parabolic SAR | TA_SAR |

**子任务**:

#### 4.1.1 创建成交量与波动率模块结构

- [ ] 创建 `crates/ta-core/src/volume/` 目录
- [ ] 创建 `crates/ta-core/src/volatility/` 目录

```
volume/
├── mod.rs
├── obv.rs
└── ad.rs

volatility/
├── mod.rs
├── atr.rs
├── natr.rs
├── trange.rs
└── sar.rs
```

#### 4.1.2 实现每个指标

**ATR (Average True Range) 示例**:

```rust
//! Average True Range (ATR) - 平均真实波幅

use crate::types::Float;
use crate::{Indicator, Resettable, Result, TalibError};

/// Average True Range
///
/// ATR 测量价格的波动性。
pub struct Atr {
    period: usize,
    prev_close: Float,
    prev_tr: Float,
    prev_atr: Float,
}

impl Atr {
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "ATR period must be greater than 0");

        Self {
            period,
            prev_close: Float::from(0.0),
            prev_tr: Float::from(0.0),
            prev_atr: Float::from(0.0),
        }
    }

    pub fn period(&self) -> usize {
        self.period
    }
}

impl Indicator for Atr {
    fn compute(&self, inputs: &[Float], outputs: &mut [Float]) -> Result<usize> {
        if inputs.len() < self.period {
            return Err(TalibError::InsufficientData {
                required: self.period,
                actual: inputs.len(),
            });
        }

        // ATR 计算：使用 True Range
        // TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        // ATR = SMA(TR, period)

        Ok(0)
    }

    fn next(&mut self, _input: Float) -> Option<Float> {
        None
    }

    fn required_output_len(&self, input_len: usize) -> usize {
        0
    }
}

impl Resettable for Atr {
    fn reset(&mut self) {
        self.prev_close = Float::from(0.0);
        self.prev_tr = Float::from(0.0);
        self.prev_atr = Float::from(0.0);
    }
}
```

**验收标准**:
- [ ] 所有 6 个指标实现完成
- [ ] 每个指标实现 `Indicator` trait
- [ ] 所有指标有完整文档
- [ ] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/volume/mod.rs`
- `crates/ta-core/src/volatility/mod.rs`
- 6 个指标实现文件
- 单元测试文件

**依赖**: Phase 1
**可并行**: 不同指标可以并行实现
**风险**: 低

---

### 任务 4.2: 创建性能基准测试

**任务 ID**: 4.2
**任务名称**: 为成交量和波动率指标创建性能基准测试
**优先级**: P1 (高)
**预估工时**: 6 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**子任务**:
- [ ] 创建基准测试文件
- [ ] 为每个指标创建基准测试

**验收标准**:
- [ ] 所有 6 个指标都有基准测试
- [ ] 性能报告生成

**交付物**:
- `benches/volume_benchmarks.rs`
- `benches/volatility_benchmarks.rs`

**依赖**: 任务 4.1
**可并行**: 可以与任务 4.3 并行
**风险**: 低

---

### 任务 4.3: 编写单元测试

**任务 ID**: 4.3
**任务名称**: 为成交量和波动率指标编写单元测试
**优先级**: P1 (高)
**预估工时**: 6 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**子任务**:
- [ ] 创建单元测试文件
- [ ] 为每个指标编写测试

**验收标准**:
- [ ] 所有 6 个指标有完整单元测试
- [ ] 代码覆盖率 > 90%

**交付物**:
- `tests/volume_tests.rs`
- `tests/volatility_tests.rs`

**依赖**: 任务 4.1
**可并行**: 可以与任务 4.2 并行
**风险**: 低

---

## Phase 5: 绑定层

**阶段名称**: Bindings
**持续时间**: 3 周（Weeks 18-20）
**目标**: 实现 Python 和 WASM 绑定
**依赖**: Phase 1-4

**阶段进度**:

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 任务 5.1: 实现 Python 绑定 | ⬜ 待开始 | - | PyO3 |
| 任务 5.2: 实现 WASM 绑定 | ⬜ 待开始 | - | wasm-bindgen |
| 任务 5.3: 编写绑定层测试 | ⬜ 待开始 | - | 跨平台测试 |

**总体进度**: 0/3 任务完成 (0%)

---

### 任务 5.1: 实现 Python 绑定

**任务 ID**: 5.1
**任务名称**: 使用 PyO3 实现 Python 绑定
**优先级**: P0 (最高)
**预估工时**: 16 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
使用 PyO3 创建 Python 绑定，暴露 Rust 核心库的所有指标。

**子任务**:

#### 5.1.1 配置 Python 绑定

- [ ] 更新 `crates/ta-py/Cargo.toml`

```toml
# crates/ta-py/Cargo.toml
[package]
name = "ta-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "ta_py"
crate-type = ["cdylib"]

[dependencies]
ta-core = { path = "../ta-core" }
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"

[features]
default = ["std"]
std = ["ta-core/std"]
```

#### 5.1.2 实现绑定代码

- [ ] 创建 `crates/ta-py/src/lib.rs`
- [ ] 为所有指标创建 Python 接口

```rust
//! TA-Lib Python Bindings

use pyo3::prelude::*;
use ta_core::{Sma, Ema, Indicator};

/// Simple Moving Average
#[pyclass]
pub struct PySma {
    inner: Sma,
}

#[pymethods]
impl PySma {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: Sma::new(period),
        }
    }

    fn compute(&self, data: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut output = vec![0.0; data.len()];
        self.inner.compute(&data, &mut output)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(output)
    }

    fn compute_to_vec(&self, data: Vec<f64>) -> PyResult<Vec<f64>> {
        self.inner.compute_to_vec(&data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }
}

/// Exponential Moving Average
#[pyclass]
pub struct PyEma {
    inner: Ema,
}

#[pymethods]
impl PyEma {
    #[new]
    fn new(period: usize) -> Self {
        Self {
            inner: Ema::new(period),
        }
    }

    fn compute(&self, data: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut output = vec![0.0; data.len()];
        self.inner.compute(&data, &mut output)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(output)
    }
}

/// TA-Lib Python module
#[pymodule]
fn ta_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySma>()?;
    m.add_class::<PyEma>()?;
    Ok(())
}
```

#### 5.1.3 创建 Python 示例

- [ ] 创建 `examples/python_example.py`

```python
#!/usr/bin/env python3
"""TA-Lib Python 绑定示例"""

import ta_py
import numpy as np

# 创建数据
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# 计算 SMA
sma = ta_py.Sma(period=3)
sma_result = sma.compute(data)
print(f"SMA(3): {sma_result}")

# 计算 EMA
ema = ta_py.Ema(period=3)
ema_result = ema.compute(data)
print(f"EMA(3): {ema_result}")

# NumPy 集成
np_data = np.array(data, dtype=np.float64)
print(f"NumPy integration: {np_data}")
```

**验收标准**:
- [ ] PyO3 绑定配置完成
- [ ] 所有 Phase 2-4 的指标都有 Python 接口
- [ ] Python 示例可以运行
- [ ] 性能测试通过

**交付物**:
- `crates/ta-py/` (完整 Python 绑定)
- `examples/python_example.py`
- Python 使用文档

**依赖**: Phase 1-4
**可并行**: 可以与任务 5.2 并行
**风险**: 中（PyO3 配置复杂）

---

### 任务 5.2: 实现 WASM 绑定

**任务 ID**: 5.2
**任务名称**: 使用 wasm-bindgen 实现 WASM 绑定
**优先级**: P0 (最高)
**预估工时**: 12 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
使用 wasm-bindgen 创建 WASM 绑定，暴露 Rust 核心库的所有指标到 Web。

**子任务**:

#### 5.2.1 配置 WASM 绑定

- [ ] 更新 `crates/ta-wasm/Cargo.toml`

```toml
# crates/ta-wasm/Cargo.toml
[package]
name = "ta-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
ta-core = { path = "../ta-core" }
wasm-bindgen = "0.2"
console_error_panic_hook = "0.1"

[dev-dependencies]
wasm-bindgen-test = "0.3"
```

#### 5.2.2 实现绑定代码

- [ ] 创建 `crates/ta-wasm/src/lib.rs`
- [ ] 为所有指标创建 WASM 接口

```rust
//! TA-Lib WASM Bindings

use wasm_bindgen::prelude::*;
use ta_core::{Sma, Ema, Indicator};

/// Simple Moving Average
#[wasm_bindgen]
pub struct Sma {
    inner: Sma,
}

#[wasm_bindgen]
impl Sma {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Sma {
        Sma {
            inner: Sma::new(period),
        }
    }

    #[wasm_bindgen]
    pub fn compute(&self, data: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; data.len()];
        match self.inner.compute(data, &mut output) {
            Ok(_) => output,
            Err(_) => output, // 简化错误处理
        }
    }

    #[wasm_bindgen]
    pub fn compute_to_vec(&self, data: &[f64]) -> Vec<f64> {
        match self.inner.compute_to_vec(data) {
            Ok(result) => result,
            Err(_) => Vec::new(),
        }
    }
}

/// Exponential Moving Average
#[wasm_bindgen]
pub struct Ema {
    inner: Ema,
}

#[wasm_bindgen]
impl Ema {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> Ema {
        Ema {
            inner: Ema::new(period),
        }
    }

    #[wasm_bindgen]
    pub fn compute(&self, data: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; data.len()];
        match self.inner.compute(data, &mut output) {
            Ok(_) => output,
            Err(_) => output,
        }
    }
}
```

#### 5.2.3 创建 WASM 示例

- [ ] 创建 `examples/wasm_example.html`
- [ ] 创建 `examples/wasm_example.js`

```html
<!-- examples/wasm_example.html -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TA-Lib WASM Example</title>
</head>
<body>
    <h1>TA-Lib WASM 示例</h1>
    <div id="output"></div>

    <script type="module">
        import init, { Sma, Ema } from './ta_wasm.js';

        async function run() {
            await init();

            // 创建数据
            const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

            // 计算 SMA
            const sma = new Sma(3);
            const smaResult = sma.compute(data);
            console.log('SMA(3):', smaResult);

            // 计算 EMA
            const ema = new Ema(3);
            const emaResult = ema.compute(data);
            console.log('EMA(3):', emaResult);

            // 显示结果
            document.getElementById('output').innerHTML =
                `<p>SMA(3): ${smaResult}</p><p>EMA(3): ${emaResult}</p>`;
        }

        run();
    </script>
</body>
</html>
```

**验收标准**:
- [ ] wasm-bindgen 绑定配置完成
- [ ] 所有 Phase 2-4 的指标都有 WASM 接口
- [ ] WASM 示例在浏览器中可以运行
- [ ] 性能测试通过

**交付物**:
- `crates/ta-wasm/` (完整 WASM 绑定)
- `examples/wasm_example.html`
- `examples/wasm_example.js`
- WASM 使用文档

**依赖**: Phase 1-4
**可并行**: 可以与任务 5.1 并行
**风险**: 中（WASM 配置复杂）

---

### 任务 5.3: 编写绑定层测试

**任务 ID**: 5.3
**任务名称**: 为 Python 和 WASM 绑定编写测试
**优先级**: P1 (高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
为 Python 和 WASM 绑定编写跨平台测试，确保绑定正常工作。

**子任务**:

- [ ] 创建 Python 绑定测试
- [ ] 创建 WASM 绑定测试
- [ ] 跨平台一致性测试

**验收标准**:
- [ ] Python 绑定测试通过
- [ ] WASM 绑定测试通过
- [ ] 跨平台结果一致

**交付物**:
- `crates/ta-py/tests/`
- `crates/ta-wasm/tests/`
- 跨平台测试文档

**依赖**: 任务 5.1, 5.2
**可并行**: 无
**风险**: 中

---

## Phase 6: 验证与优化

**阶段名称**: Validation & Optimization
**持续时间**: 4 周（Weeks 21-24）
**目标**: 完整测试套件、性能优化、文档完善
**依赖**: Phase 1-5

**阶段进度**:

| 任务 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 任务 6.1: 全面测试 | ⬜ 待开始 | - | 所有平台 |
| 任务 6.2: 性能优化 | ⬜ 待开始 | - | SIMD 优化 |
| 任务 6.3: 文档完善 | ⬜ 待开始 | - | 用户文档 |
| 任务 6.4: 发布准备 | ⬜ 待开始 | - | 0.1.0 版本 |

**总体进度**: 0/4 任务完成 (0%)

---

### 任务 6.1: 全面测试

**任务 ID**: 6.1
**任务名称**: 运行所有平台的全面测试
**优先级**: P0 (最高)
**预估工时**: 16 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
在所有支持平台上运行完整的测试套件，验证功能和性能。

**子任务**:

- [ ] 运行 x86_64 测试
- [ ] 运行 ARM64 测试
- [ ] 运行 WASM 测试
- [ ] 验证跨平台结果一致性
- [ ] 性能基准对比

**验收标准**:
- [ ] 所有平台测试通过
- [ ] 跨平台结果一致
- [ ] 性能基准达标

**交付物**:
- 测试报告
- 性能报告
- 跨平台验证报告

**依赖**: Phase 1-5
**可并行**: 可以与任务 6.2 部分并行
**风险**: 中（跨平台测试复杂）

---

### 任务 6.2: 性能优化

**任务 ID**: 6.2
**任务名称**: 优化所有指标的性能
**优先级**: P1 (高)
**预估工时**: 16 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
分析性能瓶颈，优化所有指标实现，确保 SIMD 加速有效。

**子任务**:

- [ ] 分析性能瓶颈
- [ ] 优化热点路径
- [ ] 验证 SIMD 加速效果
- [ ] 对比 TA-Lib C 性能

**验收标准**:
- [ ] 性能瓶颈识别完成
- [ ] 优化后性能提升 > 20%
- [ ] SIMD 加速比验证通过
- [ ] 性能接近或超越 TA-Lib C

**交付物**:
- 性能分析报告
- 优化后的代码
- 性能对比报告

**依赖**: 任务 6.1 (全面测试)
**可并行**: 可以与任务 6.3 并行
**风险**: 中

---

### 任务 6.3: 文档完善

**任务 ID**: 6.3
**任务名称**: 编写完整的用户文档
**优先级**: P1 (高)
**预估工时**: 12 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
编写完整的用户文档，包括 API 文档、使用指南、性能调优指南。

**子任务**:

- [ ] 完善 rustdoc API 文档
- [ ] 编写 Python 绑定文档
- [ ] 编写 WASM 绑定文档
- [ ] 编写性能调优指南
- [ ] 编写迁移指南

**验收标准**:
- [ ] rustdoc 文档完整
- [ ] Python 文档完整
- [ ] WASM 文档完整
- [ ] 示例代码可运行
- [ ] 文档覆盖率 > 95%

**交付物**:
- API 文档
- 用户指南
- 性能调优指南
- 迁移指南
- 示例代码

**依赖**: Phase 1-5
**可并行**: 可以与任务 6.2 并行
**风险**: 低

---

### 任务 6.4: 发布准备

**任务 ID**: 6.4
**任务名称**: 准备 0.1.0 版本发布
**优先级**: P0 (最高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ⬜ 待开始

**描述**:
准备 0.1.0 版本的发布，包括版本号更新、CHANGELOG、CI 配置。

**子任务**:

- [ ] 更新版本号到 0.1.0
- [ ] 编写 CHANGELOG.md
- [ ] 配置 GitHub Actions 自动发布
- [ ] 准备 crates.io 发布
- [ ] 准备 PyPI 发布
- [ ] 准备 npm 发布

**验收标准**:
- [ ] 版本号更新完成
- [ ] CHANGELOG 完整
- [ ] CI 发布配置完成
- [ ] 所有目标平台可以发布

**交付物**:
- 更新后的 Cargo.toml
- CHANGELOG.md
- GitHub Actions 发布配置
- 发布检查清单

**依赖**: 任务 6.1-6.3
**可并行**: 无
**风险**: 低

---

## 风险管理

### 技术风险

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| AVX-512 硬件可用性有限 | 高 | 中 | 使用 std::arch 直接实现，不依赖 wide；提供标量回退 |
| WASM SIMD128 限制 | 中 | 低 | 提前测试，必要时回退到标量 |
| 跨平台测试困难 | 中 | 中 | 使用 CI 和模拟器，重点测试 x86_64 和 ARM64 |
| 性能未达预期 | 低 | 高 | 早期基准测试，及时调整算法 |
| SIMD 和标量结果精度不一致 | 中 | 高 | 严格的单元测试和一致性验证 |

### 项目风险

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| 工期延误 | 中 | 高 | 分阶段交付，优先完成关键路径 |
| 资源不足 | 低 | 高 | 提前规划资源分配，必要时调整范围 |
| 需求变更 | 中 | 中 | 保持架构灵活，易于适应变更 |

---

## 质量保证

### 代码质量标准

- [ ] 代码覆盖率 > 90%
- [ ] 所有 clippy lint 通过
- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 性能基准测试通过

### 测试策略

1. **单元测试**: 测试每个函数和模块
2. **集成测试**: 测试跨模块交互
3. **性能测试**: 确保 SIMD 加速效果
4. **跨平台测试**: 验证所有平台一致性
5. **精度测试**: 与 TA-Lib C 实现对比

### 文档要求

- [ ] 所有公共 API 有 rustdoc 文档
- [ ] 所有示例代码可以运行
- [ ] 架构设计文档完整
- [ ] 用户使用指南完整

---

## 附录

### A. 修订记录

#### v3.0 (2026-01-30) - 完整合并版

**变更摘要**:
1. **整合两份计划**: 合并 IMPLEMENTATION_PLAN.md v2.0 和 simd-implementation.md
2. **补充完整 Phase 2-6**: 包含所有指标实现的详细任务
3. **统一架构决策**: 明确使用手动调度（不使用 multiversion）
4. **完整项目范围**: 6 个阶段，28 周，150+ 指标
5. **详细任务分解**: 36 个任务，每个都有详细步骤
6. **完整的验收标准**: 每个任务都有明确的验收标准

**新增内容**:
- Phase 2: 重叠研究指标（16 个指标）
- Phase 3: 动量指标（33 个指标）
- Phase 4: 成交量与波动率（6 个指标）
- Phase 5: 绑定层（Python + WASM）
- Phase 6: 验证与优化（4 个任务）

**影响评估**:
- **风险**: 中等（架构清晰，任务详细）
- **收益**: 极高（完整的实施路线图）
- **工作量**: 28 周（约 7 个月）
- **兼容性**: 100% 向后兼容

### B. 术语表

**核心术语**:

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
- **OnceLock**: Rust 的线程安全一次性初始化原语

**技术分析术语**:

- **MA**: Moving Average - 移动平均
- **EMA**: Exponential Moving Average - 指数移动平均
- **RSI**: Relative Strength Index - 相对强弱指标
- **MACD**: Moving Average Convergence Divergence - 平滑异同移动平均线
- **ATR**: Average True Range - 平均真实波幅
- **OBV**: On Balance Volume - 能量潮指标
- **ADX**: Average Directional Index - 平均趋向指标
- **Momentum**: 动量 - 价格变动的速度
- **Volatility**: 波动率 - 价格变动的幅度

**Rust 特定术语**:

- **trait**: Rust 的特性定义，类似于其他语言的接口
- **crate**: Rust 的包/库单元
- **workspace**: Rust 的工作空间，可以包含多个 crate
- **feature**: Rust 的条件编译特性
- **no_std**: 不使用标准库的编译配置
- **zero-copy**: 零拷贝 - 不需要内存拷贝的 API 设计

---

**计划创建**: Sisyphus (Merger)
**最后更新**: 2026-01-30
**版本**: 3.0
