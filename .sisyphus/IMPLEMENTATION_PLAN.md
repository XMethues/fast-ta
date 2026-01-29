# TA-Lib Rust 重写项目 - 完整实施计划

**版本**: v1.5
**创建日期**: 2026-01-29
**最后更新**: 2026-01-29
**状态**: 执行中 (Phase 1: 2/5 任务完成)  

---

## 目录

1. [项目概述](#项目概述)
2. [实施阶段总览](#实施阶段总览)
3. [Phase 1: 核心基础设施](#phase-1-核心基础设施)
4. [Phase 2: 重叠研究指标](#phase-2-重叠研究指标)
5. [Phase 3: 动量指标](#phase-3-动量指标)
6. [Phase 4: 成交量与波动率](#phase-4-成交量与波动率)
7. [Phase 5: 绑定层](#phase-5-绑定层)
8. [Phase 6: 验证与优化](#phase-6-验证与优化)
9. [风险管理](#风险管理)
10. [质量保证](#质量保证)

---

## 项目概述

### 项目目标
将 TA-Lib（技术分析库）从 C/C++ 移植到 Rust，实现：
- 150+ 技术指标
- 零拷贝 API
- SIMD 优化（使用 `std::simd`）
- 批量与流式双模式
- Python 和 WASM 绑定

### 成功标准
- 100% 数值精度匹配 TA-Lib C（ε < 1e-10）
- 所有 150+ 指标实现
- SIMD 优化在 x86_64 和 aarch64 上有效
- 流式 API 延迟 < 1ms

### 关键技术决策
- **SIMD**: 使用 `wide` 库（`f64x4`/`f32x8`）替代 `std::simd`，因为 `std::simd` 目前处于 nightly 不稳定状态
- **架构**: 4-crate workspace（core, py, wasm, benchmarks）
- **API**: 零拷贝（`&[Float]` / `&mut [Float]`），`Float` 类型通过条件编译确定
- **指标接口**: 统一 `Indicator` trait，采用混合方案（性能 + 易用性）
  - `compute(&self, inputs, outputs)` - **零拷贝批量计算**（性能优先，无内存分配）
  - `compute_to_vec(&self, inputs)` - **便捷批量计算**（易用性优先，返回 Vec）
  - `next(&mut self, input)` - 获取最新值（实时流式）
  - `stream(&mut self, inputs)` - 流式处理多个输入
- **浮点精度**: 通过 Cargo features 在编译时选择 `f32` 或 `f64`（默认 `f64`），使用条件编译实现零成本抽象

---

## 实施阶段总览

| 阶段 | 名称 | 持续时间 | 主要交付物 | 依赖 |
|------|------|----------|------------|------|
| **Phase 1** | 核心基础设施 | 4 周 | Workspace, traits, SIMD 层 | 无 |
| **Phase 2** | 重叠研究指标 | 4 周 | 16 个移动平均指标 | Phase 1 |
| **Phase 3** | 动量指标 | 4 周 | 33 个动量指标 | Phase 1 |
| **Phase 4** | 成交量与波动率 | 3 周 | 6 个指标 | Phase 1 |
| **Phase 5** | 绑定层 | 3 周 | Python 和 WASM 绑定 | Phase 1-4 |
| **Phase 6** | 验证与优化 | 4 周 | 完整测试套件，性能优化 | Phase 1-5 |

**总工期**: 26 周（约 6 个月）

**关键路径**: Phase 1 → Phase 2 → Phase 6

**可并行**: Phase 2, 3, 4 可以并行执行

---

## Phase 1: 核心基础设施

**阶段名称**: Core Infrastructure
**持续时间**: 4 周（Weeks 1-4）
**目标**: 建立项目基础架构，实现核心 traits 和 SIMD 抽象层
**依赖**: 无

**阶段进度**:

| 任务 | 状态 | 完成日期 |
|------|------|----------|
| 任务 1.1: 创建 Workspace 结构 | ✅ 已完成 | 2026-01-29 |
| 任务 1.2: 实现错误类型系统 | ✅ 已完成 | 2026-01-29 |
| 任务 1.3: 实现核心 Traits | ⬜ 待开始 | - |
| 任务 1.4: 实现 SIMD 抽象层 | ⬜ 待开始 | - |
| 任务 1.5: 设置测试基础设施 | ⬜ 待开始 | - |

**总体进度**: 2/5 任务完成 (40%)  

### 任务 1.1: 创建 Workspace 结构 ✅

**任务 ID**: 1.1
**任务名称**: 创建虚拟 workspace 和 crate 结构
**优先级**: P0 (最高)
**预估工时**: 8 小时
**负责人**: TBD
**状态**: ✅ 已完成 (2026-01-29)

**描述**:
创建 Rust 虚拟 workspace，建立 4 个 crate 的目录结构，配置基础 Cargo.toml。

**子任务**:

- [x] 1.1.1 创建 workspace 根目录和 Cargo.toml
  - 配置 workspace members
  - 设置默认编译选项
  - 配置 workspace 级依赖
  - 添加 `wide = "0.7"` 到 `[workspace.dependencies]`

- [x] 1.1.2 创建 `ta-core` crate
  - 创建目录结构: `crates/ta-core/src/`
  - 配置 Cargo.toml (no_std, core dependencies)
  - **配置浮点精度特性**: 添加 `f32` 和 `f64` 特性（默认 `f64`）
    ```toml
    [features]
    default = ["f64"]
    f32 = []
    f64 = []
    ```
  - **定义条件编译浮点类型**: 在 `crates/ta-core/src/types.rs` 中定义：
    ```rust
    #[cfg(feature = "f32")]
    pub type Float = f32;

    #[cfg(feature = "f64")]
    pub type Float = f64;

    #[cfg(feature = "f32")]
    pub use wide::f32x8 as SimdFloat;

    #[cfg(feature = "f64")]
    pub use wide::f64x4 as SimdFloat;

    #[cfg(feature = "f32")]
    pub const LANES: usize = 8;

    #[cfg(feature = "f64")]
    pub const LANES: usize = 4;
    ```
  - 创建空的 lib.rs

- [x] 1.1.3 创建 `ta-py` crate
  - 创建目录结构: `crates/ta-py/src/`
  - 配置 Cargo.toml (PyO3 dependency)
  - 创建空的 lib.rs

- [x] 1.1.4 创建 `ta-wasm` crate
  - 创建目录结构: `crates/ta-wasm/src/`
  - 配置 Cargo.toml (wasm-bindgen dependency)
  - 创建空的 lib.rs

- [x] 1.1.5 创建 `ta-benchmarks` crate
  - 创建目录结构: `crates/ta-benchmarks/benches/`
  - 配置 Cargo.toml (Criterion dependency)
  - 创建空的 benchmark 文件

**验收标准**:
- [x] `cargo build` 在 workspace 根目录成功执行
- [x] 所有 4 个 crate 可以独立编译
- [x] 目录结构符合设计文档规范
- [x] 每个 Cargo.toml 包含正确的元数据和依赖

**交付物**:
- `/Cargo.toml` (workspace root)
- `/crates/ta-core/Cargo.toml`
- `/crates/ta-py/Cargo.toml`
- `/crates/ta-wasm/Cargo.toml`
- `/crates/ta-benchmarks/Cargo.toml`
- 对应的 `src/` 和 `benches/` 目录

**依赖**: 无  
**可并行**: 子任务 1.1.2 - 1.1.5 可以并行  
**风险**: 低  

---

### 任务 1.2: 实现错误类型系统 ✅

**任务 ID**: 1.2
**任务名称**: 实现 TalibError 枚举和错误处理
**优先级**: P0 (最高)
**预估工时**: 6 小时
**负责人**: TBD
**状态**: ✅ 已完成 (2026-01-29)

**描述**:
定义完整的错误类型系统，实现 `std::error::Error` trait，支持错误转换和上下文信息。

**子任务**:

- [x] 1.2.1 定义 `TalibError` 枚举
  - 创建 `crates/ta-core/src/error.rs`
  - 定义所有错误变体:
    - `InvalidInput` (无效输入)
    - `InvalidPeriod` (无效周期)
    - `InsufficientData` (数据不足)
    - `InvalidParameter` (无效参数)
    - `ComputationError` (计算错误)
    - `NotImplemented` (未实现)

- [x] 1.2.2 实现 `std::error::Error` trait
  - 实现 `Display` trait (人类可读错误消息)
  - 实现 `Error` trait (标准错误接口)
  - 实现 `source()` 方法 (错误链支持)

- [x] 1.2.3 实现错误转换
  - 为常见类型实现 `From` trait:
    - `From<std::io::Error>`
    - `From<std::num::ParseFloatError>`
    - 额外添加: `From<std::num::ParseIntError>`
  - 添加错误上下文信息

- [x] 1.2.4 创建 Result 类型别名
  - `pub type Result<T> = core::result::Result<T, TalibError>;`
  - 在 `lib.rs` 中导出

- [x] 1.2.5 编写单元测试
  - 测试每个错误变体的创建
  - 测试错误消息格式
  - 测试错误转换
  - 测试错误链
  - ✅ 19 个单元测试全部通过

**验收标准**:
- [x] 所有错误变体可以正确创建
- [x] `TalibError` 实现 `std::error::Error`
- [x] 错误消息清晰、有用
- [x] 所有单元测试通过 (19/19)
- [x] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/error.rs`
- 单元测试文件
- 文档字符串

**依赖**: 任务 1.1 (workspace 结构)  
**可并行**: 无  
**风险**: 低  

---

### 任务 1.3: 实现核心 Traits

**任务 ID**: 1.3
**任务名称**: 实现统一 Indicator trait（支持批量、流式和单值查询）
**优先级**: P0 (最高)
**预估工时**: 12 小时
**负责人**: TBD

**描述**:
定义统一的 `Indicator` trait，同时支持批量计算、获取最新值和流式处理，建立指标实现的标准接口。

**设计理念**:
- 一个统一的 trait 支持三种使用模式
- 采用混合方案（方案3）：同时提供零拷贝核心接口和便捷包装接口
- 零拷贝 `compute()` 方法（性能优先）：使用输出缓冲区，无内存分配
- 便捷 `compute_to_vec()` 方法（易用性优先）：返回 Vec，自动管理内存
- 用户可以根据场景选择合适的接口

**子任务**:

- [ ] 1.3.1 定义 `Indicator` trait
  - 创建 `crates/ta-core/src/traits.rs`
  - 定义统一 trait:
    ```rust
    pub trait Indicator<const N: usize = 1> {
        type Input;
        type Output;

        /// 返回产生第一个有效输出所需的历史数据长度
        fn lookback(&self) -> usize;

        /// 零拷贝批量计算（性能优先）
        ///
        /// # Arguments
        /// * `inputs` - 输入数据
        /// * `outputs` - 输出缓冲区，必须足够大：`outputs.len() >= inputs.len() - lookback()`
        ///
        /// # Returns
        /// 实际写入的输出数量
        ///
        /// # Performance
        /// 零内存分配，适合高频调用场景
        ///
        /// # Example
        /// ```rust
        /// let sma = Sma::new(20)?;
        /// let prices: &[Float] = &large_price_array;
        ///
        /// // 预先分配输出缓冲区（可重用）
        /// let mut outputs = vec![0.0; prices.len() - 19];
        ///
        /// // 零拷贝计算
        /// let count = sma.compute(prices, &mut outputs)?;
        /// ```
        fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) -> Result<usize>;

        /// 便捷批量计算（易用性优先）
        ///
        /// # Returns
        /// 包含所有计算结果的 Vec
        ///
        /// # Performance
        /// 需要一次内存分配，适合一般使用场景
        ///
        /// # Example
        /// ```rust
        /// let sma = Sma::new(20)?;
        /// let prices = &[1.0, 2.0, 3.0, 4.0, 5.0];
        ///
        /// // 简单直接
        /// let results = sma.compute_to_vec(prices)?;
        /// ```
        fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
            let lookback = self.lookback();
            if inputs.len() <= lookback {
                return Ok(Vec::new());
            }

            let mut outputs = vec![Self::Output::default(); inputs.len() - lookback];
            let count = self.compute(inputs, &mut outputs)?;
            debug_assert_eq!(count, outputs.len());
            Ok(outputs)
        }

        /// 获取最新值（流式模式）
        ///
        /// # Returns
        /// `Some(output)` 如果有足够的累积数据
        /// `None` 如果数据不足
        ///
        /// # Example
        /// ```rust
        /// let mut sma = Sma::new(20)?;
        ///
        /// for price in prices {
        ///     if let Some(value) = sma.next(price) {
        ///         println!("SMA: {}", value);
        ///     }
        /// }
        /// ```
        fn next(&mut self, input: Self::Input) -> Option<Self::Output>;

        /// 流式处理（批量流式模式）
        ///
        /// # Example
        /// ```rust
        /// let mut sma = Sma::new(20)?;
        /// let prices = &[1.0, 2.0, 3.0, 4.0, 5.0];
        ///
        /// let results: Vec<_> = sma.stream(prices);
        /// // results: [None, None, None, None, None, ...]
        /// ```
        fn stream(&mut self, inputs: &[Self::Input]) -> Vec<Option<Self::Output>> {
            inputs.iter().map(|&input| self.next(input)).collect()
        }
    }
    ```
  - 添加文档字符串和示例

- [ ] 1.3.2 定义辅助 trait
  - 定义 `Resettable` trait（可选，用于重置指标状态）:
    ```rust
    pub trait Resettable {
        fn reset(&mut self);
    }
    ```
  - 添加文档字符串

- [ ] 1.3.3 实现示例指标（SMA）
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
    - 核心方法 `compute()`：**必须使用 `wide` SIMD 加速**
      - 对于小窗口（period < LANES * 2）：使用标量实现
      - 对于大窗口：使用 SIMD（`SimdFloat`）进行向量化计算
      - 使用 `types::SimdFloat`（条件编译：`f32x8` 或 `f64x4`）
      - 使用 `types::LANES` 常量（8 或 4）
    - `compute_to_vec()`：直接使用 trait 默认实现（复用 `compute()`）
    - `next()`：滚动和算法，支持流式处理
    - `stream()`：直接使用 trait 默认实现
  - 实现 `Resettable` trait
  - **性能要求**：SIMD 加速比 > 2x（数据量 > 1000，period >= LANES * 2）
  - 添加完整文档和示例，说明SIMD优化策略

- [ ] 1.3.4 编写单元测试
  - 测试批量计算模式 (`compute`)
  - 测试流式模式 (`next`, `stream`)
  - 测试边界条件（空数组、单元素、非对齐长度）
  - 测试错误处理
  - 测试 `Resettable` trait

- [ ] 1.3.5 添加示例代码和文档
  - 批量计算示例
  - 流式处理示例
  - 单值查询示例
  - 性能最佳实践说明

**验收标准**:
- [ ] `Indicator` trait 定义完成且有完整文档
- [ ] `Resettable` trait 定义完成（可选）
- [ ] SMA 示例指标实现并测试通过
- [ ] 零拷贝 `compute()` 接口正常工作（性能验证）
- [ ] 便捷 `compute_to_vec()` 接口正常工作（易用性验证）
- [ ] 流式 `next()` 和 `stream()` 接口正常工作
- [ ] 三种模式（批量、流式、单值查询）都能正常工作
- [ ] 所有单元测试通过
- [ ] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/traits.rs`
- `crates/ta-core/src/overlap/sma.rs`
- 单元测试文件
- 文档和示例

**依赖**: 任务 1.2 (错误类型系统)  
**可并行**: 部分子任务可并行  
**风险**: 中等 (traits 设计需要仔细考虑)  

---

### 任务 1.4: 实现 SIMD 辅助函数

**任务 ID**: 1.4
**任务名称**: 实现 SIMD 辅助函数（为指标提供SIMD支持）
**优先级**: P0 (最高)
**预估工时**: 10 小时
**负责人**: TBD

**描述**:
实现基于 `wide` 库的 SIMD 辅助函数，为所有指标实现提供 SIMD 加速支持。使用 `types.rs` 中定义的 `SimdFloat` 和 `LANES` 常量，**不使用 `#[cfg]` 条件编译**。由于 `std::simd` 是 nightly 特性，使用 `wide` 库提供稳定的跨平台 SIMD 支持。

**依赖添加**:  
在 workspace 根目录的 Cargo.toml 的 `[workspace.dependencies]` 中添加:  
```toml
wide = "0.7"
```

在 `ta-core` 的 Cargo.toml 的 `[dependencies]` 中添加:  
```toml
wide.workspace = true
```

**子任务**:

- [ ] 1.4.1 创建 `simd.rs` 模块
  - 创建 `crates/ta-core/src/simd.rs`
  - 添加模块文档
  - 说明使用 `wide` 库而不是 `std::simd` 的原因
  - 说明依赖 `types.rs` 中的 `SimdFloat` 和 `LANES` 常量

- [ ] 1.4.2 实现 SIMD 辅助函数
  ```rust
  use crate::types::{Float, SimdFloat, LANES};

  /// 使用 SIMD 计算数组和
  ///
  /// # Arguments
  /// * `data` - 输入数据
  ///
  /// # Returns
  /// 所有元素的和
  ///
  /// # Example
  /// ```rust
  /// let data: &[Float] = &[1.0, 2.0, 3.0, 4.0];
  /// let sum = sum_simd(data);
  /// ```
  pub fn sum_simd(data: &[Float]) -> Float {
      let chunks = data.chunks_exact(LANES);
      let remainder = chunks.remainder();

      let mut sum_vec = SimdFloat::splat(0.0);
      for chunk in chunks {
          let vec = SimdFloat::from_slice(chunk);
          sum_vec += vec;
      }

      let mut sum = sum_vec.reduce_sum();
      for &x in remainder { sum += x; }
      sum
  }
  ```

- [ ] 1.4.3 实现 SIMD 加速的求和函数（针对SMA）
  - 优化SMA场景的滚动和计算
  - 提供批量滑动窗口和计算
  - 使用 `sum_simd()` 作为基础

- [ ] 1.4.4 编写单元测试
  - 测试 `sum_simd` 基础功能
  - 测试 `sum_simd` 边界条件（空数组、单元素、非对齐长度）
  - 测试与标量实现结果一致
  - 性能基准测试（SIMD vs 标量）

**验收标准**:
- [ ] `sum_simd` 通过所有单元测试
- [ ] SIMD 实现与标量实现结果一致
- [ ] 性能基准显示 SIMD 优于标量（数据量 > 1000 时）
- [ ] 代码覆盖率 > 95%

**交付物**:
- `crates/ta-core/src/simd.rs`
- 单元测试文件
- 性能基准测试
- 使用文档

**依赖**: 任务 1.1 (workspace 结构) - ✅ 已完成
**被依赖**: 任务 1.3.3 (SMA 实现) 需要使用本任务提供的 SIMD 函数
**可并行**: 无（需要 task 1.1 完成）
**风险**: 低

---

### 任务 1.5: 设置测试基础设施

**任务 ID**: 1.5
**任务名称**: 设置测试基础设施和工具
**优先级**: P1 (高)
**预估工时**: 6 小时
**负责人**: TBD

**描述**:
配置测试框架、工具和 CI/CD 管道，确保代码质量和测试覆盖率。

**子任务**:

- [ ] 1.5.1 配置测试框架
  - 在 workspace 根目录的 `Cargo.toml` 添加测试依赖：
    ```toml
    [workspace.dev-dependencies]
    criterion.workspace = true
    ```
  - 配置 `cargo-tarpaulin`（代码覆盖率工具）
  - 配置测试配置（测试超时、并行度等）

- [ ] 1.5.2 配置代码质量工具
  - 在 workspace 根目录的 `Cargo.toml` 添加 lint 配置：
    ```toml
    [workspace.lints.rust]
    unsafe_code = "forbid"
    missing_docs = "warn"
    rust_2018_idioms = "warn"

    [workspace.lints.clippy]
    all = "warn"
    pedantic = "warn"
    nursery = "warn"
    missing_errors_doc = "allow"
    missing_panics_doc = "allow"
    ```
  - 配置 `rustfmt`（代码格式化）
  - 配置 `.rustfmt.toml`

- [ ] 1.5.3 创建 CI/CD 配置
  - 创建 `.github/workflows/ci.yml`（GitHub Actions）
  - 配置测试矩阵：
    - 稳定 Rust 版本（1.78+）
    - f32 和 f64 特性
  - 添加步骤：
    - 格式检查（`rustfmt -- --check`）
    - Clippy 检查
    - 编译检查（`cargo build`）
    - 单元测试（`cargo test`）
    - 代码覆盖率（`cargo tarpaulin`）
    - 发布编译（`cargo build --release`）

- [ ] 1.5.4 配置性能基准测试
  - 配置 Criterion 基准测试框架
  - 创建基准测试模板
  - 添加基准结果存储和对比配置

- [ ] 1.5.5 创建测试辅助模块
  - 创建 `crates/ta-core/src/test_utils.rs`
  - 提供测试数据生成函数
  - 提供测试辅助函数（随机数据、边界数据等）
  - 实现与 TA-Lib C 对比测试的基础设施

**验收标准**:
- [ ] 测试框架配置完成
- [ ] 代码覆盖率工具可以正常运行
- [ ] CI/CD 管道配置完成
- [ ] 所有 lint 规则配置完成
- [ ] 基准测试可以运行
- [ ] 测试辅助模块完成

**交付物**:
- `.github/workflows/ci.yml`
- `.rustfmt.toml`
- `Cargo.toml`（更新测试依赖和 lint 配置）
- `crates/ta-core/src/test_utils.rs`
- CI/CD 配置文档

**依赖**: 任务 1.1 (workspace 结构), 任务 1.2 (错误类型系统), 任务 1.3 (核心 Traits), 任务 1.4 (SIMD 辅助函数)
**可并行**: 可以与任务 1.3 和 1.4 并行
**风险**: 低

---

## Phase 1: 核心基础设施 - 总结

**阶段名称**: Core Infrastructure
**持续时间**: 4 周（Weeks 1-4）
**目标**: 建立项目基础架构，实现核心 traits 和 SIMD 抽象层

**阶段任务**:

| 任务 | 状态 | 完成日期 |
|------|------|----------|
| 任务 1.1: 创建 Workspace 结构 | ✅ 已完成 | 2026-01-29 |
| 任务 1.2: 实现错误类型系统 | ✅ 已完成 | 2026-01-29 |
| 任务 1.3: 实现核心 Traits | ⬜ 待开始 | - |
| 任务 1.4: 实现 SIMD 辅助函数 | ⬜ 待开始 | - |
| 任务 1.5: 设置测试基础设施 | ⬜ 待开始 | - |

**总体进度**: 2/5 任务完成 (40%)

**阶段交付物**:
- [ ] 完整的 Workspace 结构
- [ ] 错误类型系统（19个测试通过）
- [ ] 统一的 `Indicator` trait（支持批量、流式、单值查询）
- [ ] SIMD 辅助函数（使用 `wide` 库）
- [ ] 测试基础设施（CI/CD、代码覆盖率、基准测试）

**阶段验收标准**:
- [ ] Workspace 可以成功编译（`cargo build`）
- [ ] 所有核心 traits 有完整文档
- [ ] SIMD 层通过所有单元测试
- [ ] 测试覆盖率 > 80%
- [ ] CI/CD 管道可以正常运行

---

## Phase 2: 重叠研究指标 (Overlap Studies)

**阶段名称**: Overlap Studies  
**持续时间**: 4 周（Weeks 5-8）  
**目标**: 实现 16 个移动平均和重叠研究指标  
**依赖**: Phase 1 (核心基础设施)  

### 任务 2.1: 实现简单移动平均 (SMA)

**任务 ID**: 2.1  
**优先级**: P0  
**预估工时**: 8 小时  

**描述**:  
实现简单移动平均 (Simple Moving Average) 指标，作为所有移动平均指标的模板实现。

**子任务**:
- [ ] 2.1.1 实现 `Sma` 结构体
  - 定义 `Sma` 结构体（使用之前 1.3.3 设计的结构）
  - 实现 `new()` 构造函数
  - 实现参数验证

- [ ] 2.1.2 实现 `Indicator` trait
  - 实现 `compute()`：使用 SIMD 加速
  - 实现 `next()`：流式处理
  - 实现 `lookback()`：返回 `period - 1`
  - 实现 `Resettable` trait

- [ ] 2.1.3 编写单元测试
  - 测试 `compute()` 基本功能
  - 测试 `next()` 流式功能
  - 测试边界条件
  - 测试与 TA-Lib C 数值一致性（ε < 1e-10）

- [ ] 2.1.4 添加文档和示例
  - 添加 Rust doc 文档
  - 添加使用示例
  - 添加性能说明

**验收标准**:
- [ ] SMA 实现完成并通过所有测试
- [ ] 数值精度与 TA-Lib C 一致（ε < 1e-10）
- [ ] SIMD 加速比 > 2x（数据量 > 1000）
- [ ] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/overlap/sma.rs`
- 单元测试文件
- 文档和示例

**依赖**: 任务 1.3 (核心 Traits), 任务 1.4 (SIMD 辅助函数)
**可并行**: 可以与任务 2.2-2.4 并行（其他重叠研究指标）
**风险**: 低

---

### 任务 2.2: 实现指数移动平均 (EMA)

**任务 ID**: 2.2  
**优先级**: P0  
**预估工时**: 6 小时  

**描述**:  
实现指数移动平均 (Exponential Moving Average) 指标。

**子任务**:
- [ ] 2.2.1 实现 `Ema` 结构体
  - 定义 `Ema` 结构体
  - 实现 `new()` 构造函数
  - 实现参数验证

- [ ] 2.2.2 实现 `Indicator` trait
  - 实现 `compute()`：递归公式实现
  - 实现 `next()`：流式处理
  - 实现 `lookback()`：返回 0（EMA 有值）
  - 实现 `Resettable` trait

- [ ] 2.2.3 编写单元测试
  - 测试基本功能
  - 测试流式功能
  - 测试与 TA-Lib C 数值一致性

- [ ] 2.2.4 添加文档和示例

**验收标准**:
- [ ] EMA 实现完成并通过所有测试
- [ ] 数值精度与 TA-Lib C 一致（ε < 1e-10）
- [ ] 代码覆盖率 > 90%

**交付物**:
- `crates/ta-core/src/overlap/ema.rs`
- 单元测试文件
- 文档和示例

**依赖**: 任务 2.1 (SMA - 作为模板参考)
**可并行**: 可以与任务 2.3, 2.4 等并行
**风险**: 低

---

[注：由于篇幅限制，任务 2.3-2.16 的详细内容遵循相同模式，包括：
- 2.3: WMA (加权移动平均)
- 2.4: DEMA (双重指数移动平均)
- 2.5: TEMA (三重指数移动平均)
- 2.6: TRIMA (三角移动平均)
- 2.7: KAMA (Kaufman自适应移动平均)
- 2.8: MAMA (MESA自适应移动平均)
- 2.9: T3 (T3移动平均)
- 2.10: BBANDS (布林带)
- 2.11: SAR (抛物线转向)
- 2.12: SAR_EXT (扩展抛物线转向)
- 2.13: HT_TRENDLINE (希尔伯特变换趋势线)
- 2.14: HT_SINE (希尔伯特变换正弦波)
- 2.15: HT_TRENDMODE (希尔伯特变换趋势模式)
- 2.16: HT_DCPHASE (希尔伯特变换主导周期相位)
]

---

## Phase 3: 动量指标 (Momentum Indicators)

**阶段名称**: Momentum Indicators  
**持续时间**: 4 周（Weeks 9-12）  
**目标**: 实现 33 个动量和震荡指标  
**依赖**: Phase 1 (核心基础设施)  

[注：详细任务遵循 Phase 2 模式，包括：
- 3.1: MACD
- 3.2: MACD_EXT
- 3.3: MACD_FIX
- 3.4: RSI
- 3.5: STOCH
- ... (共33个指标)
]

---

## Phase 4: 成交量与波动率 (Volume & Volatility)

**阶段名称**: Volume & Volatility Indicators  
**持续时间**: 3 周（Weeks 13-15）  
**目标**: 实现 6 个成交量和波动率指标  
**依赖**: Phase 1-3  

[注：详细任务包括：
- 4.1: OBV (On Balance Volume)
- 4.2: AD (Chaikin A/D Line)
- 4.3: ATR (Average True Range)
- 4.4: NATR (Normalized ATR)
- 4.5: TRANGE (True Range)
- 4.6: ADOSC (Chaikin A/D Oscillator)
]

---

## Phase 5: 绑定层 (Bindings)

**阶段名称**: Language Bindings  
**持续时间**: 3 周（Weeks 16-18）  
**目标**: 实现 Python 和 WASM 绑定  
**依赖**: Phase 1-4  

### 任务 5.1: 实现 Python 绑定

**任务 ID**: 5.1  
**优先级**: P1  
**预估工时**: 16 小时  

**描述**:  
使用 PyO3 实现 Python 绑定，让 Python 用户可以调用 Rust 实现的技术指标。

**子任务**:
- [ ] 5.1.1 配置 PyO3 环境
  - 更新 `ta-py/Cargo.toml` 添加 PyO3 依赖
  - 配置 Python 模块入口
  - 配置 maturin（Python 包构建工具）

- [ ] 5.1.2 实现核心模块绑定
  - 创建 `ta-py/src/lib.rs`
  - 实现模块初始化
  - 导出核心类型和函数

- [ ] 5.1.3 实现指标函数绑定
  - 为每个指标创建 Python 函数包装
  - 处理 Python 类型转换（list ↔ Vec）
  - 处理错误转换（Rust Error → Python Exception）

- [ ] 5.1.4 实现高级 API
  - 创建 Pythonic 的 API 设计
  - 支持 NumPy 数组输入/输出（可选）
  - 支持 Pandas Series/DataFrame（可选）

- [ ] 5.1.5 编写 Python 测试
  - 使用 pytest 编写测试
  - 测试所有绑定的指标函数
  - 测试错误处理
  - 测试数值精度

- [ ] 5.1.6 编写文档和示例
  - 编写 Python API 文档
  - 创建使用示例
  - 创建 Jupyter Notebook 示例
  - 编写安装指南

**验收标准**:
- [ ] Python 包可以成功构建（`maturin build`）
- [ ] Python 包可以成功安装（`pip install`）
- [ ] 所有绑定的指标可以在 Python 中调用
- [ ] Python 测试全部通过
- [ ] 数值精度与 Rust 实现一致（ε < 1e-10）

**交付物**:
- `ta-py/src/lib.rs`（Python 绑定实现）
- `ta-py/Cargo.toml`（PyO3 配置）
- Python 测试文件
- Python 文档和示例
- `pyproject.toml`（Python 包配置）

**依赖**: 任务 1.1-1.5 (Phase 1 完成), 任务 2-4 (指标实现完成)
**可并行**: 可以与任务 5.2 (WASM 绑定) 并行
**风险**: 中（PyO3 绑定复杂性）

---

### 任务 5.2: 实现 WASM 绑定

**任务 ID**: 5.2  
**优先级**: P1  
**预估工时**: 16 小时  

**描述**:  
使用 wasm-bindgen 实现 WebAssembly 绑定，让 JavaScript/TypeScript 用户可以在浏览器或 Node.js 中调用 Rust 实现的技术指标。

**子任务**:
- [ ] 5.2.1 配置 wasm-bindgen 环境
  - 更新 `ta-wasm/Cargo.toml` 添加 wasm-bindgen 依赖
  - 配置 `wasm-pack` 工具
  - 配置 WebAssembly 模块入口

- [ ] 5.2.2 实现核心模块绑定
  - 创建 `ta-wasm/src/lib.rs`
  - 实现模块初始化
  - 使用 `#[wasm_bindgen]` 导出函数
  - 配置 panic hook（用于调试）

- [ ] 5.2.3 实现指标函数绑定
  - 为每个指标创建 JavaScript 函数包装
  - 处理类型转换（JavaScript Array ↔ Rust Vec）
  - 处理错误转换（Rust Error → JavaScript Error/exception）
  - 支持 Float32Array 和 Float64Array（可选）

- [ ] 5.2.4 实现高级 API
  - 创建 JavaScript/TypeScript 友好的 API 设计
  - 支持 TypeScript 类型定义（.d.ts）
  - 支持 npm 包发布配置

- [ ] 5.2.5 编写 JavaScript 测试
  - 使用 Jest 或 Mocha 编写测试
  - 测试所有绑定的指标函数
  - 测试错误处理
  - 测试数值精度

- [ ] 5.2.6 编写文档和示例
  - 编写 JavaScript API 文档
  - 创建使用示例（HTML + JavaScript）
  - 创建 Node.js 示例
  - 编写安装和使用指南

**验收标准**:
- [ ] WebAssembly 模块可以成功构建（`wasm-pack build`）
- [ ] npm 包可以成功发布（`wasm-pack publish` 或手动）
- [ ] 所有绑定的指标可以在 JavaScript 中调用
- [ ] JavaScript 测试全部通过
- [ ] 数值精度与 Rust 实现一致（ε < 1e-10）
- [ ] 在主流浏览器中正常工作（Chrome, Firefox, Safari, Edge）

**交付物**:
- `ta-wasm/src/lib.rs`（WASM 绑定实现）
- `ta-wasm/Cargo.toml`（wasm-bindgen 配置）
- JavaScript/TypeScript 测试文件
- TypeScript 类型定义文件（`.d.ts`）
- `package.json`（npm 包配置）
- 文档和示例

**依赖**: 任务 1.1-1.5 (Phase 1 完成), 任务 2-4 (指标实现完成)
**可并行**: 可以与任务 5.1 (Python 绑定) 并行
**风险**: 中（WASM 绑定和浏览器兼容性）

---

## Phase 6: 验证与优化 (Validation & Optimization)

**阶段名称**: Validation & Optimization  
**持续时间**: 4 周（Weeks 23-26）  
**目标**: 完成全面测试、性能优化和文档完善  
**依赖**: Phase 1-5  

### 任务 6.1: 完成全面测试

**任务 ID**: 6.1  
**优先级**: P0  
**预估工时**: 40 小时  

**描述**:  
完成所有指标的全面测试，包括单元测试、集成测试、对比测试和性能测试。

**子任务**:
- [ ] 6.1.1 完成单元测试
  - 为所有 150+ 指标编写单元测试
  - 测试边界条件（空输入、单元素、最大值、最小值）
  - 测试错误处理（无效参数、溢出等）
  - 目标：代码覆盖率 > 95%

- [ ] 6.1.2 完成集成测试
  - 测试指标组合使用
  - 测试批量计算和流式计算一致性
  - 测试多线程安全性（如果适用）

- [ ] 6.1.3 完成对比测试
  - 与 TA-Lib C 对比所有指标
  - 确保数值精度 ε < 1e-10
  - 记录任何不一致性并分析原因

- [ ] 6.1.4 完成性能测试
  - 测试所有指标的性能
  - 测试不同数据量的性能
  - 测试 SIMD 加速效果
  - 生成性能基准报告

**验收标准**:
- [ ] 所有 150+ 指标有单元测试
- [ ] 代码覆盖率 > 95%
- [ ] 所有对比测试通过（ε < 1e-10）
- [ ] 性能基准报告完成

**交付物**:
- 完整的测试套件
- 测试报告
- 性能基准报告

**依赖**: 任务 2-5 (所有指标实现完成)
**可并行**: 无
**风险**: 中（测试工作量大）

---

### 任务 6.2: 性能优化

**任务 ID**: 6.2  
**优先级**: P1  
**预估工时**: 32 小时  

**描述**:  
基于性能测试结果，对所有指标进行性能优化。

**子任务**:
- [ ] 6.2.1 分析性能瓶颈
  - 分析性能测试报告
  - 识别性能瓶颈
  - 确定优化优先级

- [ ] 6.2.2 优化 SIMD 使用
  - 优化 SIMD 向量化
  - 优化内存访问模式
  - 确保 cache 友好

- [ ] 6.2.3 优化算法实现
  - 优化滚动窗口计算
  - 优化递归公式实现
  - 减少不必要的计算

- [ ] 6.2.4 验证优化效果
  - 重新运行性能测试
  - 验证性能提升
  - 确保数值精度不变

**验收标准**:
- [ ] 所有指标性能提升 > 20%（相对于优化前）
- [ ] SIMD 指标加速比 > 2x
- [ ] 数值精度保持不变（ε < 1e-10）

**交付物**:
- 优化后的指标实现
- 性能优化报告

**依赖**: 任务 6.1 (性能测试完成)
**可并行**: 无
**风险**: 中（优化可能影响精度）

---

### 任务 6.3: 完善文档

**任务 ID**: 6.3  
**优先级**: P1  
**预估工时**: 24 小时  

**描述**:  
完善所有文档，包括 API 文档、用户指南、开发者文档等。

**子任务**:
- [ ] 6.3.1 完善 API 文档
  - 为所有公共 API 添加完整文档
  - 添加使用示例
  - 确保文档覆盖率 > 95%

- [ ] 6.3.2 编写用户指南
  - 编写快速入门指南
  - 编写详细使用指南
  - 编写示例教程

- [ ] 6.3.3 编写开发者文档
  - 编写架构设计文档
  - 编写贡献指南
  - 编写内部实现文档

- [ ] 6.3.4 编写迁移指南
  - 编写从 TA-Lib C 迁移指南
  - 编写从其他库迁移指南

- [ ] 6.3.5 创建示例项目
  - 创建完整示例项目
  - 创建 Jupyter Notebook 示例（Python）
  - 创建 Web 示例（WASM）

**验收标准**:
- [ ] 所有公共 API 有完整文档
- [ ] 文档覆盖率 > 95%
- [ ] 用户指南完成
- [ ] 开发者文档完成
- [ ] 示例项目完成

**交付物**:
- 完整的 API 文档（`cargo doc` 生成）
- 用户指南
- 开发者文档
- 示例项目
- 迁移指南

**依赖**: 任务 2-5 (所有指标实现完成)
**可并行**: 可以与任务 6.1, 6.2 并行
**风险**: 低

---

## Phase 1-6 实施计划总结

**项目总工期**: 26 周（约 6 个月）

**阶段总览**:

| 阶段 | 名称 | 周数 | 主要交付物 |
|------|------|------|------------|
| Phase 1 | 核心基础设施 | 4 | Workspace, Traits, SIMD 层, 测试框架 |
| Phase 2 | 重叠研究指标 | 4 | 16 个移动平均指标 |
| Phase 3 | 动量指标 | 4 | 33 个动量指标 |
| Phase 4 | 成交量与波动率 | 3 | 6 个指标 |
| Phase 5 | 绑定层 | 3 | Python 和 WASM 绑定 |
| Phase 6 | 验证与优化 | 4 | 完整测试套件，性能优化，文档 |

**关键路径**: Phase 1 → Phase 2 → Phase 3 → Phase 6

**可并行任务**:
- Phase 2, 3, 4 可以部分并行
- Phase 5 可以与 Phase 2-4 并行
- Phase 6 中的文档编写可以与 Phase 2-5 并行

**项目总交付物**:
- 150+ 技术指标实现
- Python 绑定（PyO3）
- WASM 绑定（wasm-bindgen）
- 完整测试套件（>95% 覆盖率）
- 完整文档（API 文档、用户指南、开发者文档）
- 性能基准报告
- CI/CD 管道

**项目成功标准**:
- [ ] 所有 150+ 指标实现
- [ ] 100% 数值精度匹配 TA-Lib C（ε < 1e-10）
- [ ] 所有测试通过（>95% 覆盖率）
- [ ] Python 和 WASM 绑定正常工作
- [ ] 性能优于 TA-Lib C（>20%）
- [ ] 完整文档完成

---

**文档版本**: v1.5  
**最后更新**: 2026-01-29  
**状态**: 执行中 (Phase 1: 2/5 任务完成)


