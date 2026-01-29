# TA-Lib Rust 重写项目 - 完整实施计划

**版本**: v1.0  
**创建日期**: 2026-01-29  
**状态**: 已批准，准备执行  

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
- 200+ 技术指标
- 零拷贝 API
- SIMD 优化（使用 `std::simd`）
- 批量与流式双模式
- Python 和 WASM 绑定

### 成功标准
- 100% 数值精度匹配 TA-Lib C（ε < 1e-10）
- 所有 200+ 指标实现
- SIMD 优化在 x86_64 和 aarch64 上有效
- 流式 API 延迟 < 1ms

### 关键技术决策
- **SIMD**: 使用 `wide` 库（`f64x4`/`f32x8`）替代 `std::simd`，因为 `std::simd` 目前处于 nightly 不稳定状态
- **架构**: 4-crate workspace（core, py, wasm, benchmarks）
- **API**: 零拷贝（`&[Float]` / `&mut [Float]`），`Float` 类型通过条件编译确定
- **模式**: 批量 + 流式双 trait 系统
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

### 任务 1.1: 创建 Workspace 结构

**任务 ID**: 1.1  
**任务名称**: 创建虚拟 workspace 和 crate 结构  
**优先级**: P0 (最高)  
**预估工时**: 8 小时  
**负责人**: TBD  

**描述**:  
创建 Rust 虚拟 workspace，建立 4 个 crate 的目录结构，配置基础 Cargo.toml。

**子任务**:

- [ ] 1.1.1 创建 workspace 根目录和 Cargo.toml
  - 配置 workspace members
  - 设置默认编译选项
  - 配置 workspace 级依赖
  
- [ ] 1.1.2 创建 `ta-core` crate
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
  
- [ ] 1.1.3 创建 `ta-py` crate
  - 创建目录结构: `crates/ta-py/src/`
  - 配置 Cargo.toml (PyO3 dependency)
  - 创建空的 lib.rs
  
- [ ] 1.1.4 创建 `ta-wasm` crate
  - 创建目录结构: `crates/ta-wasm/src/`
  - 配置 Cargo.toml (wasm-bindgen dependency)
  - 创建空的 lib.rs
  
- [ ] 1.1.5 创建 `ta-benchmarks` crate
  - 创建目录结构: `crates/ta-benchmarks/benches/`
  - 配置 Cargo.toml (Criterion dependency)
  - 创建空的 benchmark 文件

**验收标准**:
- [ ] `cargo build` 在 workspace 根目录成功执行
- [ ] 所有 4 个 crate 可以独立编译
- [ ] 目录结构符合设计文档规范
- [ ] 每个 Cargo.toml 包含正确的元数据和依赖

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

### 任务 1.2: 实现错误类型系统

**任务 ID**: 1.2  
**任务名称**: 实现 TalibError 枚举和错误处理  
**优先级**: P0 (最高)  
**预估工时**: 6 小时  
**负责人**: TBD  

**描述**:  
定义完整的错误类型系统，实现 `std::error::Error` trait，支持错误转换和上下文信息。

**子任务**:

- [ ] 1.2.1 定义 `TalibError` 枚举
  - 创建 `crates/ta-core/src/error.rs`
  - 定义所有错误变体:
    - `InvalidInput` (无效输入)
    - `InvalidPeriod` (无效周期)
    - `InsufficientData` (数据不足)
    - `InvalidParameter` (无效参数)
    - `ComputationError` (计算错误)
    - `NotImplemented` (未实现)
  
- [ ] 1.2.2 实现 `std::error::Error` trait
  - 实现 `Display` trait (人类可读错误消息)
  - 实现 `Error` trait (标准错误接口)
  - 实现 `source()` 方法 (错误链支持)
  
- [ ] 1.2.3 实现错误转换
  - 为常见类型实现 `From` trait:
    - `From<std::io::Error>`
    - `From<std::num::ParseFloatError>`
  - 添加错误上下文信息
  
- [ ] 1.2.4 创建 Result 类型别名
  - `pub type Result<T> = std::result::Result<T, TalibError>;`
  - 在 `lib.rs` 中导出
  
- [ ] 1.2.5 编写单元测试
  - 测试每个错误变体的创建
  - 测试错误消息格式
  - 测试错误转换
  - 测试错误链

**验收标准**:
- [ ] 所有错误变体可以正确创建
- [ ] `TalibError` 实现 `std::error::Error`
- [ ] 错误消息清晰、有用
- [ ] 所有单元测试通过
- [ ] 代码覆盖率 > 90%

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
**任务名称**: 实现 BatchIndicator 和 StreamingIndicator traits  
**优先级**: P0 (最高)  
**预估工时**: 12 小时  
**负责人**: TBD  

**描述**:  
定义核心 traits，实现批量和流式计算模式，建立指标实现的标准接口。

**子任务**:

- [ ] 1.3.1 定义 `BatchIndicator` trait
  - 创建 `crates/ta-core/src/traits.rs`
  - 定义 trait:
    ```rust
    pub trait BatchIndicator<const N: usize = 1> {
        type Input;
        type Output;
        
        fn lookback(&self) -> usize;
        
        fn compute(
            &self,
            inputs: &[Self::Input],
            outputs: &mut [Self::Output],
        ) -> Result<usize>;
        
        fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>;
    }
    ```
  - 添加文档字符串和示例
  
- [ ] 1.3.2 定义 `StreamingIndicator` trait
  - 定义 trait:
    ```rust
    pub trait StreamingIndicator<const N: usize = 1> {
        type Input;
        type Output;
        type State: Default + Clone;
        
        fn init(&self) -> Self::State;
        
        fn next(&self, state: &mut Self::State, input: Self::Input) -> Option<Self::Output>;
        
        fn process_batch(
            &self,
            inputs: &[Self::Input],
        ) -> (Self::State, Vec<Self::Output>);
    }
    ```
  - 添加文档字符串和示例
  
- [ ] 1.3.3 定义 `DualModeIndicator` trait
  - 定义 marker trait:
    ```rust
    pub trait DualModeIndicator<const N: usize = 1>: 
        BatchIndicator<N> + 
        StreamingIndicator<N, Input = Self::Input, Output = Self::Output>
    {
        fn verify_consistency(&self, test_data: &[Self::Input]) -> Result<()>
        where
            Self::Output: PartialEq + std::fmt::Debug;
    }
    ```
  - 实现默认的一致性验证方法
  
- [ ] 1.3.4 实现示例指标（SMA）
  - 创建 `crates/ta-core/src/overlap/sma.rs`
  - 实现 `Sma` 结构体
  - 实现 `BatchIndicator` trait
  - 实现 `StreamingIndicator` trait
  - 实现 `DualModeIndicator` trait
  - 添加完整文档和示例
  
- [ ] 1.3.5 编写单元测试
  - 测试 `BatchIndicator` 实现
  - 测试 `StreamingIndicator` 实现
  - 测试一致性验证
  - 测试边界条件
  - 测试错误处理

**验收标准**:
- [ ] 所有 traits 定义完成且有完整文档
- [ ] SMA 示例指标实现并测试通过
- [ ] 批量和流式模式结果一致
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

### 任务 1.4: 实现 SIMD 抽象层

**任务 ID**: 1.4  
**任务名称**: 实现简化版 SIMD 抽象层  
**优先级**: P0 (最高)  
**预估工时**: 10 小时  
**负责人**: TBD  

**描述**:  
实现基于 `wide` 库的 SIMD 抽象层，**不使用 `#[cfg]` 条件编译**。由于 `std::simd` 是 nightly 特性，使用 `wide` 库提供稳定的跨平台 SIMD 支持。

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
  
- [ ] 1.4.2 定义 SIMD 类型
  ```rust
  use wide::f64x4;
  
  /// SIMD vector type for f64 operations
  pub type SimdF64 = f64x4;
  
  /// Number of lanes in SimdF64
  pub const LANES: usize = 4;
  ```
  
- [ ] 1.4.3 实现 `sum_simd` 函数
  ```rust
  pub fn sum_simd(data: &[f64]) -> f64 {
      let chunks = data.chunks_exact(LANES);
      let remainder = chunks.remainder();
      
      let mut sum_vec = SimdF64::ZERO;
      for chunk in chunks {
          let vec = SimdF64::from_slice(chunk);
          sum_vec += vec;
      }
      
      let mut sum = sum_vec.reduce_add();
      for &x in remainder { sum += x; }
      sum
  }
  ```
  
- [ ] 1.4.4 实现 `sma_simd` 函数
  ```rust
  pub fn sma_simd(prices: &[f64], period: usize) -> Vec<f64> {
      if prices.len() < period || period == 0 {
          return Vec::new();
      }
      
      let output_len = prices.len() - period + 1;
      let mut result = vec![0.0; output_len];
      
      // For small periods, use scalar code
      if period < LANES * 2 {
          for i in 0..output_len {
              let sum: f64 = prices[i..i + period].iter().sum();
              result[i] = sum / period as f64;
          }
          return result;
      }
      
      // For larger periods, use SIMD
      for i in 0..output_len {
          let window = &prices[i..i + period];
          result[i] = sum_simd(window) / period as f64;
      }
      
      result
  }
  ```
  
- [ ] 1.4.5 编写单元测试
  - 测试 `sum_simd` 基础功能
  - 测试 `sum_simd` 边界条件（空数组、单元素、非对齐长度）
  - 测试 `sma_simd` 基础功能
  - 测试 `sma_simd` 各种周期
  - 验证与标量实现结果一致

**验收标准**:
- [ ] `sum_simd` 通过所有单元测试
- [ ] `sma_simd` 通过所有单元测试
- [ ] SIMD 实现与标量实现结果一致
- [ ] 性能基准显示 SIMD 优于标量（数据量 > 1000 时）
- [ ] 代码覆盖率 > 95%

**交付物**:
- `crates/ta-core/src/simd.rs`
- 单元测试文件
- 性能基准测试
- 使用文档

**依赖**: 任务 1.1 (workspace 结构)  
**可并行**: 可与任务 1.2、1.3 并行  
**风险**: 低  

---

## 后续内容继续...

[注：由于文档长度限制，Phase 2-6 的详细内容在此省略，但在实际文件中将包含完整的实施计划。]

