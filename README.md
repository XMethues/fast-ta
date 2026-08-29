# fast-ta

[![CI](https://github.com/XMethues/fast-ta/actions/workflows/ci.yml/badge.svg)](https://github.com/XMethues/fast-ta/actions/workflows/ci.yml)
[![Rust 1.89+](https://img.shields.io/badge/rust-1.89%2B-blue.svg)](https://www.rust-lang.org/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#许可证)

面向批处理与实时数据流的 Rust 技术分析库。`fast-ta` 已覆盖固定目录中的全部
**161 个 TA-Lib 命名指标**，并为每个指标提供一致的 Rust-first 执行契约。

> 当前版本为 `0.2.1`。指标目录与数学语义以 TA-Lib 为参考，但本项目不复刻
> TA-Lib C API，也不承诺 C ABI 兼容。

## 主要能力

- **完整指标目录**：覆盖 Cycle、Math Operators、Math Transform、Momentum、
  Overlap Studies、Pattern Recognition、Price Transform、Statistic、Volatility
  和 Volume 十类指标。
- **四种执行模式**：Owned Compact Output、caller-owned Batch、可复用的
  Prepared Batch Runner，以及具备独立状态的 Streaming Computation。
- **明确的输出对齐**：`OutputRange` 同时给出首个有效输入位置和有效元素数，
  不使用隐式填充或哨兵值。
- **双精度选择**：默认使用 `f64`，也可构建为 `f32`。
- **跨环境支持**：支持 `std`、`no_std`、WebAssembly，并提供 Python crate。
- **运行时 SIMD**：在支持的平台选择 x86 AVX2/AVX-512、AArch64 NEON 或
  WASM SIMD128，否则回退到 scalar 实现。
- **语义优先的性能验证**：跨语言 benchmark 会先检查输出范围、元素数量、
  浮点结果和 Pattern Signal，再记录计时结果。

## 快速开始

### 构建与测试

```bash
# 构建默认 workspace 成员
cargo build

# 测试核心库
cargo test -p fast-ta

# 使用 f32
cargo test -p fast-ta --no-default-features --features f32,std

# 检查 no_std
cargo check -p fast-ta --lib --no-default-features --features f64,core_error
```

`ta-py` 需要本机 Python，因此不属于默认 workspace 构建：

```bash
cargo build -p ta-py
```

更多构建方式见 [BUILD.md](BUILD.md)。

### Rust 示例

```rust
use fast_ta::{
    overlap::SMAConfig,
    IndicatorConfig,
    StreamingComputation,
};

fn main() -> fast_ta::Result<()> {
    let prices = [1.0, 2.0, 3.0, 4.0, 5.0];
    let config = SMAConfig::new(3)?;

    // 分配并返回紧凑结果；有效值对应原输入索引 2..5。
    let result = config.compute(&prices)?;
    assert_eq!(result.range().beg_idx, 2);
    assert_eq!(result.values(), &[2.0, 3.0, 4.0]);

    // caller-owned 输出，适合控制分配。
    let mut output = [0.0; 3];
    let range = config.compute_into(&prices, &mut output)?;
    assert_eq!(range.nb_element, 3);

    // 每个 stream 独立持有 warm-up 与滚动状态。
    let mut stream = config.stream()?;
    assert_eq!(stream.next(1.0)?, None);
    assert_eq!(stream.next(2.0)?, None);
    assert_eq!(stream.next(3.0)?, Some(2.0));

    Ok(())
}
```

Prepared Batch Runner 可为固定容量预先准备状态，并在重复调用之间复用：

```rust
use fast_ta::{overlap::SMAConfig, IndicatorConfig, PreparedBatchRunner};

let config = SMAConfig::new(14)?;
let mut runner = config.prepare_batch(4096)?;
let mut output = vec![0.0; prices.len() - config.lookback()];
let range = runner.compute_into(&prices, &mut output)?;
# Ok::<(), fast_ta::TalibError>(())
```

## 执行模型

| 模式 | 输出所有权 | 适用场景 |
|---|---|---|
| Owned Compact Output | 库分配 | 简洁的一次性计算 |
| caller-owned Batch | 调用方提供 | 可控分配、低延迟批处理 |
| Prepared Batch Runner | 调用方提供并复用 | 重复计算、固定容量 worker |
| Streaming Computation | 每个 stream 独立状态 | 实时 tick/candle 数据 |

所有模式共享同一不可变 Indicator Configuration。验证失败时，caller-owned
输出不会被部分改写；Prepared Runner 超出预设容量时会返回明确错误。

## 精度与验证

性能矩阵不是“先计时、后假设正确”。每个可比较 case 在计时前都会验证：

- Rust 与 TA-Lib C 使用相同参数和确定性输入；
- `output_begin`、`output_count` 与每列长度一致；
- 浮点输出逐元素满足对应 `f64`/`f32` 精度容差；
- Pattern Recognition 的整数信号逐元素精确一致；
- Owned、caller-owned、Prepared 与 Streaming 模式结果一致。

当前固定代表性矩阵包含 15 个指标、3 种输入长度和 6 条执行路径，共
**270/270 条通过语义验证并完成测量**。完整方法、环境和置信区间见
[CATALOGUE_MATRIX_REPORT.txt](crates/ta-benchmarks/CATALOGUE_MATRIX_REPORT.txt)。

## 性能概览

Apple M2、`f64`、release/bench profile、TA-Lib 0.6.4 的固定结果中，
65,536 个观测值下 caller-owned Rust/C 延迟比为：

| 指标 | fast-ta / TA-Lib C | 结论 |
|---|---:|---|
| MACD | 0.462x | fast-ta 更快 |
| SMA | 0.594x | fast-ta 更快 |
| BBANDS | 0.639x | fast-ta 更快 |
| RSI | 0.929x | fast-ta 更快 |
| ATR | 1.013x | 性能持平 |
| ADX | 1.062x | 接近持平 |
| HT_DCPHASE | 1.234x | TA-Lib C 更快 |
| CDL3WHITESOLDIERS | 4.396x | TA-Lib C 更快 |

15 个代表性指标的 caller-owned 几何比为 `1.207x`。该结果是特定机器上的
可复现参考，不代表所有平台；只有相同输入、参数、输出边界与执行模式的
Rust/C 行才进入汇总。运行完整固定矩阵：

```bash
python3 crates/ta-benchmarks/scripts/run_catalogue_matrix.py \
  --python /path/to/python3.12
```

依赖、固定版本、发布规则和报告重建命令见 [BUILD.md](BUILD.md)。

## Workspace

```text
fast-ta/
├── crates/
│   ├── fast-ta/        # 指标、执行契约、验证与 SIMD
│   ├── ta-py/          # Python 绑定
│   ├── ta-wasm/        # wasm-bindgen 绑定
│   └── ta-benchmarks/  # Criterion 与跨语言性能矩阵
├── docs/adr/           # 系统级架构决策
├── BUILD.md            # 构建、测试和 benchmark 指南
└── CONTEXT.md          # 领域词汇与边界
```

## 支持矩阵

| 能力 | 状态 |
|---|---|
| 默认 `f64 + std` | 支持 |
| `f32 + std` | 支持 |
| `f64/f32 + no_std` | 支持 |
| x86_64 Linux | CI 验证 |
| AArch64 macOS | CI 验证 |
| wasm32 | CI 验证 |
| Python crate | 需要本机 Python |

最低支持 Rust 版本为 **1.89**。

## 许可证

项目 crate 元数据采用 **MIT OR Apache-2.0** 双重许可。

## 致谢

指标命名和参考语义来自 [TA-Lib](https://ta-lib.org/)；固定比较使用
TA-Lib 0.6.4 官方 C 实现及其 Python binding。
