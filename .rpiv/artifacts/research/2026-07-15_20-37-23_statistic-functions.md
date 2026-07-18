---
date: 2026-07-15T20:37:23+0800
author: unknown
commit: ffd2451
branch: main
repository: fast-ta
topic: "实现 Statistic Functions（统计函数），共 9 个"
tags: [research, codebase, statistic-functions, rolling-statistics, linear-regression]
status: ready
last_updated: 2026-07-15T20:37:23+0800
last_updated_by: unknown
---

# Research: 实现 Statistic Functions（统计函数），共 9 个

## Research Question

实现 Statistic Functions（统计函数），共 9 个：

- VAR
- STDDEV
- CORREL
- BETA
- LINEARREG
- LINEARREG_SLOPE
- LINEARREG_INTERCEPT
- LINEARREG_ANGLE
- TSF

推荐顺序：

1. 公共 rolling statistics / regression helpers
2. VAR、STDDEV
3. CORREL、BETA
4. 5 个线性回归函数
5. Inventory、测试、Criterion benchmarks

## Summary

`ta-core` 当前没有 `statistic` 源码模块、统计数值测试或统计 benchmark；九个函数仅在 inventory 中登记为 `Planned`。现有架构已经提供完整的 TA-Lib 风格协议：batch kernel 紧凑写入调用方缓冲区并返回 `OutputRange`，`*_vec` wrapper 生成等长、前导 NaN 的 padded 输出，同名 struct 实现 batch 与 streaming surface。

九个函数可以在不修改 `Indicator`、`StreamingIndicator`、`TalibError` 或公共输出协议的前提下接入。VAR/STDDEV 共享 rolling variance moments；CORREL/BETA 共享 paired moments 的形状，但 BETA 对相邻价格收益率运算并具有不同 lookback；五个回归函数共享 slope/intercept，只在最终投影上不同。

开发者已选择 TA-Lib 语义 parity、O(n) rolling 算法、官方 period 边界、`statistic` 自有的 `PairInput`/`PairTick`，以及九个函数的 compact 与 vec Criterion 覆盖。正常数学退化（常量窗口、零方差、零分母）属于合法输出，不映射为 `TalibError`。

## Detailed Findings

### Existing Output and Module Contracts

- `OutputRange` 对应 TA-Lib 的 `outBegIdx`/`outNBElement`：kernel 从输出缓冲区索引 0 开始紧凑写入，`beg_idx` 只描述结果在原输入中的位置（`crates/ta-core/src/common.rs:16-27`）。
- 标准 period 窗口的 lookback 是 `period - 1`，输出数是 `input_len.saturating_sub(lookback)`（`crates/ta-core/src/common.rs:81-104`）。BETA 是例外，其 lookback 为 `period`。
- 空输入且 period 合法时，`validate_input_len()` 返回零输出；非空但不足一个窗口时返回 `InsufficientData`（`crates/ta-core/src/common.rs:106-114`）。空结果规范化为 `OutputRange::empty() == (0, 0)`（`crates/ta-core/src/common.rs:39-46`）。
- 输出缓冲区只需容纳 compact count；不足时返回 `InvalidInput`，大于 count 的尾部不属于有效输出（`crates/ta-core/src/common.rs:116-126`）。
- `padded_from_compact()` 创建与输入同长的 vector，Float padding 为 NaN，并把 compact 前缀复制到 `beg_idx` 起始的位置（`crates/ta-core/src/common.rs:61-71`, `crates/ta-core/src/common.rs:179-200`）。
- `Indicator` 已明确分离 borrowed input、compact mutable output 和 owned padded output；统计函数不需要第二套输出协议（`crates/ta-core/src/traits.rs:28-53`）。
- 根模块当前公开 overlap、price transform、volatility、volume、math transform 和 math operators，但没有 `statistic`（`crates/ta-core/src/lib.rs:32-50`）。inventory 中的 `rust_module() == "statistic"` 只是元数据，不会自动创建 Rust 模块（`crates/ta-core/src/inventory.rs:81-99`）。
- 现有 group facade 使用私有叶子模块和显式公开重导出；算法共享 helper 可以保持在 group 内部，而无需进入 crate-root helper facade（`crates/ta-core/src/price_transform/mod.rs:6-16`）。

### VAR and STDDEV

- `AVGDEV` 是最接近的完整 API 先例：period、finite input、input count、output capacity、compact kernel、padded wrapper、同名 struct、streaming ring 和 reset 都已存在（`crates/ta-core/src/price_transform/avgdev.rs:16-149`）。
- `SUM` 是 O(n) batch rolling 先例：首窗口求和一次，后续仅加入新值并减去旧值（`crates/ta-core/src/math_operators/rolling.rs:81-97`）。`AVGDEV` 则每个输出重新扫描窗口两次（`crates/ta-core/src/price_transform/avgdev.rs:27-35`）。
- 官方 VAR 是总体方差：`sum(x²)/N - (sum(x)/N)²`，不是以 `N-1` 为分母的样本方差。
- 官方 VAR API 包含 `nbdev` 参数，但计算忽略该参数。STDDEV 使用相同 variance，再对大于官方零阈值的结果计算 `sqrt(variance) * nbdev`。
- TA-Lib 的零/负阈值是 `1e-14`。STDDEV 的 variance 小于该阈值时输出 `0.0`；常量窗口不是错误。
- 官方 period 范围为：VAR `1..=100000`，STDDEV `2..=100000`。本地通用 `validate_period()` 只拒绝 0，因此函数特定边界不能完全由现有 helper 表达（`crates/ta-core/src/common.rs:87-104`）。
- 开发者选择 TA-Lib rolling moments，而不是 Welford、补偿或逐窗双遍。该选择保持 O(n)，但保留 `E[x²]-E[x]²` 在大基数近常量数据上的消减行为。
- `Float` 在默认构建中是 `f64`，`f32` feature 下是 `f32`（`crates/ta-core/src/types.rs:21-33`）。TA-Lib 的 single-input C 路径可用 double 累加，因此本项目 f32 parity 只能采用容差，不应承诺逐位一致。
- AVGDEV 的 streaming 实现按 ring 的物理顺序重算（`crates/ta-core/src/price_transform/avgdev.rs:131-139`）；数学上顺序不敏感，但浮点累加顺序会在 ring wrap 后与 batch 不同。统计 helper若要求本地 batch/stream 精确一致，需要共享更新路径或恢复逻辑顺序。

Authoritative semantics: [TA-Lib VAR](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/src/ta_func/ta_VAR.c), [TA-Lib STDDEV](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/src/ta_func/ta_STDDEV.c), [TA-Lib zero threshold](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/src/ta_func/ta_utility.h).

### CORREL and BETA

- `BinaryInput<'a>`/`BinaryTick` 已证明两个 borrowed slices 与两个 streaming scalar 能满足 `Indicator`/`StreamingIndicator` 的关联类型约束（`crates/ta-core/src/math_operators/arithmetic.rs:13-29`, `crates/ta-core/src/math_operators/arithmetic.rs:90-120`）。
- arithmetic kernel 的校验顺序是等长、finite、输出容量，然后计算（`crates/ta-core/src/math_operators/arithmetic.rs:35-48`）。该宏固定 lookback 0、每 tick 立即输出且没有状态，因此不能直接承载 rolling paired indicators。
- 开发者选择在 `statistic` facade 中定义共享 `PairInput`/`PairTick`，避免公共统计 API 依赖 `math_operators` 的类型所有权。
- CORREL 直接对 period 对原始观测维护 `sum_x`、`sum_y`、`sum_x2`、`sum_y2`、`sum_xy`；lookback 为 `period - 1`。
- CORREL 的 centered variance product 小于官方阈值时输出 `0.0`。任一侧为常量、period 1 或浮点消减导致的非正分母都不是 `TalibError`，也不是 warm-up NaN。
- BETA 不是原始价格水平的 covariance/variance。它先计算相邻价格的简单收益率 `(current - previous) / previous`，再计算 real1 return 对 real0 return 的回归斜率。
- BETA 的分母是 real0 returns 的 variance，因此参数顺序有语义且函数不对称。互换输入通常产生不同结果。
- 前一价格绝对接近零时，官方 BETA 将该 return 记为 `0.0`，而不是除零或报错。return variance 分母退化时输出 `0.0`。
- period 个 returns 需要 period+1 个原始价格，故 BETA lookback 为 `period`；CORREL 的相同 period lookback 为 `period - 1`。
- 两者可以共享 paired moments 的内部形状，但 BETA 还需要 previous raw pair，并把 transformed returns 而非原始 tick 送入 moments。

Authoritative semantics: [TA-Lib BETA](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/ta_codegen/input/beta/beta.c), [TA-Lib CORREL](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/ta_codegen/input/correl/correl.c).

### Linear Regression Family and TSF

- LINEARREG、LINEARREG_SLOPE、LINEARREG_INTERCEPT、LINEARREG_ANGLE 和 TSF 使用固定窗口横坐标 `x=0..N-1`，共享 `sum_x`、`sum_x2`、`sum_y`、`sum_xy`、slope 和 intercept。
- 对 period `N`：`sum_x=N(N-1)/2`，`sum_x2=N(N-1)(2N-1)/6`；这些常量只依赖 period。`N>=2` 时 x 方差分母固定非零，与 y 是否常量无关。
- 五个函数的 period 范围均为 `2..=100000`，lookback 均为 `N-1`。常量 y 窗口仍是合法拟合，不产生数学退化错误。
- LINEARREG 输出窗口末端 `b + m*(N-1)`；LINEARREG_SLOPE 输出 `m`；LINEARREG_INTERCEPT 输出 `b`。
- LINEARREG_ANGLE 输出 `atan(m) * 180/pi`，单位是度。本地 ATAN 使用 `Float::atan()` 并输出弧度，因此 ANGLE 只能对齐其浮点 atan 语义，不能直接复用 ATAN 的公开 batch 输出（`crates/ta-core/src/math_transform/mod.rs:20-40`, `crates/ta-core/src/math_transform/mod.rs:106-110`）。
- TSF 输出 `b + m*N`，即预测下一坐标。其结果仍对齐到产生预测的当前窗口末端，因此 TSF 的 lookback 与 LINEARREG 相同，不额外右移。
- 对窗口 `[1,2,3]`、period 3，五个结果分别是 LINEARREG `3`、SLOPE `1`、INTERCEPT `1`、ANGLE `45` 度、TSF `4`。
- 回归窗口对顺序敏感。WMA streaming 已展示从 ring 当前 index 恢复最旧到最新逻辑顺序的先例（`crates/ta-core/src/overlap/wma.rs:139-163`）。
- 当前 TA-Lib main 已将回归 batch 路径改为 O(1) recurrence，并显式使用 FMA；与 release 或普通 `m*x+b` 的低位输出可能不同，但不改变公开数学语义。

Authoritative semantics: [TA-Lib LINEARREG](https://ta-lib.org/functions/linearreg), [TA-Lib SLOPE](https://ta-lib.org/functions/linearreg_slope), [TA-Lib INTERCEPT](https://ta-lib.org/functions/linearreg_intercept), [TA-Lib ANGLE](https://ta-lib.org/functions/linearreg_angle), [TA-Lib TSF](https://ta-lib.org/functions/tsf).

### Batch, Streaming, and Reset Surfaces

- `Indicator::compute()` 使用 `&self`，现有 stateful struct 的 batch 计算只读取配置并调用纯 batch kernel；之前的 streaming 历史不会影响 batch 结果（`crates/ta-core/src/price_transform/avgdev.rs:92-111`）。
- `StreamingIndicator::next()` 使用 `None` 表示 warm-up、`Some` 表示有效输出、`Err` 表示无效 tick 或计算失败（`crates/ta-core/src/traits.rs:56-69`）。
- 标准 lookback `period-1` 意味着前 `period-1` 个 tick 返回 `None`，第 period 个 tick 产生首个输出。BETA 前 period 个原始 tick 返回 `None`，第 period+1 个 tick 产生首个输出。
- 现有 rolling indicators 在修改 state 前校验 tick finite，非法 tick 不推进窗口（`crates/ta-core/src/price_transform/avgdev.rs:114-129`）。
- inventory 测试手工断言每个当前实现 struct 同时实现 `Indicator` 和 `StreamingIndicator`（`crates/ta-core/tests/inventory.rs:132-235`）。新增类型不会被 inventory 自动发现，必须进入手工类型清单。
- `Resettable` 是独立 trait，inventory 当前没有编译期断言（`crates/ta-core/src/traits.rs:71-75`）。不过所有有状态 rolling 类型都按惯例重置 buffer、index、count 和累计量，保留 period 配置（`crates/ta-core/src/price_transform/avgdev.rs:142-149`）。
- 五个回归 struct、VAR、STDDEV、CORREL 和 BETA 均有 streaming warm-up 状态；现有项目惯例支持它们提供 reset，即使 inventory 不强制该 trait。

### Validation and Degenerate Mathematics

- `TalibError` 已覆盖 `InvalidInput`、`InvalidPeriod`、`InsufficientData`、`InvalidParameter` 和 `ComputationError`（`crates/ta-core/src/error.rs:12-50`）。没有统计语义要求新增 error variant。
- `InvalidInput` 已用于非有限输入、paired 长度不一致和 compact 输出缓冲区不足（`crates/ta-core/src/common.rs:116-177`）。
- `InsufficientData` 适用于合法 period、非空输入但不足一个窗口；普通函数要求至少 period 个点，BETA 要求至少 period+1 个原始点（`crates/ta-core/src/common.rs:106-114`）。
- 开发者选择官方边界：VAR/CORREL/BETA 最小 period 1；STDDEV 与五个回归函数最小 period 2；所有九个函数最大 period 100000。
- 空输入沿用本地约定：period 合法、paired slices 同为空、输出为空时成功返回 `OutputRange::empty()`；一个 paired input 为空而另一个非空时为 `InvalidInput`。
- period 1 下，VAR 每点输出 0；CORREL 每点输出 0；BETA 需要两个原始价格才能形成一个 return。STDDEV 和回归族拒绝 period 1。
- 常量窗口下，VAR/STDDEV/CORREL/BETA 输出 0；LINEARREG/INTERCEPT/TSF 输出常量；SLOPE/ANGLE 输出 0。它们都不是 `ComputationError`。
- 现有 core 只保证输入 finite，没有统一要求结果 finite；DIV、EXP、LN、SQRT 等允许 IEEE 数学结果通过。统计模块单独把中间 overflow 改成 `ComputationError` 会形成新的跨模块政策。
- `ComputationError` 当前几乎没有被算法路径使用；正常零分母或固定 x 回归不应成为其首个常规用途。

### Inventory and Public Discoverability

- 九项记录当前位于 `StatisticFunctions` 分组且全部为 `Planned`（`crates/ta-core/src/inventory.rs:283-292`）。
- `IMPLEMENTED_FUNCTION_COUNT` 当前为 45；九项全部实现后对应计数为 54，总官方函数数仍为 161（`crates/ta-core/src/inventory.rs:8-12`）。
- `function()` 只是对静态 inventory 做大写名称精确匹配，状态迁移不需要修改查询逻辑（`crates/ta-core/src/inventory.rs:324-326`）。
- inventory 测试同时比较手工 implemented 名称数组长度、动态状态计数和每个名称的状态（`crates/ta-core/tests/inventory.rs:68-130`）。
- `deferred_functions_remain_planned()` 当前仍把 VAR 作为 Planned 代表（`crates/ta-core/tests/inventory.rs:239-244`）。
- 全仓搜索确认，inventory 之外当前只有该 deferred 测试引用九个名称；ta-py、ta-wasm 和 benchmark 尚无统计 API 接线。

### Test Surface

- 现有 AVGDEV 测试覆盖普通数值、compact `OutputRange`、padded NaN、trait compute、坏长度、NaN、数据不足和零周期（`crates/ta-core/tests/price_transform.rs:54-96`）。
- 当前错误测试多数只断言 `.is_err()`，没有锁定具体 variant；也没有系统覆盖空输入、Infinity、输出缓冲区不足、streaming/reset 或 batch/stream 对齐。
- VAR/STDDEV 的区分向量必须能识别总体方差与样本方差；`[1,2,3]`、period 3 的官方 VAR 为 `2/3`。
- BETA 的 golden 必须锁定参数方向和收益率转换；CORREL golden 需覆盖正相关、负相关和常量侧。
- 回归族可用 `[1,2,3]`、period 3 同时锁定 endpoint、slope、intercept、degrees 和 next-point forecast。
- 精确断言适用于 `OutputRange`、NaN padding、常量零结果、warm-up count、reset 和 error variant。外部浮点 oracle 需要绝对+相对容差，并为 f32 使用更宽容差。
- 选择的 TA-Lib rolling 公式需要普通尺度 golden 和大基数近常量压力向量分开表达；后者记录选定公式的消减行为，而不是改用理论稳定值替代 parity。

### Criterion Benchmarks

- `series_fixture()` 提供确定性、有限、非恒定单序列；`paired_fixture()` 生成等长、相关但不相同的成对序列，分别适用于七个单输入函数和 CORREL/BETA（`crates/ta-benchmarks/benches/first_tranche.rs:27-41`）。
- compact benchmark 把 fixture 和 caller-owned output 分配置于 `b.iter()` 外，但仍测量 public kernel 的 period、finite、长度和容量校验（`crates/ta-benchmarks/benches/first_tranche.rs:72-89`）。
- vec benchmark 在 `b.iter()` 内分配返回 vector，因此包含 kernel、临时 compact buffer、padded vector 初始化与 copy 成本（`crates/ta-benchmarks/benches/first_tranche.rs:91-100`, `crates/ta-core/src/common.rs:179-210`）。
- 开发者选择九个函数全部覆盖 compact 与 vec API，而不是只选择家族代表。
- 当前 `SIZES` 为 1024、16384、65536，`PERIOD` 固定为 20（`crates/ta-benchmarks/benches/first_tranche.rs:22-23`）。固定 period 下，O(n) 和 O(n×period) 都随输入长度近似线性，无法从 size 曲线证明 rolling helper 避免重复窗口扫描。
- `first_tranche` bench target 已在 Cargo 中声明且 `harness=false`（`crates/ta-benchmarks/Cargo.toml:17-19`）。新增 benchmark 函数必须显式进入 `criterion_group!`；Cargo 不会自动注册文件内函数（`crates/ta-benchmarks/benches/first_tranche.rs:512-522`）。
- 历史 benchmark 显示 rolling 算法优化有效，而通用 SIMD 尝试曾回退；统计实现的 scalar rolling baseline 是性能判断的必要对照。

## Code References

- `crates/ta-core/src/common.rs:16-58` — Compact `OutputRange` 坐标和 empty semantics。
- `crates/ta-core/src/common.rs:61-85` — Float/i32 padding 与 output count。
- `crates/ta-core/src/common.rs:87-126` — Period、输入数量和输出容量校验。
- `crates/ta-core/src/common.rs:128-177` — Paired 等长与 finite 校验。
- `crates/ta-core/src/common.rs:179-210` — Compact buffer 与 padded vector 转换。
- `crates/ta-core/src/traits.rs:28-75` — Batch、streaming 和 reset traits。
- `crates/ta-core/src/error.rs:12-50` — 可用错误 variants。
- `crates/ta-core/src/types.rs:21-33` — `Float` f32/f64 feature boundary。
- `crates/ta-core/src/lib.rs:32-59` — Crate-root modules 和 common helper facade。
- `crates/ta-core/src/price_transform/avgdev.rs:16-50` — 单输入 period kernel 与 vec wrapper 模板。
- `crates/ta-core/src/price_transform/avgdev.rs:55-149` — Stateful struct、Indicator、streaming 和 reset 模板。
- `crates/ta-core/src/math_operators/rolling.rs:14-24` — Rolling kernel 公共验证形状。
- `crates/ta-core/src/math_operators/rolling.rs:81-97` — O(n) SUM batch recurrence。
- `crates/ta-core/src/math_operators/arithmetic.rs:13-29` — 双输入 slice/tick 类型先例。
- `crates/ta-core/src/math_operators/arithmetic.rs:31-120` — 双输入校验、wrapper 和 trait wiring。
- `crates/ta-core/src/overlap/wma.rs:139-163` — 顺序敏感 ring-buffer 遍历。
- `crates/ta-core/src/math_transform/mod.rs:20-40` — Float transform batch behavior。
- `crates/ta-core/src/math_transform/mod.rs:106-110` — ATAN radians semantics。
- `crates/ta-core/src/price_transform/mod.rs:6-16` — 私有叶子模块与公开 facade pattern。
- `crates/ta-core/src/inventory.rs:81-99` — Statistic group 到 `statistic` module string 的映射。
- `crates/ta-core/src/inventory.rs:283-292` — 九个 Planned records。
- `crates/ta-core/src/inventory.rs:324-326` — Inventory name lookup。
- `crates/ta-core/tests/price_transform.rs:54-96` — AVGDEV 数值/API/error 测试先例。
- `crates/ta-core/tests/inventory.rs:68-130` — Implemented count/status 三方断言。
- `crates/ta-core/tests/inventory.rs:132-235` — Indicator/StreamingIndicator 编译期类型清单。
- `crates/ta-core/tests/inventory.rs:239-244` — VAR deferred assertion。
- `crates/ta-benchmarks/benches/first_tranche.rs:22-41` — Size、period 和 fixtures。
- `crates/ta-benchmarks/benches/first_tranche.rs:72-103` — Compact/vec Criterion pattern。
- `crates/ta-benchmarks/benches/first_tranche.rs:512-522` — Criterion registration/entrypoint。
- `crates/ta-benchmarks/Cargo.toml:17-19` — `first_tranche` bench target。

## Integration Points

### Inbound References

- `crates/ta-core/src/inventory.rs:283-292` — 当前唯一完整登记九个函数名称的运行时代码 surface；状态均为 Planned。
- `crates/ta-core/tests/inventory.rs:239-244` — 当前唯一 inventory 之外的名称引用，只引用 VAR 并要求其保持 Planned。
- `crates/ta-core/tests/inventory.rs:132-235` — 公共 indicator 类型的手工编译期消费者；不会自动发现新统计类型。
- `crates/ta-benchmarks/benches/first_tranche.rs:10-19` — benchmark 当前导入多个 core groups，但没有 `statistic`。
- 全仓检索未发现 ta-py、ta-wasm 或其他 core 模块调用这些九个符号。

### Outbound Dependencies

- `crates/ta-core/src/common.rs:81-210` — 所有统计 kernel 依赖 lookback/count、错误校验、compact buffer 和 padding 协议。
- `crates/ta-core/src/traits.rs:28-75` — 同名 structs 依赖 batch、streaming 和 reset contracts。
- `crates/ta-core/src/error.rs:12-50` — 参数、长度、finite 和数据不足的 typed errors。
- `crates/ta-core/src/types.rs:21-33` — 所有中间状态和公开输出受 `Float` precision feature 约束。
- `crates/ta-core/src/math_transform/mod.rs:106-110` — ANGLE 与现有 Float atan 语义对齐，但不依赖其 batch API。

### Infrastructure Wiring

- `crates/ta-core/src/lib.rs:32-50` — 官方 group 的 crate-root module declarations；当前缺少 statistic。
- `crates/ta-core/src/price_transform/mod.rs:6-16` — group facade 的私有实现/公开重导出 wiring 先例。
- `crates/ta-core/src/inventory.rs:8-12` — 161 总数与 45 implemented count。
- `crates/ta-core/tests/inventory.rs:68-130` — implemented 名称、状态和 count 同步点。
- `crates/ta-core/tests/inventory.rs:132-235` — 九个 structs 的 trait discoverability 同步点。
- `crates/ta-benchmarks/benches/first_tranche.rs:512-522` — 新 Statistic benchmark group 的显式注册点。
- `crates/ta-benchmarks/Cargo.toml:17-19` — 现有 bench binary 已能承载新增 group，无需新 target。

## Architecture Insights

- 统计模块可以完全沿用 compact kernel + padded wrapper + struct + streaming/reset 的既有 contract；无需修改核心 traits。
- `statistic` 是官方 group facade 的正确公共边界。rolling moments 和 regression fit 是 group 内部算法组件，不是新的 crate-root public utilities。
- 开发者选择 statistic-owned `PairInput`/`PairTick`。这保留双输入 API 复用，同时避免 `statistic -> math_operators` 的公共类型耦合。
- VAR/STDDEV 可以共享一个 raw variance state；CORREL/BETA 可以共享 paired moments 组件；五个 regression exports 可以共享 fit state和投影逻辑。共享发生在实现层，不意味着跨独立 public API 调用缓存结果。
- BETA 的 return transformation 和额外 previous tick 使其不能与 CORREL 共享完全相同的 public lookback/state transition。
- TA-Lib parity 在本研究中表示参数、公式、阈值、退化输出和对齐语义一致；f32 与算法优化导致的低位差异使用容差管理，而非 bit-exact 承诺。
- 数值退化与 API 错误是不同层次：零 variance/denominator 是合法统计结果，非有限输入、长度不匹配、非法 period 和不足数据才是 typed errors。
- 所有九个 compact 与 vec benchmark 都属于 public API 性能；内部 helper 性能只能通过代码复杂度审查或增加 period 维度间接观察。

## Precedents & Lessons

5 类相近历史变化已分析。

### Precedent: First-tranche core contracts and APIs

**Commit(s)**: `3754ae9` — "Add TA-Lib first-tranche core APIs" (2026-07-05); `29f9886` — "Add first-tranche benchmarks" (2026-07-05)

**Blast radius**: 24 files across 5 layers

- core API/ — 建立 common helpers、traits 和 root facade。
- indicators/ — 加入 AVGDEV、binary/rolling operators 与 transforms。
- inventory/tests/benchmarks — 同步 ledger、公共 surface 和 Criterion target。

**Follow-up fixes**:

- 未发现 fix-labelled follow-up；`f5c9ed1` 后续扩展 moving averages 并保留 rolling kernel 优化。

**Lessons from docs**:

- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — Statistic scope 是九项，BETA/CORREL 为 paired inputs，VAR 不是 moving average。
- `.rpiv/artifacts/benchmarks/2026-07-05_simd-attempt-comparison.md` — 通用 SIMD 优化发生回退，保留优化前必须 benchmark。

**Takeaway**: 统计组应复制已稳定的双 API/inventory/test/benchmark wiring，并补足 first tranche 当时未提供的 parity vectors。

### Precedent: Period-based moving-average family

**Commit(s)**: `f5c9ed1` — "Add TA-Lib moving-average core APIs" (2026-07-09)

**Blast radius**: 27 files across 5 layers

- core facade/ — 扩展公开类型而未重写输出协议。
- indicators/ — 加入七个 period-based APIs 与 streaming state。
- inventory/tests/benchmarks — 同步状态、对齐、reset、f32 和 Criterion。

**Follow-up fixes**:

- 未发现纠正该 tranche 的后续提交。

**Lessons from docs**:

- `.rpiv/artifacts/validation/2026-07-07_21-50-54_ema-wma-trima-dema-tema-t3-ma.md` — composite lookback 采用 checked arithmetic，benchmark buffer 保持在 timed loop 外。
- `.rpiv/artifacts/benchmarks/2026-07-05_after-algo-opt-comparison.md` — sliding-window/monotonic-deque 优化比通用 SIMD 更符合该代码库。

**Takeaway**: Period/lookback 边界、streaming/reset 和线性时间 rolling kernel 是同一交付面的组成部分。

### Precedent: Complete Volatility and Volume groups

**Commit(s)**: `b2e1e0b` — "Add TA-Lib volatility and volume core APIs" (2026-07-13); `7dd357d` — "Add volatility and volume benchmarks" (2026-07-13)

**Blast radius**: 14 files across 5 layers

- core facade/ — 新增两个 official group modules。
- indicators/ — 加入 group-local helpers 与 multi-input indicators。
- inventory/tests/benchmarks — 同步完整 group 状态和公共 consumers。

**Follow-up fixes**:

- 截至当前提交未发现 follow-up fixes。

**Lessons from docs**:

- `.rpiv/artifacts/plans/2026-07-13_15-53-08_volume-indicators.md` — 复用 common validators、named SoA inputs，并显式处理 warm-up/denominator decisions。
- `.rpiv/artifacts/validation/2026-07-13_18-22-08_实现volume分组指标.md` — group 增量不需要修改 traits、adapters、manifests 或 SIMD backends。

**Takeaway**: 自包含 `statistic` facade 比重开稳定 core contracts 更符合最近的 group-level precedent。

### Precedent: Indicator trait and SMA churn

**Commit(s)**: `3683ed3` — "任务 1.3: 实现核心 Traits" (2026-01-29); `491d9e8` — "任务 1.3: 实现核心 Traits (更新版-强制SIMD加速+零拷贝)" (2026-01-29)

**Blast radius**: 85 files across 5 layers

- core API/ — traits 与 root facade 多次变化。
- overlap/benchmarks/ — SMA、streaming、NaN 和 benchmark surface 被反复重写。
- repository hygiene/ — 早期提交混入生成文件。

**Follow-up fixes**:

- `bd5b5a3`, `6528112`, `f833937`, `3267f43`, `896d1d7` — SMA/trait/streaming surface 在 2026-02-02 至 2026-02-05 间多次删除、重加或改写。

**Lessons from docs**:

- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — 既有 churn 集中于 compute、next、warm-up、NaN 和 Result semantics。

**Takeaway**: `Indicator` 的 GAT input 已能表达 paired series；Statistic Functions 不构成重写 trait 的理由。

### Precedent: Error system and std/no_std boundary

**Commit(s)**: `2335e94` — "任务 1.2: 实现错误类型系统" (2026-01-29)

**Blast radius**: 694 files across 3 layers，其中绝大多数是误提交的生成文件。

**Follow-up fixes**:

- `ee346da` — "remove target from git" (2026-01-29) — 删除 tracked build artifacts。
- `29fb091` — "update" (2026-02-03) — 调整 error imports 和 std/no_std boundary。

**Lessons from docs**:

- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — 现有 error variants 已覆盖统计模块需要的公开失败语义。

**Takeaway**: 复用现有错误并保持 no_std/Float 边界；不要为了合法数学退化扩展 `TalibError`。

### Composite Lessons

- 最近成功的 group 增量都保持 common/traits/error 稳定，只新增 group facade、实现、测试、inventory 和 benchmark wiring。
- Compact range、padded NaN、streaming warm-up 和 reset 必须作为一个一致 contract 验证，不能只测试 batch 数值。
- Rolling scalar baseline 是当前项目最可靠的性能路径；`f5c9ed1` 和相关 benchmark artifact 支持先做 O(n) 算法，再考虑 SIMD。
- Inventory count、九条状态、implemented 名称数组、deferred 列表和 trait assertion 清单是五个独立但同步的 discoverability surface。
- TA-Lib parity vectors 是统计组比 first tranche 更需要补强的证据，因为分母、收益率方向、阈值和预测坐标都无法仅靠类型系统表达。

## Historical Context (from `.rpiv/artifacts/`)

- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — TA-Lib core inventory research。
- `.rpiv/artifacts/designs/2026-07-04_17-28-24_rust-talib-core-foundation-first-tranche.md` — Foundation first-tranche design。
- `.rpiv/artifacts/plans/2026-07-05_09-51-36_rust-talib-core-foundation-first-tranche.md` — Foundation first-tranche implementation plan。
- `.rpiv/artifacts/validation/2026-07-05_11-35-24_ta-lib-rust-core-foundation-first-tranche.md` — Foundation validation。
- `.rpiv/artifacts/plans/2026-07-05_21-38-45_ema-wma-trima-dema-tema-t3-ma.md` — Moving-average family plan。
- `.rpiv/artifacts/validation/2026-07-07_21-50-54_ema-wma-trima-dema-tema-t3-ma.md` — Moving-average family validation。
- `.rpiv/artifacts/benchmarks/2026-07-05_after-algo-opt-comparison.md` — Rolling algorithm benchmark comparison。
- `.rpiv/artifacts/benchmarks/2026-07-05_simd-attempt-comparison.md` — SIMD attempt comparison。
- `.rpiv/artifacts/plans/2026-07-09_21-49-41_volatility-indicators.md` — Volatility group plan。
- `.rpiv/artifacts/validation/2026-07-12_14-15-57_实现volatility分组指标.md` — Volatility group validation。
- `.rpiv/artifacts/plans/2026-07-13_15-53-08_volume-indicators.md` — Volume group plan。
- `.rpiv/artifacts/validation/2026-07-13_18-22-08_实现volume分组指标.md` — Volume group validation。

## Developer Context

**Q (`crates/ta-core/src/math_operators/rolling.rs:81-97`, `crates/ta-core/src/price_transform/avgdev.rs:27-35`): 统计 helper 应优先 TA-Lib parity、稳定 rolling，还是逐窗稳定双遍？**

A: 选择 TA-Lib rolling。公开语义对齐 TA-Lib，内部保持 O(n)，external golden 使用容差；不使用 Welford fallback 或 O(n×period) 双遍作为主路径。

**Q (`crates/ta-core/src/common.rs:88-104`): 通用 helper 只拒绝 period=0，新统计模块是否采用官方函数特定边界？**

A: 采用官方边界。VAR/CORREL/BETA 为 `1..=100000`；STDDEV 和五个回归函数为 `2..=100000`。

**Q (`crates/ta-core/src/math_operators/arithmetic.rs:15-29`, `crates/ta-core/src/price_transform/mod.rs:12-16`): BETA/CORREL 的公共 paired input/tick 类型归属在哪里？**

A: 在 `statistic` facade 暴露共享 Pair 类型，避免依赖 `math_operators::BinaryInput`/`BinaryTick`，也不为两个函数重复定义类型。

**Q (`crates/ta-benchmarks/benches/first_tranche.rs:3-6`, `crates/ta-benchmarks/benches/first_tranche.rs:72-103`): 九个统计函数的 Criterion 覆盖 compact、vec 或家族代表中的哪一层？**

A: 九个函数的 compact 和 vec API 全部纳入 benchmark。

## Related Research

- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md`
- `.rpiv/artifacts/research/2026-07-09_21-23-34_volatility-indicators.md`

## Open Questions

- Golden oracle 应固定到最新稳定 release `v0.7.1`，还是本研究使用且包含 O(1) regression/FMA 变更的 TA-Lib commit `1bec05cf72fa790e2e3ecca40e6607de15fe0a30`？公开语义相同，但长序列低位结果可能不同。
- f64/f32 的绝对+相对容差尚未确定；现有 tests 多用固定绝对容差，无法覆盖大数量级统计输出。
- Criterion 的 period 矩阵尚未确定。当前固定 period 20 可以测吞吐量，但不能验证算法成本不随 period 线性增长。
