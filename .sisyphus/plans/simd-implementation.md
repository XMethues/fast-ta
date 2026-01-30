# SIMD 手动调度实施计划

**状态**: 待实施  
**优先级**: P0 (最高)  
**预计工期**: 46 小时  
**创建日期**: 2026-01-30  

---

## 项目概述

基于分析结果，实施 **Option 3（函数指针 + wide库）** 的手动 SIMD 调度系统：

- ❌ 不使用 multiversion crate
- ❌ 不使用 SSE/SSE2/SSE3/SSE4.1
- ✅ 支持 AVX2 和 AVX-512
- ✅ 支持 ARM64 NEON 和 WASM SIMD128
- ✅ 单二进制文件，运行时调度
- ✅ 零运行时开销（启动后）

---

## 架构设计

### 核心组件

```
simd/
├── mod.rs              # 公共 API 和运行时检测
├── dispatch.rs         # 函数指针调度系统
├── types.rs            # SIMD 相关类型和常量
├── scalar.rs           # 标量回退实现
└── arch/
    ├── x86_64/
    │   ├── avx2.rs     # AVX2 实现（256-bit）
    │   └── avx512.rs   # AVX-512 实现（512-bit，使用 std::arch）
    ├── aarch64/
    │   └── neon.rs     # ARM64 NEON 实现（128-bit）
    └── wasm32/
        └── simd128.rs  # WASM SIMD128 实现（128-bit）
```

### 调度流程

```rust
// 1. 启动时检测 CPU 特性
static DISPATCH: OnceLock<ComputeFn> = OnceLock::new();

fn init() {
    DISPATCH.get_or_init(|| {
        if is_x86_feature_detected!("avx512f") {
            compute_avx512
        } else if is_x86_feature_detected!("avx2") {
            compute_avx2
        } else {
            compute_scalar
        }
    });
}

// 2. 运行时调用
pub fn compute(data: &[f64]) -> f64 {
    init();
    DISPATCH.get().unwrap()(data)
}
```

---

## 详细任务清单

### 阶段 1：基础框架（8 小时）

- [ ] **任务 1.1**: 创建 `simd/mod.rs`
  - [ ] 定义 `SimdLevel` 枚举（Scalar, Avx2, Avx512, Neon, Simd128）
  - [ ] 实现 `detect_simd_level()` 函数
  - [ ] 使用 `OnceLock` 缓存检测结果
  - [ ] 添加 x86_64 检测逻辑（AVX2, AVX-512）
  - [ ] 添加 ARM64 检测逻辑（NEON）
  - [ ] 添加 WASM 检测逻辑（SIMD128）
  - 验收标准：
    - [ ] 在 x86_64 Linux 上正确检测 AVX2/AVX-512
    - [ ] 在 ARM64 macOS 上正确检测 NEON
    - [ ] 单元测试通过

- [ ] **任务 1.2**: 创建 `simd/dispatch.rs`
  - [ ] 定义 `ComputeFn` 类型别名
  - [ ] 实现 `DispatchTable` 结构体
  - [ ] 实现全局 `DISPATCH_TABLE`（使用 `OnceLock`）
  - [ ] 实现 `get_compute_fn()` 函数
  - [ ] 添加架构特定的调度逻辑
  - 验收标准：
    - [ ] 首次调用时初始化调度表
    - [ ] 后续调用使用缓存的函数指针
    - [ ] 性能测试：调度开销 < 10ns

- [ ] **任务 1.3**: 创建 `simd/types.rs`
  - [ ] 定义 SIMD 相关的公共类型
  - [ ] 定义常量（如 `LANES_64`, `LANES_128`, `LANES_256`, `LANES_512`）
  - [ ] 添加文档说明各平台的 lane 数量
  - 验收标准：
    - [ ] 类型定义清晰
    - [ ] 文档完整

- [ ] **任务 1.4**: 创建 `simd/scalar.rs`
  - [ ] 实现标量回退版本的 `compute` 函数
  - [ ] 实现 `sum` 函数
  - [ ] 实现 `dot_product` 函数
  - [ ] 添加单元测试
  - 验收标准：
    - [ ] 所有函数有完整测试
    - [ ] 性能基准：作为 baseline

### 阶段 2：x86_64 实现（16 小时）

- [ ] **任务 2.1**: 创建 `simd/arch/x86_64/avx2.rs`
  - [ ] 实现 `compute` 函数（使用 `#[target_feature(enable = "avx2")]`）
  - [ ] 使用 `wide::f64x4` 进行 256-bit SIMD 计算
  - [ ] 实现滑动窗口优化（避免重复计算）
  - [ ] 处理余数元素
  - [ ] 添加单元测试
  - 验收标准：
    - [ ] 在支持 AVX2 的 CPU 上运行
    - [ ] 结果与标量实现一致（误差 < 1e-10）
    - [ ] 性能比标量快 2x 以上

- [ ] **任务 2.2**: 创建 `simd/arch/x86_64/avx512.rs`
  - [ ] 实现 `compute` 函数（使用 `#[target_feature(enable = "avx512f")]`）
  - [ ] 使用 `std::arch::x86_64::_mm512_*` intrinsics
  - [ ] 实现 512-bit SIMD 计算（8 lanes for f64）
  - [ ] 添加单元测试（标记为 `#[ignore]` 除非有 AVX-512 硬件）
  - 验收标准：
    - [ ] 代码编译通过（使用 `#[cfg(target_feature = "avx512f")]`）
    - [ ] 逻辑正确（可在 AVX-512 模拟器上测试）

- [ ] **任务 2.3**: 创建 x86_64 集成测试
  - [ ] 创建 `tests/x86_64_simd.rs`
  - [ ] 测试 AVX2 路径
  - [ ] 测试 AVX-512 路径（条件编译）
  - [ ] 测试标量回退
  - [ ] 性能基准测试
  - 验收标准：
    - [ ] 所有测试通过
    - [ ] 基准测试显示预期性能提升

### 阶段 3：ARM64 实现（8 小时）

- [ ] **任务 3.1**: 创建 `simd/arch/aarch64/neon.rs`
  - [ ] 实现 `compute` 函数（ARM64 NEON）
  - [ ] 使用 `wide::f64x4`（在 NEON 上映射到 2x 128-bit 寄存器）
  - [ ] 实现 NEON 特定的优化
  - [ ] 添加单元测试
  - 验收标准：
    - [ ] 在 ARM64 设备上运行（如 Apple Silicon Mac）
    - [ ] 结果与标量一致
    - [ ] 性能优于标量

- [ ] **任务 3.2**: 创建 ARM64 集成测试
  - [ ] 创建 `tests/aarch64_simd.rs`
  - [ ] 测试 NEON 路径
  - [ ] 条件编译确保只在 ARM64 运行
  - 验收标准：
    - [ ] 测试通过

### 阶段 4：WASM 实现（6 小时）

- [ ] **任务 4.1**: 创建 `simd/arch/wasm32/simd128.rs`
  - [ ] 实现 `compute` 函数（WASM SIMD128）
  - [ ] 使用 `wide::f64x4`（在 WASM 上映射到 v128）
  - [ ] 处理 WASM 特定的限制（128-bit 最大）
  - [ ] 添加单元测试
  - 验收标准：
    - [ ] 编译为 WASM 目标（`wasm32-unknown-unknown`）
    - [ ] 在浏览器或 Node.js 中运行
    - [ ] 结果正确

- [ ] **任务 4.2**: 创建 WASM 构建配置
  - [ ] 更新 `Cargo.toml` 添加 WASM 特性
  - [ ] 创建 `wasm-pack` 配置
  - [ ] 添加 WASM 示例
  - 验收标准：
    - [ ] `wasm-pack build` 成功
    - [ ] 示例在浏览器中运行

### 阶段 5：集成与测试（8 小时）

- [ ] **任务 5.1**: 集成到现有 SMA 实现
  - [ ] 修改 `overlap/sma.rs` 使用新的 SIMD 调度层
  - [ ] 保留向后兼容的 API
  - [ ] 添加配置选项（启用/禁用 SIMD）
  - 验收标准：
    - [ ] 现有测试通过
    - [ ] 新 SIMD 路径工作正常

- [ ] **任务 5.2**: 创建综合测试套件
  - [ ] 跨平台测试（Linux x86_64, macOS ARM64, WASM）
  - [ ] 性能回归测试
  - [ ] 精度验证（与标量实现对比）
  - 验收标准：
    - [ ] CI 通过所有测试
    - [ ] 性能提升达到预期（2x+ for AVX2）

- [ ] **任务 5.3**: 编写文档
  - [ ] API 文档（rustdoc）
  - [ ] 架构设计文档
  - [ ] 性能调优指南
  - [ ] 迁移指南（从旧实现迁移）
  - 验收标准：
    - [ ] 文档完整
    - [ ] 示例代码可运行

---

## 风险与缓解策略

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| AVX-512 不支持 | 高 | 中 | 使用 std::arch 直接实现，不依赖 wide |
| WASM SIMD128 限制 | 中 | 低 | 提前测试，必要时回退到标量 |
| 跨平台测试困难 | 中 | 中 | 使用 CI 和模拟器，重点测试 x86_64 和 ARM64 |
| 性能未达预期 | 低 | 高 | 早期基准测试，及时调整算法 |

---

## 验收标准总结

### 功能验收
- [ ] 在 x86_64 上自动选择 AVX2 或 AVX-512
- [ ] 在 ARM64 上使用 NEON
- [ ] 在 WASM 上使用 SIMD128（如果可用）
- [ ] 无 SIMD 时回退到标量
- [ ] 所有路径产生相同结果（误差 < 1e-10）

### 性能验收
- [ ] AVX2 比标量快 2x 以上
- [ ] AVX-512 比 AVX2 快 1.5x 以上（在支持硬件上）
- [ ] NEON 比标量快 2x 以上
- [ ] 调度开销 < 10ns

### 质量验收
- [ ] 代码覆盖率 > 90%
- [ ] 所有 CI 测试通过
- [ ] 文档完整
- [ ] 无内存安全警告

---

## 下一步行动

1. **审查本计划**：确认所有任务和验收标准
2. **分配资源**：确定负责每个任务的开发人员
3. **启动实施**：使用 `/start-work` 命令开始第一阶段
4. **定期检查**：每周审查进度，调整计划

---

**计划创建**: Prometheus (Plan Builder)  
**最后更新**: 2026-01-30  
**版本**: 1.0