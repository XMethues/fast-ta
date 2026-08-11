import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";

const require = createRequire(import.meta.url);
const [scalarPath, simdPath, outputPath] = process.argv.slice(2);
if (!scalarPath || !simdPath || !outputPath) {
  throw new Error("usage: node typprice_runtime.mjs SCALAR_BINDING SIMD_BINDING OUTPUT_JSONL");
}

const scalarModule = require(path.resolve(scalarPath));
const simdModule = require(path.resolve(simdPath));
const lengths = [256, 4096, 65536];
const sampleCount = 31;
const bootstrapResamples = 2000;

function fixture(size) {
  const close = new Float64Array(size);
  const open = new Float64Array(size);
  const high = new Float64Array(size);
  const low = new Float64Array(size);
  for (let index = 0; index < size; index += 1) {
    close[index] = index * 0.001 + ((index * 37) % 101) + 1.0;
    open[index] = close[index] + ((index % 9) - 4.0) * 0.035;
    high[index] = Math.max(open[index], close[index]) + 0.5 + (index % 11) * 0.03;
    low[index] = Math.min(open[index], close[index]) - 0.5 - (index % 7) * 0.025;
  }
  return { high, low, close };
}

function invoke(module, inputs) {
  const result = module.typprice(inputs.high, inputs.low, inputs.close);
  try {
    return {
      outputBegin: result.output_begin,
      outputCount: result.output_count,
      values: result.values,
    };
  } finally {
    result.free();
  }
}

function assertExact(actual, expected, context) {
  if (actual.length !== expected.length) {
    throw new Error(`${context}: output length ${actual.length} != ${expected.length}`);
  }
  for (let index = 0; index < actual.length; index += 1) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(`${context}: output[${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

function scalarReference(inputs) {
  const output = new Float64Array(inputs.high.length);
  for (let index = 0; index < output.length; index += 1) {
    output[index] = (inputs.high[index] + inputs.low[index] + inputs.close[index]) / 3.0;
  }
  return output;
}

function iterations(size) {
  if (size === 256) return 500;
  if (size === 4096) return 100;
  return 8;
}

function median(values) {
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor(ordered.length / 2)];
}

function confidenceInterval(samples) {
  let state = 0x243f6a88 >>> 0;
  const medians = [];
  for (let resampleIndex = 0; resampleIndex < bootstrapResamples; resampleIndex += 1) {
    const resample = [];
    for (let index = 0; index < samples.length; index += 1) {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      resample.push(samples[state % samples.length]);
    }
    medians.push(median(resample));
  }
  medians.sort((left, right) => left - right);
  return [
    medians[Math.floor(bootstrapResamples * 0.025)],
    medians[Math.floor(bootstrapResamples * 0.975)],
  ];
}

function checksum(values) {
  let hash = 0xcbf29ce484222325n;
  const hashBytes = (bytes) => {
    for (const byte of bytes) {
      hash ^= BigInt(byte);
      hash = BigInt.asUintN(64, hash * 0x100000001b3n);
    }
  };
  hashBytes(new TextEncoder().encode("float"));
  const shape = new ArrayBuffer(16);
  const shapeView = new DataView(shape);
  shapeView.setBigUint64(0, 0n, true);
  shapeView.setBigUint64(8, BigInt(values.length), true);
  hashBytes(new Uint8Array(shape));
  hashBytes(new Uint8Array(values.buffer, values.byteOffset, values.byteLength));
  return `fnv1a64:${hash.toString(16).padStart(16, "0")}`;
}

function verifyErrors(module, label) {
  let mismatchMessage = "";
  try {
    module.typprice(new Float64Array([1, 2]), new Float64Array([1]), new Float64Array([1, 2]));
  } catch (error) {
    mismatchMessage = String(error.message ?? error);
  }
  if (!mismatchMessage.includes("must have the same length")) {
    throw new Error(`${label}: unequal lengths did not preserve validation semantics: ${mismatchMessage}`);
  }

  let finiteMessage = "";
  try {
    module.typprice(new Float64Array([1]), new Float64Array([Number.NaN]), new Float64Array([1]));
  } catch (error) {
    finiteMessage = String(error.message ?? error);
  }
  if (!finiteMessage.includes("low[0] must be finite")) {
    throw new Error(`${label}: non-finite input did not preserve validation semantics: ${finiteMessage}`);
  }
  return { mismatchMessage, finiteMessage };
}

function measure(module, inputs) {
  let sink = 0;
  for (let warmup = 0; warmup < 10; warmup += 1) {
    const output = invoke(module, inputs);
    sink += output.values[output.values.length - 1];
  }
  const count = iterations(inputs.high.length);
  const samples = [];
  for (let sample = 0; sample < sampleCount; sample += 1) {
    const started = performance.now();
    for (let iteration = 0; iteration < count; iteration += 1) {
      const output = invoke(module, inputs);
      sink += output.values[output.values.length - 1];
    }
    samples.push(((performance.now() - started) * 1_000_000) / count);
  }
  if (!Number.isFinite(sink)) throw new Error("benchmark output sink became non-finite");
  const [ci95LowerNs, ci95UpperNs] = confidenceInterval(samples);
  return { medianNs: median(samples), ci95LowerNs, ci95UpperNs };
}

const scalarBackend = scalarModule.typprice_backend();
const simdBackend = simdModule.typprice_backend();
if (scalarBackend !== "scalar") {
  throw new Error(`scalar module selected ${scalarBackend}`);
}
if (simdBackend !== "simd128") {
  throw new Error(`SIMD module selected ${simdBackend}`);
}

const scalarErrors = verifyErrors(scalarModule, "scalar");
const simdErrors = verifyErrors(simdModule, "simd128");
const records = [
  {
    record: "metadata",
    indicator: "TYPPRICE",
    indicator_definition: "TYPPRICE: Typical Price",
    parameters: "none",
    fixture: "catalogue_fixture_v1:f64le",
    platform: "wasm32-unknown-unknown",
    runtime: process.version,
    cpu: os.cpus()[0]?.model ?? "unknown",
    os: `${os.platform()} ${os.release()} ${os.arch()}`,
    commit: process.env.QUALIFICATION_COMMIT ?? "unknown",
    workflow_run_id: process.env.GITHUB_RUN_ID ?? "unknown",
    workflow_run_url:
      process.env.GITHUB_RUN_ID &&
      process.env.GITHUB_SERVER_URL &&
      process.env.GITHUB_REPOSITORY
        ? `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`
        : "unknown",
    workflow_job: process.env.GITHUB_JOB ?? "unknown",
    scalar_feature_flags: "-C target-feature=-simd128",
    simd_feature_flags: "-C target-feature=+simd128",
    scalar_backend: scalarBackend,
    simd_backend: simdBackend,
    sample_count: sampleCount,
  },
  {
    record: "validation",
    indicator: "TYPPRICE",
    unequal_lengths_verified: true,
    non_finite_verified: true,
    scalar_unequal_lengths_error: scalarErrors.mismatchMessage,
    scalar_non_finite_error: scalarErrors.finiteMessage,
    simd_unequal_lengths_error: simdErrors.mismatchMessage,
    simd_non_finite_error: simdErrors.finiteMessage,
  },
];

for (const inputLength of lengths) {
  const inputs = fixture(inputLength);
  const reference = scalarReference(inputs);
  const scalarOutput = invoke(scalarModule, inputs);
  const simdOutput = invoke(simdModule, inputs);
  if (scalarOutput.outputBegin !== 0 || scalarOutput.outputCount !== inputLength) {
    throw new Error(`scalar: invalid Output Range at ${inputLength}`);
  }
  if (simdOutput.outputBegin !== 0 || simdOutput.outputCount !== inputLength) {
    throw new Error(`simd128: invalid Output Range at ${inputLength}`);
  }
  assertExact(scalarOutput.values, reference, `scalar length ${inputLength}`);
  assertExact(simdOutput.values, scalarOutput.values, `simd128 length ${inputLength}`);
  const outputChecksum = checksum(scalarOutput.values);

  for (const [backend, module] of [["scalar", scalarModule], ["simd128", simdModule]]) {
    const timing = measure(module, inputs);
    records.push({
      record: "measurement",
      indicator: "TYPPRICE",
      indicator_family: "Price Transform",
      indicator_definition: "TYPPRICE: Typical Price",
      case_id: "TYPPRICE",
      mode: "public WASM typprice",
      backend,
      parameters: "none",
      input_length: inputLength,
      output_kind: "float",
      output_arity: 1,
      output_begin: 0,
      output_count: inputLength,
      output_checksum: outputChecksum,
      equivalent_to_scalar: true,
      semantic_status: "verified",
      timing_status: "measured",
      sample_count: sampleCount,
      median_ns: timing.medianNs,
      ci95_lower_ns: timing.ci95LowerNs,
      ci95_upper_ns: timing.ci95UpperNs,
      throughput_observations_per_second: inputLength * 1_000_000_000 / timing.medianNs,
      fixture: "catalogue_fixture_v1:f64le",
      timed_boundary: "public wasm-bindgen typprice; input conversion, validation, allocation, result values copy included",
    });
  }
}

fs.mkdirSync(path.dirname(path.resolve(outputPath)), { recursive: true });
fs.writeFileSync(outputPath, `${records.map((record) => JSON.stringify(record)).join("\n")}\n`);
console.log(`WASM TYPPRICE qualification passed: ${outputPath}`);
