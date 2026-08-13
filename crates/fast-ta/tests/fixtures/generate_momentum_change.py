#!/usr/bin/env python3
"""Generate independent MOM/ROC/ROCP/ROCR/ROCR100 reference literals.

Reference: TA-Lib revision e64d2ac896c595f38d65e44c812efbfdac8a64cf,
src/ta_func/ta_{MOM,ROC,ROCP,ROCR,ROCR100}.c. The generator deliberately
spells out the five definitions and is not coupled to the Rust implementation.
An exactly zero trailing value maps every normalized definition to zero; a
near-zero nonzero value is divided normally.
"""

PERIOD = 3
INPUT = [10.0, 11.0, 12.0, 15.0, 0.0, 20.0, 1e-12, -5.0, 25.0, 30.0, 5.0, 40.0]


def denominator_or_zero(current, trailing, formula):
    return 0.0 if trailing == 0.0 else formula(current, trailing)


def calculate(name, current, trailing):
    if name == "MOM":
        return current - trailing
    if name == "ROC":
        return denominator_or_zero(current, trailing, lambda c, t: (c / t - 1.0) * 100.0)
    if name == "ROCP":
        return denominator_or_zero(current, trailing, lambda c, t: (c - t) / t)
    if name == "ROCR":
        return denominator_or_zero(current, trailing, lambda c, t: c / t)
    if name == "ROCR100":
        return denominator_or_zero(current, trailing, lambda c, t: c / t * 100.0)
    raise ValueError(name)


for definition in ("MOM", "ROC", "ROCP", "ROCR", "ROCR100"):
    values = [
        calculate(definition, INPUT[index], INPUT[index - PERIOD])
        for index in range(PERIOD, len(INPUT))
    ]
    literals = ",\n    ".join(f"{value!r} as Float" for value in values)
    print(f"pub const {definition}_EXPECTED: &[Float] = &[\n    {literals},\n];\n")
