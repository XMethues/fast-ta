use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=catalogue-cases.tsv");
    println!("cargo:rerun-if-env-changed=CATALOGUE_TALIB_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_CATALOGUE_MATRIX").is_none() {
        return;
    }
    generate_catalogue_cases();

    let Ok(library_dir) = std::env::var("CATALOGUE_TALIB_LIB_DIR") else {
        println!(
            "cargo:warning=catalogue-matrix is compile-checked without native linkage; use the documented runner to execute it"
        );
        return;
    };
    println!("cargo:rustc-link-search=native={library_dir}");
    println!("cargo:rustc-link-lib=dylib=ta-lib");
}

fn generate_catalogue_cases() {
    let manifest = std::fs::read_to_string("catalogue-cases.tsv")
        .expect("read canonical catalogue case manifest");
    let mut rows = manifest.lines();
    assert_eq!(
        rows.next(),
        Some("kind\tid\tfamily\tdefinition\tparameters\toutput_kind\toutput_arity"),
        "unexpected catalogue case manifest header"
    );

    let mut cases = Vec::new();
    for (index, row) in rows.enumerate() {
        let fields = row.split('\t').collect::<Vec<_>>();
        assert_eq!(
            fields.len(),
            7,
            "catalogue-cases.tsv line {} must have seven tab-separated fields",
            index + 2
        );
        let output_arity = fields[6].parse::<usize>().unwrap_or_else(|_| {
            panic!(
                "invalid output arity on catalogue-cases.tsv line {}",
                index + 2
            )
        });
        cases.push((fields, output_arity));
    }
    assert!(
        !cases.is_empty(),
        "catalogue case manifest must not be empty"
    );

    let mut generated = String::new();
    writeln!(
        generated,
        "/// Canonical representative Indicator Catalogue measurement matrix."
    )
    .unwrap();
    writeln!(
        generated,
        "pub const MATRIX: [CaseSpec; {}] = [",
        cases.len()
    )
    .unwrap();
    for (fields, output_arity) in cases {
        writeln!(generated, "    CaseSpec {{").unwrap();
        writeln!(generated, "        kind: CaseKind::{},", fields[0]).unwrap();
        writeln!(generated, "        id: {:?},", fields[1]).unwrap();
        writeln!(generated, "        family: {:?},", fields[2]).unwrap();
        writeln!(generated, "        definition: {:?},", fields[3]).unwrap();
        writeln!(generated, "        parameters: {:?},", fields[4]).unwrap();
        writeln!(generated, "        output_kind: {:?},", fields[5]).unwrap();
        writeln!(generated, "        output_arity: {output_arity},").unwrap();
        writeln!(generated, "    }},").unwrap();
    }
    writeln!(generated, "];").unwrap();

    let output =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("catalogue_cases.rs");
    std::fs::write(output, generated).expect("write generated catalogue cases");
}
