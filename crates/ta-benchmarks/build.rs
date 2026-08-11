fn main() {
    println!("cargo:rerun-if-env-changed=CATALOGUE_TALIB_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_CATALOGUE_MATRIX").is_none() {
        return;
    }
    let Ok(library_dir) = std::env::var("CATALOGUE_TALIB_LIB_DIR") else {
        println!(
            "cargo:warning=catalogue-matrix is compile-checked without native linkage; use the documented runner to execute it"
        );
        return;
    };
    println!("cargo:rustc-link-search=native={library_dir}");
    println!("cargo:rustc-link-lib=dylib=ta-lib");
}
