fn main() {
    println!("cargo:rerun-if-env-changed=SMA_TALIB_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_SMA_THREE_WAY").is_none() {
        return;
    }
    let Ok(library_dir) = std::env::var("SMA_TALIB_LIB_DIR") else {
        println!(
            "cargo:warning=sma-three-way is compile-checked without native linkage; use the documented runner to execute it"
        );
        return;
    };
    println!("cargo:rustc-link-search=native={library_dir}");
    println!("cargo:rustc-link-lib=dylib=ta-lib");
}
