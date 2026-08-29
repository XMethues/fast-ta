#[cfg(feature = "catalogue-matrix")]
fn main() {
    let mut arguments = std::env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| "catalogue-evidence".into());
    let Some(path) = arguments.next() else {
        eprintln!(
            "usage: {} RAW_TSV",
            std::path::Path::new(&program).display()
        );
        std::process::exit(2);
    };
    if arguments.next().is_some() {
        eprintln!(
            "usage: {} RAW_TSV",
            std::path::Path::new(&program).display()
        );
        std::process::exit(2);
    }

    if let Err(error) =
        ta_benchmarks::catalogue_evidence::read_publishable_evidence(std::path::Path::new(&path))
    {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "catalogue-matrix"))]
fn main() {
    eprintln!("catalogue-evidence requires the catalogue-matrix feature");
    std::process::exit(2);
}
