#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::path::PathBuf;

use ta_benchmarks::performance_qualification::{
    qualify_typprice, NativePlatform, Provenance, QualificationRequest,
};

fn value(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.to_owned())
}

fn required(name: &str) -> String {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("{name} must be set by the qualification adapter"))
}

fn provenance() -> Provenance {
    let run_id = value(
        "QUALIFICATION_WORKFLOW_RUN_ID",
        &value("GITHUB_RUN_ID", "local"),
    );
    let run_url = value(
        "QUALIFICATION_WORKFLOW_RUN_URL",
        &match (
            std::env::var("GITHUB_SERVER_URL"),
            std::env::var("GITHUB_REPOSITORY"),
        ) {
            (Ok(server), Ok(repository)) => format!("{server}/{repository}/actions/runs/{run_id}"),
            _ => "local://native-typprice-qualification".to_owned(),
        },
    );
    Provenance {
        runtime: required("QUALIFICATION_RUNTIME"),
        os: required("QUALIFICATION_OS"),
        architecture: required("QUALIFICATION_ARCHITECTURE"),
        cpu: required("QUALIFICATION_CPU"),
        cpu_features: value("QUALIFICATION_CPU_FEATURES", "runtime-detected"),
        rust_profile: value("QUALIFICATION_RUST_PROFILE", "release"),
        cargo_features: std::env::var("QUALIFICATION_CARGO_FEATURES")
            .or_else(|_| std::env::var("QUALIFICATION_FEATURES"))
            .unwrap_or_else(|_| "unknown".to_owned()),
        target_features: value("QUALIFICATION_TARGET_FEATURES", "runtime-detected"),
        commit: required("QUALIFICATION_COMMIT"),
        qualification_command: required("QUALIFICATION_COMMAND"),
        source_repository: value("GITHUB_REPOSITORY", "XMethues/fast-ta"),
        workflow_name: value("GITHUB_WORKFLOW", "Platform qualification"),
        workflow_ref: value("GITHUB_WORKFLOW_REF", "local"),
        workflow_run_id: run_id,
        workflow_run_url: run_url,
        workflow_job: value(
            "QUALIFICATION_WORKFLOW_JOB",
            &value("GITHUB_JOB", "local-native-typprice"),
        ),
        workflow_attempt: value("GITHUB_RUN_ATTEMPT", "1"),
    }
}

fn platform() -> NativePlatform {
    #[cfg(target_arch = "x86_64")]
    {
        NativePlatform::X86
    }
    #[cfg(target_arch = "aarch64")]
    {
        NativePlatform::Aarch64 {
            talib_library: std::env::var_os("QUALIFICATION_TALIB_LIBRARY").map(PathBuf::from),
        }
    }
}

#[test]
#[ignore = "native Performance Qualification is run explicitly by the platform workflow"]
fn qualify_public_typprice_native() {
    let outcome = qualify_typprice(QualificationRequest {
        platform: platform(),
        provenance: provenance(),
    })
    .expect("native TYPPRICE qualification failed before producing valid evidence");

    let output = PathBuf::from(required("QUALIFICATION_OUTPUT"));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).expect("create qualification output directory");
    }
    std::fs::write(&output, outcome.jsonl()).expect("write qualification evidence");
    println!(
        "native TYPPRICE qualification evidence: {}",
        output.display()
    );
    outcome
        .require_pass()
        .expect("native TYPPRICE production selection failed its performance gate");
}
