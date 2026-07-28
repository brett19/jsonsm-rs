//! One workload, one engine, no warm-up — the shape `perf stat` wants.
//!
//! ```text
//! cargo build --release -p jsonsm-bench
//! perf stat -r 5 -e instructions,cycles \\
//!     target/release/jsonsm-bench match/late_field sse2 2000
//! ```
//!
//! `all` runs the full table instead, which is what `cargo bench -p jsonsm-bench` does.
//! Run with no arguments for the workload list.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        None | Some("-h") | Some("--help") => {
            eprintln!("usage: jsonsm-bench all [iters] [reps]");
            eprintln!("       jsonsm-bench <workload> <engine> [iters]");
            eprintln!("\nengines: scalar | sse2 | avx2 | hybrid | slow");
            eprintln!("workloads:");
            for w in jsonsm_bench::workloads() {
                eprintln!("  {}", w.name);
            }
        }
        Some("all") => jsonsm_bench::run_table(
            args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2000),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5),
        ),
        Some(name) => jsonsm_bench::run_single(
            name,
            args.get(2).map(String::as_str).unwrap_or("sse2"),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000),
        ),
    }
}
