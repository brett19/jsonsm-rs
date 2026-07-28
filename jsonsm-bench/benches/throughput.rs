//! `cargo bench -p jsonsm-bench` — the whole table.
//!
//! Sweeps and repetitions are tunable without recompiling:
//!
//! ```text
//! JSONSM_BENCH_ITERS=5000 JSONSM_BENCH_REPS=7 cargo bench -p jsonsm-bench
//! ```
//!
//! This reports wall-clock MB/s, which on this codebase is the *secondary* metric — see the
//! crate documentation for why, and for the `perf stat` invocation that produces the numbers
//! quoted for this project.

fn env_or(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // Cargo passes libtest flags (`--bench`, and `--save-baseline` and friends when a harness
    // understands them). None apply here; accepting and ignoring them keeps `cargo bench`
    // working rather than failing on an unexpected argument.
    jsonsm_bench::run_table(
        env_or("JSONSM_BENCH_ITERS", 2000),
        env_or("JSONSM_BENCH_REPS", 5),
    );
}
