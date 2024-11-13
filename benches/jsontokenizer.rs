use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jsonsm_rs::{jsontokenizer::JsonTokenizer, jsontokenizer_token::JsonTokenType};
use std::fs;

fn tokenize(json_bytes: &[u8]) {
    let mut t = JsonTokenizer::new(json_bytes);

    loop {
        let token = t.step().unwrap();
        if token.token_type == JsonTokenType::End {
            break;
        }
    }
}

fn hash(json_bytes: &[u8]) -> u64 {
    let mut pos: usize = 0;
    let mut x: u64 = 0;
    loop {
        if pos >= json_bytes.len() {
            break;
        }

        x += json_bytes[pos] as u64;
        pos += 1;
    }
    x
}

fn skipthrough(json_bytes: &[u8]) {
    let mut t = JsonTokenizer::new(json_bytes);
    t.skip_value().unwrap();
}

fn criterion_testdata(c: &mut Criterion, name: &str, path: &str) {
    let testdata = fs::read_to_string(path).expect("Should have been able to read test file");
    let json_bytes = testdata.as_bytes();

    let mut group = c.benchmark_group(name);
    group.throughput(criterion::Throughput::Bytes(json_bytes.len() as u64));
    group.bench_function("hash", |b| b.iter(|| hash(black_box(json_bytes))));
    group.bench_function("tokenize", |b| b.iter(|| tokenize(black_box(json_bytes))));
    group.bench_function("skip", |b| b.iter(|| skipthrough(black_box(json_bytes))));
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    criterion_testdata(c, "people", "testdata/people.json");
    criterion_testdata(c, "bigvector", "testdata/bigvector.json");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
