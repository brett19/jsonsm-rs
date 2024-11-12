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

fn criterion_benchmark(c: &mut Criterion) {
    let testdata = fs::read_to_string("testdata/people.json")
        .expect("Should have been able to read test file");
    let json_bytes = testdata.as_bytes();

    let mut group = c.benchmark_group("tokenize-throughput");
    group.throughput(criterion::Throughput::Bytes(json_bytes.len() as u64));
    group.bench_function("hash", |b| b.iter(|| hash(black_box(json_bytes))));
    group.bench_function("tokenize", |b| b.iter(|| tokenize(black_box(json_bytes))));
    group.bench_function("skipthrough", |b| {
        b.iter(|| skipthrough(black_box(json_bytes)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
