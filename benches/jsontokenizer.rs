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

fn criterion_benchmark(c: &mut Criterion) {
    let testdata = fs::read_to_string("testdata/people.json")
        .expect("Should have been able to read test file");
    let json_bytes = testdata.as_bytes();

    let mut group = c.benchmark_group("tokenize-throughput");
    group.throughput(criterion::Throughput::Bytes(json_bytes.len() as u64));
    group.bench_function("tokenize", |b| b.iter(|| tokenize(black_box(json_bytes))));
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
