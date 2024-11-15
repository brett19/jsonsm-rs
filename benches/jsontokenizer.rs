#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jsonsm_rs::{
    jsontokenizer::JsonTokenizer,
    jsontokenizer_token::JsonTokenType,
    jsontokenizerx::JsonTokenizerX,
    simdsearch::search_bytes_simd_u8x16,
    simdsearch_ops::{SimdSearchEq, SimdSearchOr},
};
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
    let mut s = JsonTokenizerX::new(json_bytes);
    s.skip_over_value().unwrap();
}

#[no_mangle]
fn find_byte_fast_aligned_16_test(data: &[u8]) -> Option<usize> {
    search_bytes_simd_u8x16(data, &mut (), &mut SimdSearchEq::new(b'e'))
}

fn find_end(json_bytes: &[u8]) {
    let mut c = SimdSearchOr::new(
        SimdSearchEq::new(b'e'),
        SimdSearchOr::new(SimdSearchEq::new(b']'), SimdSearchEq::new(b'}')),
    );
    let x = match search_bytes_simd_u8x16(json_bytes, &mut (), &mut c) {
        Some(i) => i,
        None => 0,
    };
    assert_eq!(x, 15473);

    let x_and_on = unsafe { json_bytes.get_unchecked(x + 1..) };
    let y = match search_bytes_simd_u8x16(x_and_on, &mut (), &mut c) {
        Some(i) => i,
        None => 0,
    };
    assert_eq!(y, 21329 - 15473 - 1);
}

fn criterion_testdata(c: &mut Criterion, name: &str, path: &str) {
    let testdatastr = fs::read_to_string(path).unwrap();
    let testdata = testdatastr.as_bytes();

    let mut group = c.benchmark_group(name);
    group.throughput(criterion::Throughput::Bytes(testdata.len() as u64));
    group.bench_function("hash", |b| b.iter(|| hash(black_box(testdata))));
    group.bench_function("tokenize", |b| b.iter(|| tokenize(black_box(testdata))));
    group.bench_function("skip", |b| b.iter(|| skipthrough(black_box(testdata))));
    group.bench_function("findend", |b| b.iter(|| find_end(black_box(testdata))));
    group.finish();
}

fn criterion_testdata_all(c: &mut Criterion) {
    let mut testdatas: Vec<String> = Vec::new();
    let mut testdataslen: usize = 0;
    for (_, path) in TESTDATAPATHS.iter() {
        let testdata = fs::read_to_string(path).unwrap();
        testdataslen += testdata.as_bytes().len();
        testdatas.push(testdata);
    }

    let mut group = c.benchmark_group("all");
    group.throughput(criterion::Throughput::Bytes(testdataslen as u64));
    group.bench_function("hash", |b| {
        b.iter(|| {
            for testdata in testdatas.iter() {
                hash(black_box(testdata.as_bytes()));
            }
        })
    });
    group.bench_function("tokenize", |b| {
        b.iter(|| {
            for testdata in testdatas.iter() {
                tokenize(black_box(testdata.as_bytes()));
            }
        })
    });
    group.bench_function("skip", |b| {
        b.iter(|| {
            for testdata in testdatas.iter() {
                skipthrough(black_box(testdata.as_bytes()));
            }
        })
    });
    group.bench_function("findend", |b| {
        b.iter(|| {
            for testdata in testdatas.iter() {
                find_end(black_box(testdata.as_bytes()));
            }
        })
    });
    group.finish();
}

const TESTDATAPATHS: [(&str, &str); 2] = [
    ("people", "testdata/people.json"),
    ("bigvector", "testdata/bigvector.json"),
];

fn criterion_benchmark(c: &mut Criterion) {
    for (name, path) in TESTDATAPATHS.iter() {
        criterion_testdata(c, name, path);
    }
    criterion_testdata_all(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
