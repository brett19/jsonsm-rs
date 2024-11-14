#![feature(portable_simd)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jsonsm_rs::{
    jsontokenizer::JsonTokenizer, jsontokenizer_token::JsonTokenType,
    jsontokenizerx::JsonTokenizerX,
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

#[inline]
unsafe fn find_byte_by_byte<const NUM_CMP: usize>(
    start: *const u8,
    end: *const u8,
    needles: &[u8; NUM_CMP],
) -> Option<*const u8> {
    let mut cur = start;
    while cur != end {
        for i in 0..NUM_CMP {
            if *cur == needles[i] {
                return Some(cur);
            }
        }
        cur = cur.add(1);
    }
    None
}

#[inline]
unsafe fn find_byte_fast_aligned_32<const NUM_CMP: usize>(
    start: *const u8,
    end: *const u8,
    needles: &[u8; NUM_CMP],
) -> Option<*const u8> {
    use std::ops::BitOr;

    const ALIGN: usize = align_of::<std::simd::u8x32>();

    let mut cur = start;
    while cur != end {
        let x = std::simd::u8x32::load_select_ptr(
            cur,
            std::simd::Mask::splat(true),
            std::simd::Simd::splat(0),
        );

        let mut orz = std::simd::Mask::splat(false);
        for i in 0..NUM_CMP {
            orz = orz.bitor(std::simd::cmp::SimdPartialEq::simd_eq(
                x,
                std::simd::Simd::splat(needles[i]),
            ));
        }

        match std::simd::Mask::first_set(orz) {
            Some(i) => return Some(cur.add(i) as *const u8),
            None => (),
        };

        cur = cur.add(ALIGN);
    }

    None
}

#[no_mangle]
unsafe fn find_byte_fast_aligned_32_test(start: *const u8, end: *const u8) -> Option<*const u8> {
    find_byte_fast_aligned_32(start, end, &[b'"', b']', b'}', b'[', b'{', b'e'])
}

#[inline]
unsafe fn find_byte_fast<const NUM_CMP: usize>(
    start: *const u8,
    end: *const u8,
    needles: &[u8; NUM_CMP],
) -> Option<*const u8> {
    const ALIGN: usize = align_of::<std::simd::u8x32>();

    let len = end.offset_from(start) as usize;
    if len < ALIGN {
        return find_byte_by_byte(start, end, needles);
    }

    let xstart = start.wrapping_add(start.align_offset(ALIGN));
    let xend = end.wrapping_sub(ALIGN - end.align_offset(ALIGN));

    if xstart > start {
        match find_byte_by_byte(start, xstart, needles) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    if xend != xstart {
        match find_byte_fast_aligned_32(xstart, xend, needles) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    if end > xend {
        match find_byte_by_byte(xend, end, needles) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    None
}

#[inline]
fn find_byte<const NUM_CMP: usize>(data: &[u8], needles: &[u8; NUM_CMP]) -> Option<usize> {
    unsafe {
        let start = data.as_ptr();
        match find_byte_fast(start, start.add(data.len()), needles) {
            Some(p) => Some(p.offset_from(start) as usize),
            None => None,
        }
    }
}

fn find_end(json_bytes: &[u8]) {
    let x = match find_byte(json_bytes, &[b'"', b']', b'}', b'[', b'{', b'e']) {
        Some(i) => i,
        None => 0,
    };
    assert_eq!(x, 15473);

    let x_and_on = unsafe { json_bytes.get_unchecked(x + 1..) };
    let y = match find_byte(x_and_on, &[b'"', b']', b'}', b'[', b'{', b'e']) {
        Some(i) => i,
        None => 0,
    };
    assert_eq!(y, 21329 - 15473 - 1);
}

fn criterion_testdata(c: &mut Criterion, name: &str, path: &str) {
    let testdata = fs::read_to_string(path).expect("Should have been able to read test file");
    let json_bytes = testdata.as_bytes();

    let mut group = c.benchmark_group(name);
    group.throughput(criterion::Throughput::Bytes(json_bytes.len() as u64));
    group.bench_function("hash", |b| b.iter(|| hash(black_box(json_bytes))));
    group.bench_function("tokenize", |b| b.iter(|| tokenize(black_box(json_bytes))));
    group.bench_function("skip", |b| b.iter(|| skipthrough(black_box(json_bytes))));
    group.bench_function("findend", |b| b.iter(|| find_end(black_box(json_bytes))));
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    criterion_testdata(c, "people", "testdata/people.json");
    criterion_testdata(c, "bigvector", "testdata/bigvector.json");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
