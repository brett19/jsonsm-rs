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
unsafe fn search_bytes_unaligned<PS: FnMut(&u8) -> bool>(
    start: *const u8,
    end: *const u8,
    mut slowfn: PS,
) -> Option<*const u8> {
    let mut cur = start;
    while cur != end {
        if slowfn(&*cur) {
            return Some(cur);
        }
        cur = cur.add(1);
    }
    None
}

#[inline]
unsafe fn search_bytes_simd_u8x16_aligned<P: FnMut(std::simd::u8x64) -> std::simd::mask8x64>(
    start: *const u8,
    end: *const u8,
    mut chkfn: P,
) -> Option<*const u8> {
    debug_assert!((start as *const std::simd::u8x64).is_aligned());
    debug_assert!((end as *const std::simd::u8x64).is_aligned());

    let mut cur = start;
    while cur != end {
        let x = std::simd::u8x64::load_select_ptr(
            cur,
            std::simd::Mask::splat(true),
            std::simd::Simd::splat(0),
        );

        let orz = chkfn(x);
        match std::simd::Mask::first_set(orz) {
            Some(i) => return Some(cur.add(i) as *const u8),
            None => (),
        };

        cur = cur.add(align_of::<std::simd::u8x64>());
    }
    None
}

#[inline]
unsafe fn search_bytes_simd_u8x16_ptr<
    PS: FnMut(&u8) -> bool,
    PF: FnMut(std::simd::u8x64) -> std::simd::mask8x64,
>(
    start: *const u8,
    end: *const u8,
    mut bytefn: PS,
    mut simdfn: PF,
) -> Option<*const u8> {
    const ALIGN: usize = align_of::<std::simd::u8x64>();

    let len = end.offset_from(start) as usize;
    if len < ALIGN {
        return search_bytes_unaligned(start, end, &mut bytefn);
    }

    let xstart = start.wrapping_add(start.align_offset(ALIGN));
    let xend = end.wrapping_sub(ALIGN - end.align_offset(ALIGN));

    if xstart > start {
        match search_bytes_unaligned(start, xstart, &mut bytefn) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    if xend != xstart {
        match search_bytes_simd_u8x16_aligned(xstart, xend, &mut simdfn) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    if end > xend {
        match search_bytes_unaligned(xend, end, &mut bytefn) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    None
}

#[inline]
fn search_bytes_simd_u8x16<
    PS: FnMut(&u8) -> bool,
    PF: FnMut(std::simd::u8x64) -> std::simd::mask8x64,
>(
    data: &[u8],
    mut bytefn: PS,
    mut simdfn: PF,
) -> Option<usize> {
    unsafe {
        let start = data.as_ptr();
        match search_bytes_simd_u8x16_ptr(start, start.add(data.len()), &mut bytefn, &mut simdfn) {
            Some(p) => Some(p.offset_from(start) as usize),
            None => None,
        }
    }
}

#[inline]
fn search_bytes_3b(data: &[u8], needles: [u8; 3]) -> Option<usize> {
    use std::ops::BitOr;
    use std::simd::cmp::SimdPartialEq;
    use std::simd::Simd;

    search_bytes_simd_u8x16(
        data,
        |c| {
            return *c == needles[0] || *c == needles[1] || *c == needles[2];
        },
        |x| {
            SimdPartialEq::simd_eq(x, Simd::splat(needles[0]))
                .bitor(SimdPartialEq::simd_eq(x, Simd::splat(needles[1])))
                .bitor(SimdPartialEq::simd_eq(x, Simd::splat(needles[2])))
        },
    )
}

#[inline]
fn search_bytes_6b(data: &[u8], needles: [u8; 6]) -> Option<usize> {
    use std::ops::BitOr;
    use std::simd::cmp::SimdPartialEq;
    use std::simd::Simd;

    search_bytes_simd_u8x16(
        data,
        |c| {
            return *c == needles[0]
                || *c == needles[1]
                || *c == needles[2]
                || *c == needles[3]
                || *c == needles[4]
                || *c == needles[5];
        },
        |x| {
            SimdPartialEq::simd_eq(x, Simd::splat(needles[0]))
                .bitor(SimdPartialEq::simd_eq(x, Simd::splat(needles[1])))
                .bitor(SimdPartialEq::simd_eq(x, Simd::splat(needles[2])))
                .bitor(SimdPartialEq::simd_eq(x, Simd::splat(needles[3])))
                .bitor(SimdPartialEq::simd_eq(x, Simd::splat(needles[4])))
                .bitor(SimdPartialEq::simd_eq(x, Simd::splat(needles[5])))
        },
    )
}

#[no_mangle]
fn find_byte_fast_aligned_16_test(data: &[u8]) -> Option<usize> {
    search_bytes_6b(data, [b'"', b'e', b']', b'}', b'[', b'{'])
}

fn find_end(json_bytes: &[u8]) {
    let x = match search_bytes_3b(json_bytes, [b'e', b']', b'}']) {
        Some(i) => i,
        None => 0,
    };
    assert_eq!(x, 15473);

    let x_and_on = unsafe { json_bytes.get_unchecked(x + 1..) };
    let y = match search_bytes_3b(x_and_on, [b'e', b']', b'}']) {
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
