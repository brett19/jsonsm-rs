use crate::simdsearch_ops::SimdSearch;

pub type SimdArr = std::simd::u8x16;
pub type SimdMask = std::simd::mask8x16;

#[inline]
unsafe fn search_bytes_unaligned<C: SimdSearch<S>, S>(
    start: *const u8,
    end: *const u8,
    state: &mut S,
    check: &mut C,
) -> Option<*const u8> {
    let mut cur = start;
    while cur != end {
        if check.for_test(state, &*cur) {
            return Some(cur);
        }
        cur = cur.add(1);
    }
    None
}

#[inline]
unsafe fn search_bytes_simd_u8x16_aligned<C: SimdSearch<S>, S>(
    start: *const u8,
    end: *const u8,
    state: &mut S,
    check: &mut C,
) -> Option<*const u8> {
    debug_assert!((start as *const SimdArr).is_aligned());
    debug_assert!((end as *const SimdArr).is_aligned());

    let mut cur = start;
    while cur != end {
        let x =
            SimdArr::load_select_ptr(cur, std::simd::Mask::splat(true), std::simd::Simd::splat(0));

        let orz = check.for_simd(state, &x);
        match std::simd::Mask::first_set(orz) {
            Some(i) => return Some(cur.add(i) as *const u8),
            None => (),
        };

        cur = cur.add(align_of::<SimdArr>());
    }
    None
}

#[inline]
unsafe fn search_bytes_simd_u8x16_ptr<C: SimdSearch<S>, S>(
    start: *const u8,
    end: *const u8,
    state: &mut S,
    check: &mut C,
) -> Option<*const u8> {
    const ALIGN: usize = align_of::<SimdArr>();

    let len = end.offset_from(start) as usize;
    if len < ALIGN {
        return search_bytes_unaligned(start, end, state, check);
    }

    let xstart = start.wrapping_add(start.align_offset(ALIGN));
    let xend = end.wrapping_sub(ALIGN - end.align_offset(ALIGN));

    if xstart > start {
        match search_bytes_unaligned(start, xstart, state, check) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    if xend != xstart {
        match search_bytes_simd_u8x16_aligned(xstart, xend, state, check) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    if end > xend {
        match search_bytes_unaligned(xend, end, state, check) {
            Some(i) => return Some(i),
            None => (),
        }
    }

    None
}

#[inline]
pub fn search_bytes_simd_u8x16<C: SimdSearch<S>, S>(
    data: &[u8],
    state: &mut S,
    check: &mut C,
) -> Option<usize> {
    unsafe {
        let start = data.as_ptr();
        match search_bytes_simd_u8x16_ptr(start, start.add(data.len()), state, check) {
            Some(p) => Some(p.offset_from(start) as usize),
            None => None,
        }
    }
}
