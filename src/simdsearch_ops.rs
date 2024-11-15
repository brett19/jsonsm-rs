use std::marker::PhantomData;

use crate::simdsearch::{SimdArr, SimdMask};

pub trait SimdSearch<S>
where
    Self: Sized,
{
    fn for_simd(&mut self, s: &mut S, v: &SimdArr) -> SimdMask;
    fn for_test(&mut self, s: &mut S, v: &u8) -> bool;

    fn or<Y: SimdSearch<S>>(self, rhs: Y) -> SimdSearchOr<Self, Y, S> {
        SimdSearchOr::new(self, rhs)
    }
    fn or_eq(self, v: u8) -> SimdSearchOr<Self, SimdSearchEq<S>, S> {
        SimdSearchOr::new(self, SimdSearchEq::new(v))
    }
    fn or_range(self, start: u8, end: u8) -> SimdSearchOr<Self, SimdSearchRange<S>, S> {
        SimdSearchOr::new(self, SimdSearchRange::new(start, end))
    }
}

pub struct SimdSearchOr<X: SimdSearch<S>, Y: SimdSearch<S>, S> {
    left: X,
    right: Y,
    phantom: PhantomData<S>,
}

impl<X: SimdSearch<S>, Y: SimdSearch<S>, S> SimdSearchOr<X, Y, S> {
    #[inline]
    pub fn new(left: X, right: Y) -> Self {
        Self {
            left,
            right,
            phantom: PhantomData,
        }
    }
}

impl<X: SimdSearch<S>, Y: SimdSearch<S>, S> SimdSearch<S> for SimdSearchOr<X, Y, S> {
    #[inline]
    fn for_simd(&mut self, s: &mut S, v: &SimdArr) -> SimdMask {
        use std::ops::BitOr;
        let left = self.left.for_simd(s, v);
        let right = self.right.for_simd(s, v);
        left.bitor(right)
    }

    #[inline]
    fn for_test(&mut self, s: &mut S, v: &u8) -> bool {
        self.left.for_test(s, v) || self.right.for_test(s, v)
    }
}

pub struct SimdSearchNot<X: SimdSearch<S>, S> {
    left: X,
    phantom: PhantomData<S>,
}

impl<X: SimdSearch<S>, S> SimdSearchNot<X, S> {
    #[inline]
    pub fn new(left: X) -> Self {
        Self {
            left,
            phantom: PhantomData,
        }
    }
}

impl<X: SimdSearch<S>, S> SimdSearch<S> for SimdSearchNot<X, S> {
    #[inline]
    fn for_simd(&mut self, s: &mut S, v: &SimdArr) -> SimdMask {
        use std::ops::Not;
        self.left.for_simd(s, v).not()
    }

    #[inline]
    fn for_test(&mut self, s: &mut S, v: &u8) -> bool {
        !self.left.for_test(s, v)
    }
}

pub struct SimdSearchEq<S> {
    needle: u8,
    phantom: PhantomData<S>,
}

impl<S> SimdSearchEq<S> {
    #[inline]
    pub fn new(needle: u8) -> Self {
        Self {
            needle,
            phantom: PhantomData,
        }
    }
}

impl<S> SimdSearch<S> for SimdSearchEq<S> {
    #[inline]
    fn for_simd(&mut self, _: &mut S, v: &SimdArr) -> SimdMask {
        use std::simd::cmp::SimdPartialEq;
        use std::simd::Simd;

        v.simd_eq(Simd::splat(self.needle))
    }

    #[inline]
    fn for_test(&mut self, _: &mut S, v: &u8) -> bool {
        *v == self.needle
    }
}

pub struct SimdSearchRange<S> {
    start: u8,
    end: u8,
    phantom: PhantomData<S>,
}

impl<S> SimdSearchRange<S> {
    #[inline]
    pub fn new(start: u8, end: u8) -> Self {
        Self {
            start,
            end,
            phantom: PhantomData,
        }
    }
}

impl<S> SimdSearch<S> for SimdSearchRange<S> {
    #[inline]
    fn for_simd(&mut self, _: &mut S, v: &SimdArr) -> SimdMask {
        use std::ops::Sub;
        use std::simd::cmp::SimdPartialOrd;
        use std::simd::Simd;

        v.sub(Simd::splat(self.start))
            .simd_le(Simd::splat(self.end - self.start))
    }

    #[inline]
    fn for_test(&mut self, _: &mut S, v: &u8) -> bool {
        *v >= self.start && *v <= self.end
    }
}

pub struct SimdSearchExec<X: SimdSearch<S>, FN: FnMut(&mut S, bool) -> bool, S> {
    left: X,
    func: FN,
    phantom: PhantomData<S>,
}

impl<X: SimdSearch<S>, FN: FnMut(&mut S, bool) -> bool, S> SimdSearchExec<X, FN, S> {
    #[inline]
    pub fn new(left: X, func: FN) -> Self {
        Self {
            left,
            func,
            phantom: PhantomData,
        }
    }
}

impl<X: SimdSearch<S>, FN: FnMut(&mut S, bool) -> bool, S> SimdSearch<S>
    for SimdSearchExec<X, FN, S>
{
    #[inline]
    fn for_simd(&mut self, s: &mut S, v: &SimdArr) -> SimdMask {
        let m = self.left.for_simd(s, v);
        if m.any() {
            let mb = m.to_bitmask();
            let mut fmb: u64 = 0;
            for i in 0..SimdArr::LEN {
                fmb |= if (self.func)(s, mb & (1 << i) != 0) {
                    1 << i
                } else {
                    0
                }
            }

            SimdMask::from_bitmask(fmb)
        } else {
            m
        }
    }

    #[inline]
    fn for_test(&mut self, s: &mut S, v: &u8) -> bool {
        let v = self.left.for_test(s, v);
        (self.func)(s, v)
    }
}

pub struct SimdSearchDualExec<
    X: SimdSearch<S>,
    Y: SimdSearch<S>,
    FN: FnMut(&mut S, bool, bool) -> bool,
    S,
> {
    left: X,
    left2: Y,
    func: FN,
    phantom: PhantomData<S>,
}

impl<X: SimdSearch<S>, Y: SimdSearch<S>, FN: FnMut(&mut S, bool, bool) -> bool, S>
    SimdSearchDualExec<X, Y, FN, S>
{
    #[inline]
    pub fn new(left: X, left2: Y, func: FN) -> Self {
        Self {
            left,
            left2,
            func,
            phantom: PhantomData,
        }
    }
}

impl<X: SimdSearch<S>, Y: SimdSearch<S>, FN: FnMut(&mut S, bool, bool) -> bool, S> SimdSearch<S>
    for SimdSearchDualExec<X, Y, FN, S>
{
    #[inline]
    fn for_simd(&mut self, s: &mut S, v: &SimdArr) -> SimdMask {
        let m1 = self.left.for_simd(s, v);
        let m2 = self.left2.for_simd(s, v);
        if (m1 | m2).any() {
            let mb1 = m1.to_bitmask();
            let mb2 = m2.to_bitmask();
            let mut fmb: u64 = 0;
            for i in 0..SimdArr::LEN {
                let v1 = mb1 & (1 << i) != 0;
                let v2 = mb2 & (1 << i) != 0;
                fmb |= if (self.func)(s, v1, v2) { 1 << i } else { 0 }
            }

            SimdMask::from_bitmask(fmb)
        } else {
            // both m1 and m2 are blank, return either
            m1
        }
    }

    #[inline]
    fn for_test(&mut self, s: &mut S, v: &u8) -> bool {
        let v1 = self.left.for_test(s, v);
        let v2 = self.left2.for_test(s, v);
        (self.func)(s, v1, v2)
    }
}

#[no_mangle]
fn test(x: &SimdArr) -> SimdMask {
    struct TestState {
        depth: isize,
    }
    let mut state = TestState { depth: 1 };

    SimdSearchExec::new(
        SimdSearchEq::new(b'{').or_eq(b'['),
        |s: &mut TestState, v| {
            s.depth += if v { 1 } else { 0 };
            false
        },
    )
    .or(SimdSearchExec::new(
        SimdSearchEq::new(b'}').or_eq(b']'),
        |s: &mut TestState, v| {
            s.depth -= if v { 1 } else { 0 };
            s.depth == 0
        },
    ))
    .for_simd(&mut state, x)
}
