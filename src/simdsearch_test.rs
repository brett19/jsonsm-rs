#[cfg(test)]
mod tests {
    use crate::{
        simdsearch::search_bytes_simd_u8x16,
        simdsearch_ops::{SimdSearch, SimdSearchDualExec, SimdSearchEq, SimdSearchExec},
    };

    fn check_one<C: SimdSearch<S>, S>(
        input: &str,
        state: &mut S,
        check: &mut C,
        expected: Option<usize>,
    ) where
        S: Clone + PartialEq + std::fmt::Debug,
    {
        let input_bytes = input.as_bytes();

        let base_state = state.clone();
        let r = search_bytes_simd_u8x16(input_bytes, state, check);
        assert_eq!(r, expected);

        for i in 0..64 {
            let mut before = vec![b'0'; i];
            let mut after = vec![b'0'; 64 - i];
            let mut tinput = Vec::with_capacity(input.len() + 64);
            tinput.append(&mut before);
            tinput.extend(input_bytes);
            tinput.append(&mut after);

            let mut tstate = base_state.clone();
            let tr = search_bytes_simd_u8x16(&tinput[..], &mut tstate, check);
            match expected {
                Some(expected) => assert_eq!(tr, Some(expected + i), "i={}", i),
                None => assert_eq!(r, None),
            }
        }
    }

    #[test]
    fn simdsearch_find() {
        check_one(
            "0000000010000000",
            &mut (),
            &mut SimdSearchEq::new(b'1'),
            Some(8),
        );
    }

    #[test]
    fn simdsearch_count_and_find() {
        #[derive(PartialEq, Clone, Debug)]
        struct CountState {
            num: usize,
        }
        let mut state = CountState { num: 0 };

        check_one(
            "0010000010002010",
            &mut state,
            &mut SimdSearchDualExec::new(
                SimdSearchEq::new(b'1'),
                SimdSearchEq::new(b'2'),
                |state: &mut CountState, v1, v2| {
                    state.num += if v1 { 1 } else { 0 };
                    if v2 {
                        assert_eq!(state.num, 2);
                    }
                    v2
                },
            ),
            Some(12),
        );
    }

    #[test]
    fn simdsearch_count() {
        #[derive(PartialEq, Clone, Debug)]
        struct CountState {
            num: usize,
        }
        let mut state = CountState { num: 0 };

        check_one(
            "0010000010000010",
            &mut state,
            &mut SimdSearchExec::new(SimdSearchEq::new(b'1'), |state: &mut CountState, v| {
                state.num += if v { 1 } else { 0 };
                false
            }),
            None,
        );
        assert_eq!(state.num, 3);
    }

    #[test]
    fn simdsearch_brett() {
        assert_eq!(
            search_bytes_simd_u8x16(
                "000000000000000000000001000000000000000000000000".as_bytes(),
                &mut (),
                &mut SimdSearchExec::new(SimdSearchEq::new(b'1'), |_, v| { v })
            ),
            Some(23)
        );
    }
}
