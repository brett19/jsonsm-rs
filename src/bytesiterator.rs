use crate::{simdsearch::search_bytes_simd_u8x16, simdsearch_ops::SimdSearch};

#[derive(Clone)]
pub struct BytesIterator<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> BytesIterator<'a> {
    #[inline]
    pub fn new(input: &'a [u8]) -> BytesIterator<'a> {
        BytesIterator { input, pos: 0 }
    }

    #[inline]
    pub fn skip(&mut self, count: usize) {
        self.pos += count;
    }

    #[inline]
    pub fn skip_until<P: FnMut(u8) -> bool>(
        &mut self,
        mut predicate: P,
    ) -> std::result::Result<(), ()> {
        if self.pos >= self.input.len() {
            return Err(());
        }

        for i in self.pos..self.input.len() {
            let c = unsafe { *self.input.get_unchecked(i) };
            if predicate(c) {
                self.pos = i;
                return Ok(());
            }
        }
        self.pos = self.input.len();
        Err(())
    }

    #[inline]
    pub fn skip_until_and_get<P: FnMut(u8) -> bool>(
        &mut self,
        mut predicate: P,
    ) -> std::result::Result<u8, ()> {
        if self.pos >= self.input.len() {
            return Err(());
        }

        for i in self.pos..self.input.len() {
            let c = unsafe { *self.input.get_unchecked(i) };
            if predicate(c) {
                self.pos = i + 1;
                return Ok(c);
            }
        }
        self.pos = self.input.len();
        Err(())
    }

    #[inline]
    pub fn skip_fast_until_and_get<C: SimdSearch<S>, S>(
        &mut self,
        state: &mut S,
        check: &mut C,
    ) -> std::result::Result<u8, ()> {
        if self.pos >= self.input.len() {
            return Err(());
        }

        match search_bytes_simd_u8x16(&self.input[self.pos..], state, check) {
            Some(i) => {
                self.pos += i + 1;
                Ok(unsafe { *self.input.get_unchecked(self.pos - 1) })
            }
            None => {
                self.pos = self.input.len();
                Err(())
            }
        }
    }

    #[inline]
    pub fn read_or_null(&mut self) -> u8 {
        if self.pos >= self.input.len() {
            return 0;
        }

        let c = unsafe { *self.input.get_unchecked(self.pos) };
        self.pos += 1;
        c
    }

    #[inline]
    pub fn read_multi<'b, const SIZE: usize>(
        &'b mut self,
    ) -> std::result::Result<&'a [u8; SIZE], ()> {
        if self.pos + SIZE > self.input.len() {
            self.pos = self.input.len();
            return Err(());
        }

        let v = unsafe { self.input.get_unchecked(self.pos..self.pos + SIZE) };

        self.pos += SIZE;
        Ok(v.try_into().unwrap())
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }
}
