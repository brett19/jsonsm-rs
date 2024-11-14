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

    #[no_mangle]
    pub fn skip_until_bracket(&mut self) -> std::result::Result<(), ()> {
        if self.pos >= self.input.len() {
            return Err(());
        }

        let bytes = unsafe { self.input.get_unchecked(self.pos..) };
        let mut i = bytes.iter();
        while i.next() != Some(&b']') {}

        Ok(())
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
    pub fn read_multi<const SIZE: usize>(&mut self) -> std::result::Result<&[u8; SIZE], ()> {
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
