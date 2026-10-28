//! Arithmetic expressions for the `Formula` node when the host has no
//! evaluator of its own.
//!
//! Numbers, `+ - * / % ^`, parentheses, `pi`, `e`, and the functions `sin cos
//! tan asin acos atan` (degrees), `atan2 sqrt abs exp ln log floor ceil round
//! min max pow`.

/// The value of `text`, or `None` when it does not parse or is not finite.
pub fn eval(text: &str) -> Option<f64> {
    let mut parser = Parser { s: text.as_bytes(), i: 0 };
    let value = parser.sum()?;
    parser.skip();
    (parser.i == parser.s.len() && value.is_finite()).then_some(value)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn skip(&mut self) {
        while self.s.get(self.i).is_some_and(u8::is_ascii_whitespace) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.skip();
        let hit = self.s.get(self.i) == Some(&c);
        self.i += usize::from(hit);
        hit
    }

    fn sum(&mut self) -> Option<f64> {
        let mut value = self.product()?;
        loop {
            if self.eat(b'+') {
                value += self.product()?;
            } else if self.eat(b'-') {
                value -= self.product()?;
            } else {
                return Some(value);
            }
        }
    }

    fn product(&mut self) -> Option<f64> {
        let mut value = self.unary()?;
        loop {
            if self.eat(b'*') {
                value *= self.unary()?;
            } else if self.eat(b'/') {
                value /= self.unary()?;
            } else if self.eat(b'%') {
                value = value.rem_euclid(self.unary()?);
            } else {
                return Some(value);
            }
        }
    }

    fn unary(&mut self) -> Option<f64> {
        if self.eat(b'-') {
            return Some(-self.unary()?);
        }
        if self.eat(b'+') {
            return self.unary();
        }
        let base = self.atom()?;
        // Right-associative, and binds tighter than a leading minus.
        if self.eat(b'^') {
            return Some(base.powf(self.unary()?));
        }
        Some(base)
    }

    fn atom(&mut self) -> Option<f64> {
        self.skip();
        if self.eat(b'(') {
            let value = self.sum()?;
            return self.eat(b')').then_some(value);
        }
        let start = self.i;
        let c = *self.s.get(self.i)?;
        if c.is_ascii_digit() || c == b'.' {
            while self.s.get(self.i).is_some_and(|c| c.is_ascii_digit() || *c == b'.') {
                self.i += 1;
            }
            if self.s.get(self.i).is_some_and(|c| matches!(c, b'e' | b'E')) {
                let mark = self.i;
                self.i += 1;
                if self.s.get(self.i).is_some_and(|c| matches!(c, b'+' | b'-')) {
                    self.i += 1;
                }
                if self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                    while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
                        self.i += 1;
                    }
                } else {
                    self.i = mark;
                }
            }
            return std::str::from_utf8(&self.s[start..self.i]).ok()?.parse().ok();
        }
        while self.s.get(self.i).is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_') {
            self.i += 1;
        }
        let name = std::str::from_utf8(&self.s[start..self.i]).ok()?.to_ascii_lowercase();
        match name.as_str() {
            "pi" => return Some(std::f64::consts::PI),
            "e" => return Some(std::f64::consts::E),
            "" => return None,
            _ => {}
        }
        if !self.eat(b'(') {
            return None;
        }
        let mut args = vec![self.sum()?];
        while self.eat(b',') {
            args.push(self.sum()?);
        }
        if !self.eat(b')') {
            return None;
        }
        let x = args[0];
        Some(match (name.as_str(), args.len()) {
            ("sin", 1) => x.to_radians().sin(),
            ("cos", 1) => x.to_radians().cos(),
            ("tan", 1) => x.to_radians().tan(),
            ("asin", 1) => x.asin().to_degrees(),
            ("acos", 1) => x.acos().to_degrees(),
            ("atan", 1) => x.atan().to_degrees(),
            ("atan2", 2) => x.atan2(args[1]).to_degrees(),
            ("sqrt", 1) => x.sqrt(),
            ("abs", 1) => x.abs(),
            ("exp", 1) => x.exp(),
            ("ln", 1) => x.ln(),
            ("log", 1) => x.log10(),
            ("floor", 1) => x.floor(),
            ("ceil", 1) => x.ceil(),
            ("round", 1) => x.round(),
            ("min", _) => args.iter().copied().fold(f64::INFINITY, f64::min),
            ("max", _) => args.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ("pow", 2) => x.powf(args[1]),
            _ => return None,
        })
    }
}
