//! ISO-8601 date/time parsing for the `DATE()` function.
//!
//! [`parse_epoch`] converts a date string to seconds since the Unix epoch (UTC). The
//! `DATE()` function returns this as a [`crate::value::FastVal::Float`], so date
//! comparisons are ordinary numeric comparisons — no implicit string→time coercion (that
//! would violate the strict-N1QL collation). Accepts `YYYY`, `YYYY-MM`, `YYYY-MM-DD`, and
//! `YYYY-MM-DD(T| )HH:MM[:SS[.fff]]` with an optional `Z` or `±HH[:MM]` timezone; missing
//! components default to the start of the period, matching gojsonsm's date handling.

/// Days from 1970-01-01 to the given proleptic-Gregorian date (Howard Hinnant's
/// `days_from_civil`, exact for all dates).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = if m > 2 { m - 3 } else { m + 9 }; // Mar=0 … Feb=11
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468
}

/// Read a run of ASCII digits starting at `*i`, returning their value (and advancing `i`).
/// Returns `None` if there is no digit.
fn read_uint(b: &[u8], i: &mut usize) -> Option<i64> {
    let start = *i;
    let mut v: i64 = 0;
    while *i < b.len() && b[*i].is_ascii_digit() {
        v = v.checked_mul(10)?.checked_add((b[*i] - b'0') as i64)?;
        *i += 1;
    }
    if *i == start {
        None
    } else {
        Some(v)
    }
}

/// Parse an ISO-8601 date/time into seconds since the Unix epoch (UTC), or `None` if the
/// string is not a valid date in the accepted forms.
pub fn parse_epoch(s: &str) -> Option<f64> {
    let s = s.trim();
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;

    let year = read_uint(b, &mut i)?;
    let mut month = 1;
    let mut day = 1;
    let (mut hh, mut mm, mut ss) = (0i64, 0i64, 0i64);
    let mut frac = 0.0f64;
    let mut tz_off = 0i64; // seconds east of UTC

    if i < n && b[i] == b'-' {
        i += 1;
        month = read_uint(b, &mut i)?;
        if i < n && b[i] == b'-' {
            i += 1;
            day = read_uint(b, &mut i)?;
            if i < n && matches!(b[i], b'T' | b't' | b' ') {
                i += 1;
                hh = read_uint(b, &mut i)?;
                if i >= n || b[i] != b':' {
                    return None;
                }
                i += 1;
                mm = read_uint(b, &mut i)?;
                if i < n && b[i] == b':' {
                    i += 1;
                    ss = read_uint(b, &mut i)?;
                    if i < n && b[i] == b'.' {
                        i += 1;
                        let start = i;
                        let digits = read_uint(b, &mut i)?;
                        let len = (i - start) as u32;
                        frac = digits as f64 / 10f64.powi(len as i32);
                    }
                }
                // Optional timezone.
                if i < n {
                    match b[i] {
                        b'Z' | b'z' => i += 1,
                        b'+' | b'-' => {
                            let sign = if b[i] == b'-' { -1 } else { 1 };
                            i += 1;
                            let oh = read_uint(b, &mut i)?;
                            let om = if i < n && b[i] == b':' {
                                i += 1;
                                read_uint(b, &mut i)?
                            } else {
                                0
                            };
                            tz_off = sign * (oh * 3600 + om * 60);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Reject trailing garbage and out-of-range components.
    if i != n
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hh > 23
        || mm > 59
        || ss > 60
    {
        return None;
    }

    let days = days_from_civil(year, month, day);
    let secs = days * 86_400 + hh * 3_600 + mm * 60 + ss - tz_off;
    Some(secs as f64 + frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_epochs() {
        assert_eq!(parse_epoch("1970-01-01T00:00:00Z"), Some(0.0));
        assert_eq!(parse_epoch("2020-01-01T00:00:00Z"), Some(1_577_836_800.0));
        assert_eq!(parse_epoch("2000-02-29T00:00:00Z"), Some(951_782_400.0)); // leap day
                                                                              // Date-only and partials default to the start of the period.
        assert_eq!(parse_epoch("2020-01-01"), Some(1_577_836_800.0));
        assert_eq!(parse_epoch("2020"), parse_epoch("2020-01-01T00:00:00"));
        assert_eq!(parse_epoch("2020-03"), parse_epoch("2020-03-01"));
    }

    #[test]
    fn timezones_and_fractions() {
        // +01:00 is one hour east: same wall clock is one hour earlier in UTC.
        assert_eq!(
            parse_epoch("2020-01-01T01:00:00+01:00"),
            Some(1_577_836_800.0)
        );
        assert_eq!(
            parse_epoch("2020-01-01T00:00:00-05:00"),
            Some(1_577_836_800.0 + 5.0 * 3600.0)
        );
        assert_eq!(parse_epoch("2020-01-01 00:00:00"), Some(1_577_836_800.0)); // space separator
        assert_eq!(parse_epoch("1970-01-01T00:00:00.5Z"), Some(0.5));
    }

    #[test]
    fn ordering_holds() {
        assert!(parse_epoch("2019-12-31").unwrap() < parse_epoch("2020-01-01").unwrap());
        assert!(
            parse_epoch("2020-01-01T00:00:01Z").unwrap()
                > parse_epoch("2020-01-01T00:00:00Z").unwrap()
        );
    }

    #[test]
    fn rejects_invalid() {
        assert_eq!(parse_epoch(""), None);
        assert_eq!(parse_epoch("not-a-date"), None);
        assert_eq!(parse_epoch("2020-13-01"), None); // month out of range
        assert_eq!(parse_epoch("2020-01-32"), None); // day out of range
        assert_eq!(parse_epoch("2020-01-01T25:00:00"), None); // hour out of range
        assert_eq!(parse_epoch("2020-01-01xyz"), None); // trailing garbage
    }
}
