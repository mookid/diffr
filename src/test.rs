use super::*;

#[test]
fn skip_all_escape_code_test() {
    assert_eq!(5, skip_all_escape_code(b"\x1b[42m@@@"));
    assert_eq!(10, skip_all_escape_code(b"\x1b[42m\x1b[33m@@@"));
    assert_eq!(0, skip_all_escape_code(b"\x1b[42@@@"));
}

#[test]
fn first_after_escape_test() {
    assert_eq!(Some(b'+'), first_after_escape(b"+abc"));
    assert_eq!(Some(b'+'), first_after_escape(b"\x1b[42m\x1b[33m+abc"));
    assert_eq!(None, first_after_escape(b"\x1b[42m"));
}

// TODO test index_of?

#[test]
fn skip_token_test() {
    assert_eq!(4, skip_token(b"abc\x1b"));
    assert_eq!(3, skip_token(b"abc\x1b["));
    assert_eq!(3, skip_token(b"abc"));
    assert_eq!(1, skip_token(b"\x1b"));
    assert_eq!(0, skip_token(b""));
}

#[test]
fn parse_line_number_test() {
    let test_ok = |ofs1, len1, ofs2, len2, input| {
        eprintln!("test_ok {}...", String::from_utf8_lossy(input));
        assert_eq!(
            Some(HunkHeader {
                minus_range: (ofs1, len1),
                plus_range: (ofs2, len2),
            }),
            parse_line_number(input)
        );
    };
    let test_fail = |input| {
        eprintln!("test_fail {}...", String::from_utf8_lossy(input));
        assert_eq!(None, parse_line_number(input));
    };
    test_ok(133, 6, 133, 8, b"@@ -133,6 +133,8 @@");
    test_ok(0, 0, 0, 1, b"@@ -0,0 +1 @@");
    test_ok(0, 0, 0, 1, b"  @@ -0,0 +1 @@");
    test_ok(0, 0, 0, 1, b"@@   -0,0 +1 @@");
    test_fail(b"@@-0,0 +1 @@");
    test_fail(b"@@ -0,0+1 @@");
    test_fail(b"@@ -0,0 +1@@");
    test_fail(b"@@ -0,0 +1 ");
    test_fail(b"-0,0 +1");
    test_fail(b"@@ 0,0 +1 @@");
    test_fail(b"@@ -0,0 1 @@");

    // overflow
    test_fail(b"@@ -0,0 +19999999999999999999 @@");

    // with escape code
    test_ok(0, 0, 0, 1, b"\x1b[42;43m@\x1b[42;43m@\x1b[42;43m \x1b[42;43m-\x1b[42;43m0\x1b[42;43m,\x1b[42;43m0\x1b[42;43m \x1b[42;43m+1 @@");
}

#[test]
fn test_width() {
    for (i, x) in WIDTH.iter().enumerate() {
        if x < &usize::max_value() {
            assert_eq!(format!("{}", x + 1).len(), i + 1);
        }
    }
    assert_eq!(0, width1(0));
    fn test(x: usize) {
        assert_eq!(format!("{}", x).len(), width1(x));
    }
    for i in 1..=10000 {
        test(i);
    }
    test(9999999999);
    test(10000000000);
    test(14284238234);
    assert_eq!("123:456".len(), HunkHeader::new(123, 5, 456, 9).width());
    assert_eq!("1122:456".len(), HunkHeader::new(123, 999, 456, 9).width());
    assert_eq!(":456".len(), HunkHeader::new(0, 0, 456, 9).width());

    for i in 0..64 {
        test(1 << i);
    }
    test(usize::max_value());
}
