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
