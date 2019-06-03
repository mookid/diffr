use super::*;

fn string_of_bytes(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf).into()
}

fn to_strings(buf: &[&[u8]]) -> Vec<String> {
    mk_vec(buf.iter().map(|buf| string_of_bytes(buf)))
}

fn mk_vec<It, T>(it: It) -> Vec<T>
where
    It: Iterator<Item = T>,
{
    it.collect()
}

fn compress_path(values: &Vec<(Vec<u8>, DiffKind)>) -> Vec<(Vec<u8>, DiffKind)> {
    let mut values = values.clone();
    let mut it = values.iter_mut();
    let mut result: Vec<(Vec<u8>, DiffKind)> = vec![];
    let mut current = it.next();
    while let Some(next) = it.next() {
        match current {
            Some(ref mut c) => {
                if c.1 == next.1 {
                    c.0.extend_from_slice(&*next.0)
                } else {
                    result.push(c.clone());
                    *c = next;
                }
            }
            None => panic!(),
        }
    }

    if let Some(last) = current {
        result.push(last.clone());
    }
    result
}

fn diff_sequences_test(expected: &[(&[u8], DiffKind)], seq_a: &[u8], seq_b: &[u8]) {
    fn mk_tokens(buf: &[u8]) -> Vec<&[u8]> {
        (0..buf.len()).map(|i| &buf[i..i + 1]).collect()
    };

    let toks_a = mk_tokens(seq_a);
    let toks_b = mk_tokens(seq_b);

    let mut v = vec![];
    let input = DiffInput::new(&toks_a, &toks_b);
    let diff = diff_sequences_simple(&input, &mut v, true);
    let diff_bwd = diff_sequences_simple(&input, &mut v, false);
    let input_r = DiffInput::new(&toks_b, &toks_a);
    let diff_r = diff_sequences_simple(&input_r, &mut v, true);
    let diff_r_bwd = diff_sequences_simple(&input_r, &mut v, false);

    let /*mut*/ path = vec![];
    let /*mut*/ path_added = vec![];
    let /*mut*/ path_removed = vec![];

    // for item in diff.path() {
    //     let kind = item.kind();
    //     if kind != Added {
    //         path_removed.push(item);
    //     }
    //     if kind != Removed {
    //         path_added.push(item);
    //     }
    //     path.push(item);
    // }

    let concat = |item: &DiffPathItem| {
        let kind = item.kind();
        let (toks, start) = if kind != Added {
            (&toks_a, item.start_removed())
        } else {
            (&toks_b, item.start_added())
        };
        let mut result = vec![];
        for buf in &toks[start..start + item.len()] {
            result.extend_from_slice(buf)
        }
        (result, kind)
    };

    let output = mk_vec(&mut path.iter().map(concat));
    let output_added = mk_vec(path_added.iter().map(concat));
    let output_removed = mk_vec(path_removed.iter().map(concat));

    let output_added = compress_path(&output_added);
    let output_removed = compress_path(&output_removed);

    let _output = mk_vec(output.iter().map(|(vec, c)| (&vec[..], c.clone())));
    let _output_added = mk_vec(output_added.iter().map(|(vec, c)| (&vec[..], c.clone())));
    let _output_removed = mk_vec(output_removed.iter().map(|(vec, c)| (&vec[..], c.clone())));

    // assert_eq!(expected, &output[..]);
    // assert_eq!(expected_added, &output_added[..]);
    // assert_eq!(expected_removed, &output_removed[..]);

    let d = expected
        .iter()
        .map(|(buf, kind)| match kind {
            Added | Removed => buf.len(),
            Keep => 0,
        })
        .fold(0, |acc, len| acc + len);

    assert_eq!(d, diff);
    assert_eq!(d, diff_r);
    assert_eq!(d, diff_bwd);
    assert_eq!(d, diff_r_bwd);
}

#[test]
fn compress_path_test() {
    let test = |expected: Vec<(Vec<u8>, DiffKind)>, input| {
        assert_eq!(expected, compress_path(&input));
    };

    test(vec![], vec![]);

    test(
        vec![(b"abc".to_vec(), Added)],
        vec![(b"abc".to_vec(), Added)],
    );
    test(
        vec![(b"abcdef".to_vec(), Added)],
        vec![(b"abc".to_vec(), Added), (b"def".to_vec(), Added)],
    );
    test(
        vec![(b"abc".to_vec(), Added), (b"def".to_vec(), Removed)],
        vec![(b"abc".to_vec(), Added), (b"def".to_vec(), Removed)],
    );

    test(
        vec![
            (b"abc".to_vec(), Added),
            (b"defghijkl".to_vec(), Removed),
            (b"xyz".to_vec(), Keep),
        ],
        vec![
            (b"abc".to_vec(), Added),
            (b"def".to_vec(), Removed),
            (b"ghi".to_vec(), Removed),
            (b"jkl".to_vec(), Removed),
            (b"xyz".to_vec(), Keep),
        ],
    );
}

#[test]
fn starts_hunk_test() {
    assert!(starts_hunk(b"@@@"));
    assert!(starts_hunk(b"\x1b[42m@@@"));
    assert!(starts_hunk(b"\x1b[42m\x1b[33m@@@"));
    assert!(!starts_hunk(b"\x1c[42m@@@"));
    assert!(!starts_hunk(b"\x1b[42m"));
    assert!(!starts_hunk(b""));
}

#[test]
fn skip_escape_code_test() {
    assert_eq!(5, skip_all_escape_code(b"\x1b[42m@@@"));
    assert_eq!(10, skip_all_escape_code(b"\x1b[42m\x1b[33m@@@"));
    assert_eq!(0, skip_all_escape_code(b"\x1b[42@@@"));
}

#[test]
fn skip_token_test() {
    assert_eq!(3, skip_token(b"abc\x1b"));
    assert_eq!(3, skip_token(b"abc"));
    assert_eq!(0, skip_token(b"\x1b"));
}

#[test]
fn diff_sequences_test_1() {
    diff_sequences_test(
        &[
            (b"ab", Removed),
            (b"c", Keep),
            (b"b", Added),
            (b"ab", Keep),
            (b"b", Removed),
            (b"a", Keep),
            (b"c", Added),
        ],
        b"abcabba",
        b"cbabac",
    )
}

#[test]
fn diff_sequences_test_2() {
    diff_sequences_test(
        &[
            (b"x", Added),
            (b"a", Keep),
            (b"x", Added),
            (b"b", Keep),
            (b"xab", Added),
            (b"c", Keep),
            (b"y", Removed),
        ],
        b"abcy",
        b"xaxbxabc",
    )
}

#[test]
fn diff_sequences_test_3() {
    diff_sequences_test(&[(b"defgh", Added), (b"abc", Removed)], b"abc", b"defgh")
}

#[test]
fn diff_sequences_test_4() {
    diff_sequences_test(
        &[(b"defg", Added), (b"abc", Removed), (b"zzz", Keep)],
        b"abczzz",
        b"defgzzz",
    )
}

#[test]
fn diff_sequences_test_5() {
    diff_sequences_test(
        &[(b"zzz", Keep), (b"defg", Added), (b"abcd", Removed)],
        b"zzzabcd",
        b"zzzefgh",
    )
}

#[test]
fn diff_sequences_test_6() {
    diff_sequences_test(&[(b"abcd", Added)], b"", b"abcd")
}

#[test]
fn diff_sequences_test_7() {
    diff_sequences_test(&[], b"", b"")
}

#[test]
fn range_equality_test() {
    let range_a = [1, 2, 3];
    let range_b = [1, 2, 3];
    let range_c = [1, 2, 4];
    assert!(range_a == range_b);
    assert!(range_a != range_c);
}

#[test]
fn aligned_test() {
    assert!(aligned(&(1, 3), &(2, 2), &(3, 1)));
    assert!(!aligned(&(1, 3), &(2, 2), &(3, 2)));
}

#[test]
fn tokenize_test() {
    fn test(expected: &[&str], buf: &[u8]) {
        let tokens = tokenize(buf);
        assert_eq!(buf.len(), tokens.iter().map(|buf| buf.len()).sum());
        for token in &tokens {
            assert!(token.len() != 0)
        }
        assert_eq!(
            mk_vec(buf.iter()),
            mk_vec(tokens.iter().flat_map(|buf| buf.iter()))
        );

        assert_eq!(expected, &to_strings(&tokens[..])[..]);
    }
    test(&[], b"");
    test(&[" "], b" ");
    test(&["a"], b"a");
    test(&["abcd", " ", "defg", " "], b"abcd defg ");
    test(&["abcd", " ", "defg"], b"abcd defg");
    test(
        &["*", "(", "abcd", ")", " ", "#", "[", "efgh", "]"],
        b"*(abcd) #[efgh]",
    );
}

#[test]
fn color_test() {
    assert_eq!(None, DiffKind::Keep.color());
}

#[test]
fn path_test() {
    fn test(expected: &[DiffPathItem], input: &[(usize, usize)]) {
        assert_eq!(expected.to_vec(), mk_vec(Diff::new(input.to_vec()).path()));
    }
    test(
        &[
            DiffPathItem(Keep, 1, 0, 0),
            DiffPathItem(Added, 2, 1, 1),
            DiffPathItem(Removed, 4, 1, 3),
        ],
        &[(0, 0), (1, 1), (1, 3), (5, 3)],
    );

    test(
        &[
            DiffPathItem(Keep, 1, 0, 0),
            DiffPathItem(Keep, 1, 1, 1),
            DiffPathItem(Added, 1, 2, 2),
            DiffPathItem(Removed, 3, 2, 3),
        ],
        &[(0, 0), (1, 1), (2, 2), (2, 3), (5, 3)],
    );
}
