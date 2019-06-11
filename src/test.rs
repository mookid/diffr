use super::*;

fn string_of_bytes(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf).into()
}

fn to_strings(buf: &[u8], tokens: &[(usize, usize)]) -> Vec<String> {
    mk_vec(
        tokens
            .iter()
            .map(|range| string_of_bytes(&buf[range.0..range.1])),
    )
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

fn dummy_tokenize<'a>(data: &'a [u8]) -> Vec<(usize, usize)> {
    let mut toks = vec![];
    for i in 0..data.len() {
        toks.push((i, i + 1));
    }
    toks
}

fn diff_sequences_test(expected: &[(&[u8], DiffKind)], seq_a: &[u8], seq_b: &[u8]) {
    let toks_a = dummy_tokenize(&seq_a);
    let toks_b = dummy_tokenize(&seq_b);
    let input = Tokens {
        added: Tokenization {
            data: &seq_b,
            tokens: &toks_b,
        },
        removed: Tokenization {
            data: &seq_a,
            tokens: &toks_a,
        },
    };
    let input_r = Tokens {
        added: Tokenization {
            data: &seq_a,
            tokens: &toks_a,
        },
        removed: Tokenization {
            data: &seq_b,
            tokens: &toks_b,
        },
    };

    let mut v = vec![];
    dbg!(&input);
    let diff = diff_sequences_simple(&input, &mut v, true);
    let diff_bwd = diff_sequences_simple(&input, &mut v, false);
    let diff_bidi = diff_sequences_bidirectional(&input, &mut v);
    let diff_r = diff_sequences_simple(&input_r, &mut v, true);
    let diff_r_bwd = diff_sequences_simple(&input_r, &mut v, false);
    let diff_r_bidi = diff_sequences_bidirectional(&input_r, &mut v);

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
        let tok = if kind != Added {
            &input.removed
        } else {
            &input.added
        };
        let mut result = vec![];
        for range in tok.tokens {
            result.extend_from_slice(&tok.data[range.0..range.1])
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
    assert_eq!(d, diff_bidi.d);
    assert_eq!(d, diff_r_bidi.d);

    for (snake, input) in &[(&diff_bidi, &input), (&diff_r_bidi, &input_r)] {
        let Snake { x0, x1, y0, y1, .. } = snake;
        assert_eq!(x0 - x1, y0 - y1);
        assert_eq!(
            mk_vec((*x0..*x1).map(|x| input.seq_a(x))),
            mk_vec((*y0..*y1).map(|y| input.seq_b(y))),
        );
    }
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
        assert_eq!(
            buf.len(),
            tokens.iter().map(|range| range.1 - range.0).sum()
        );
        for token in &tokens {
            assert!(token.0 < token.1)
        }
        assert_eq!(
            mk_vec(buf.iter()),
            mk_vec(tokens.iter().flat_map(|range| &buf[range.0..range.1]))
        );

        let foo = mk_vec(
            tokens
                .iter()
                .map(|range| &buf[range.0..range.1])
                .map(|buf| string_of_bytes(buf)),
        );

        let foo = mk_vec(foo.iter().map(|str| &**str));

        assert_eq!(&*expected, &*foo);

        // TODO
        assert_eq!(expected, &to_strings(&buf, &tokens)[..]);
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
