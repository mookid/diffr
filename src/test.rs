use super::*;
use DiffKind::*;

// The kind of diff and the size of the piece of diff
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DiffPathItem(DiffKind, usize, usize, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiffKind {
    Keep,
    Added,
    Removed,
}

impl DiffPathItem {
    fn kind(&self) -> DiffKind {
        self.0
    }
}

fn string_of_bytes(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf).into()
}

fn to_strings<It>(buf: &[u8], tokens: It) -> Vec<String>
where
    It: Iterator<Item = (usize, usize)>,
{
    mk_vec(tokens.map(|range| string_of_bytes(&buf[range.0..range.1])))
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
    let mut result = vec![];
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

fn dummy_tokenize<'a>(data: &'a [u8]) -> Vec<HashedSliceRef> {
    let mut toks = vec![];
    for i in 0..data.len() {
        toks.push((i, i + 1, hash_slice(&data[i..i + 1])));
    }
    toks
}

fn really_tokenize<'a>(data: &'a [u8]) -> Vec<HashedSliceRef> {
    let mut toks = vec![];
    tokenize(0, &mut toks, data);
    toks
}

fn diff_sequences_test(expected: &[(&[u8], DiffKind)], seq_a: &[u8], seq_b: &[u8]) {
    diff_sequences_test_aux(expected, seq_a, seq_b, dummy_tokenize)
}

fn diff_sequences_test_tokenized(expected: &[(&[u8], DiffKind)], seq_a: &[u8], seq_b: &[u8]) {
    diff_sequences_test_aux(expected, seq_a, seq_b, really_tokenize)
}

fn diff_sequences_test_aux<Tok>(
    expected: &[(&[u8], DiffKind)],
    seq_a: &[u8],
    seq_b: &[u8],
    tok: Tok,
) where
    Tok: Fn(&[u8]) -> Vec<HashedSliceRef>,
{
    let toks_a = tok(&seq_a);
    let toks_b = tok(&seq_b);
    let input = Tokens {
        added: Tokenization::new(&seq_b, &toks_b),
        removed: Tokenization::new(&seq_a, &toks_a),
    };
    let input_r = Tokens {
        added: Tokenization::new(&seq_a, &toks_a),
        removed: Tokenization::new(&seq_b, &toks_b),
    };

    let mut v = vec![];
    let result = diff_sequences_simple_forward(&input, &mut v);
    let result_bwd = diff_sequences_simple_backward(&input, &mut v);
    let result_bidi = diff_sequences_bidirectional(&input, &mut v);
    let result_r = diff_sequences_simple(&input_r, &mut v, true);
    let result_r_bwd = diff_sequences_simple(&input_r, &mut v, false);
    let result_r_bidi = diff_sequences_bidirectional(&input_r, &mut v);

    let mut result_complete = vec![];
    diff(&input, &mut v, &mut result_complete);
    let mut result_r_complete = vec![];
    diff(&input_r, &mut v, &mut result_r_complete);

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
            Added | Removed => tok(buf).len(),
            Keep => 0,
        })
        .fold(0, |acc, len| acc + len);

    assert_eq!(d, result);
    assert_eq!(d, result_r);
    assert_eq!(d, result_bwd);
    assert_eq!(d, result_r_bwd);
    assert_eq!(d, result_bidi);
    assert_eq!(d, result_r_bidi);

    for complete in &[&result_complete, &result_r_complete] {
        let all_snakes = complete.iter().fold(0, |acc, s| acc + s.len);

        let d_calc = input.n() + input.m() - 2 * to_usize(all_snakes);
        assert_eq!(d, d_calc);
    }
    // construct edit script
    let mut x0 = 0;
    let mut y0 = 0;
    let mut script = vec![];
    for snake in result_complete {
        let Snake {
            x0: x, y0: y, len, ..
        } = snake;

        if x0 != x {
            assert!(x0 < x);
            let lo = &input.removed.seq_index(x0);
            let hi = &input.removed.seq_index(x - 1);
            script.push((input.removed.data[lo.0..hi.1].to_vec(), Removed));
        }
        if y0 != y {
            assert!(y0 < y);
            let lo = &input.added.seq_index(y0);
            let hi = &input.added.seq_index(y - 1);
            script.push((input.added.data[lo.0..hi.1].to_vec(), Added));
        }

        let mut added = vec![];
        let mut removed = vec![];
        for i in 0..len {
            let r = &input.removed.seq_index(x + i);
            removed.extend_from_slice(&input.removed.data[r.0..r.1]);
            let r = &input.added.seq_index(y + i);
            added.extend_from_slice(&input.added.data[r.0..r.1]);
        }

        assert_eq!(added, removed, "{:?}", snake);
        script.push((added.to_vec(), Keep));

        x0 = x + len;
        y0 = y + len;
    }

    let x = input.removed.nb_tokens();
    let x0 = to_usize(x0);
    if x0 != x {
        assert!(x0 < x);
        script.push((input.removed.data[x0..x].to_vec(), Removed));
    }
    let y = input.added.nb_tokens();
    let y0 = to_usize(y0);
    if y0 != y {
        assert!(y0 < y);
        script.push((input.added.data[y0..y].to_vec(), Added));
    }

    assert_eq!(
        &*mk_vec(expected.iter().map(|p| (string_of_bytes(p.0), p.1))),
        &*mk_vec(script.iter().map(|p| (string_of_bytes(&p.0), p.1))),
    );
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
fn skip_escape_code_test() {
    assert_eq!(5, skip_all_escape_code(b"\x1b[42m@@@"));
    assert_eq!(10, skip_all_escape_code(b"\x1b[42m\x1b[33m@@@"));
    assert_eq!(0, skip_all_escape_code(b"\x1b[42@@@"));
}

#[test]
fn skip_token_test() {
    assert_eq!(4, skip_token(b"abc\x1b"));
    assert_eq!(3, skip_token(b"abc\x1b["));
    assert_eq!(3, skip_token(b"abc"));
    assert_eq!(1, skip_token(b"\x1b"));
    assert_eq!(0, skip_token(b""));
}

#[test]
fn diff_sequences_test_1() {
    diff_sequences_test(
        &[
            (b"a", Removed),
            (b"c", Added),
            (b"b", Keep),
            (b"c", Removed),
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
        &[(b"xaxbx", Added), (b"abc", Keep), (b"y", Removed)],
        b"abcy",
        b"xaxbxabc",
    )
}

#[test]
fn diff_sequences_test_3() {
    diff_sequences_test(&[(b"abc", Removed), (b"defgh", Added)], b"abc", b"defgh")
}

#[test]
fn diff_sequences_test_4() {
    diff_sequences_test(
        &[(b"abc", Removed), (b"defg", Added), (b"zzz", Keep)],
        b"abczzz",
        b"defgzzz",
    )
}

#[test]
fn diff_sequences_test_5() {
    diff_sequences_test(
        &[(b"zzz", Keep), (b"abcd", Removed), (b"efgh", Added)],
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
fn diff_sequences_test_8() {
    // This tests the recursion in diff
    diff_sequences_test(
        &[
            (b"a", Removed),
            (b"c", Added),
            (b"b", Keep),
            (b"c", Removed),
            (b"a", Keep),
            (b"b", Removed),
            (b"ba", Keep),
            (b"a", Removed),
            (b"cc", Added),
            (b"b", Keep),
            (b"c", Removed),
            (b"ab", Keep),
            (b"b", Removed),
            (b"a", Keep),
            (b"a", Removed),
            (b"cc", Added),
            (b"b", Keep),
            (b"c", Removed),
            // this is weird; the 2 next should be combined?
            (b"a", Keep),
            (b"b", Keep),
            (b"b", Removed),
            (b"a", Keep),
            (b"c", Added),
        ],
        b"abcabbaabcabbaabcabba",
        b"cbabaccbabaccbabac",
    )
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
fn tokenize_test() {
    fn test(expected: &[&str], buf: &[u8]) {
        let mut tokens = vec![];
        tokenize(0, &mut tokens, buf);
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
        let tokens = tokens.iter().map(|(lo, hi, _)| (*lo, *hi));
        assert_eq!(expected, &to_strings(&buf, tokens)[..]);
    }
    test(&[], b"");
    test(&[" "], b" ");
    test(&["a"], b"a");
    test(&["abcd", " ", "defg", " "], b"abcd defg ");
    test(&["abcd", " ", "defg"], b"abcd defg");
    test(&["abcd", "    ", "defg"], b"abcd    defg");
    test(&["abcd", "\t    ", "defg"], b"abcd\t    defg");
    test(
        &["*", "(", "abcd", ")", " ", "#", "[", "efgh", "]"],
        b"*(abcd) #[efgh]",
    );
}

#[test]
fn find_splitting_point_test() {
    fn test(expected: isize, seq_a: &[u8], seq_b: &[u8]) {
        let toks_a = dummy_tokenize(&seq_a);
        let toks_b = dummy_tokenize(&seq_b);
        let input = Tokens {
            added: Tokenization::new(&seq_b, &toks_b),
            removed: Tokenization::new(&seq_a, &toks_a),
        };

        assert_eq!(expected, find_splitting_point(&input).sp);
        for i in 0..expected {
            assert_eq!(input.removed.seq(i), input.added.seq(i));
        }
        for i in expected..to_isize(input.removed.nb_tokens()) {
            assert_eq!(input.removed.seq(i), input.added.seq(i + 1));
        }
    }

    test(0, b"abc", b"zabc");
    test(1, b"abc", b"azbc");
    test(2, b"abc", b"abzc");
    test(3, b"abc", b"abcz");
}

fn get_lcs(seq_a: &[u8], seq_b: &[u8]) -> Vec<Vec<u8>> {
    fn subsequences(seq_a: &[u8]) -> Vec<Vec<u8>> {
        let res: Vec<Vec<u8>> = {
            if seq_a.len() == 0 {
                vec![vec![]]
            } else if seq_a.len() == 1 {
                vec![vec![], seq_a.to_owned()]
            } else {
                let (seq_a1, seq_a2) = seq_a.split_at(seq_a.len() / 2);
                let mut res = vec![];
                for part1 in subsequences(&seq_a1) {
                    for part2 in subsequences(seq_a2) {
                        let mut seq = vec![];
                        seq.extend_from_slice(&part1);
                        seq.extend_from_slice(&part2);
                        res.push(seq);
                    }
                }
                res
            }
        };
        assert_eq!(res.len(), 1 << seq_a.len());
        res
    }
    fn is_subseq(subseq: &[u8], seq: &[u8]) -> bool {
        if subseq.len() == 0 {
            true
        } else {
            let target = subseq[0];
            for i in 0..seq.len() {
                if seq[i] == target {
                    return is_subseq(&subseq[1..], &seq[i + 1..]);
                }
            }
            false
        }
    }

    let mut bests = vec![];
    let mut best_len = 0;
    for subseq in subsequences(seq_a) {
        if subseq.len() < best_len || !is_subseq(&*subseq, seq_b) {
            continue;
        }
        if best_len < subseq.len() {
            bests.clear();
            best_len = subseq.len();
        }
        if best_len <= subseq.len() {
            bests.push(subseq)
        }
    }
    bests
}

#[test]
fn test_get_lcs() {
    dbg!(get_lcs(b"abcd", b"cdef"));
    let expected: &[u8] = b"cd";
    assert_eq!(
        expected,
        &**get_lcs(b"abcd", b"cdef").iter().next().unwrap()
    )
}

#[test]
fn test_lcs_random() {
    fn test_lcs(seq_a: &[u8], seq_b: &[u8]) {
        let toks_a = dummy_tokenize(&seq_a);
        let toks_b = dummy_tokenize(&seq_b);
        let input = Tokens {
            added: Tokenization::new(&seq_b, &toks_b),
            removed: Tokenization::new(&seq_a, &toks_a),
        };
        let mut v = vec![];
        let mut dst = vec![];
        diff(&input, &mut v, &mut dst);

        // check that dst content defines a subsequence of seq_a and seq_b
        let mut diff_lcs = vec![];
        for Snake { x0, y0, len, .. } in dst {
            let part_seq_a = (x0..x0 + len)
                .flat_map(|idx| input.removed.seq(idx).data.iter().cloned())
                .collect::<Vec<_>>();
            let part_seq_b = (y0..y0 + len)
                .flat_map(|idx| input.added.seq(idx).data.iter().cloned())
                .collect::<Vec<_>>();
            assert_eq!(&*part_seq_a, &*part_seq_b);
            diff_lcs.extend_from_slice(&*part_seq_a);
        }

        // bruteforce check that it is the longest
        assert!(get_lcs(seq_a, seq_b)
            .iter()
            .filter(|seq| **seq == diff_lcs)
            .next()
            .is_some());
    }

    let len_a = 6;
    let len_b = 6;
    let nletters = 3_u8;
    let mut seq_a = vec![b'1'; len_a];
    let mut seq_b = vec![b'1'; len_b];
    for i in 0..len_a {
        for j in 0..len_b {
            for la in 0..nletters {
                for lb in 0..nletters {
                    seq_a[i] = la;
                    seq_b[j] = lb;
                    test_lcs(&seq_a, &seq_b);
                }
            }
        }
    }
}

#[should_panic]
#[test]
fn to_usize_checked_negative_test() {
    to_usize(-1_isize);
}

#[test]
fn split_lines_test() {
    let input: &[u8] = b"abcd\nefgh\nij";
    let split = LineSplit {
        data: input.to_vec(),
        line_lens: vec![5, 5, 2],
    };
    check_split(input, &split)
}

#[test]
fn split_lines_append_test() {
    let input: &[u8] = b"abcd\nefgh\nij";
    let mut split = LineSplit::default();
    split.append_line(&input[..3]);
    split.append_line(&input[3..6]);
    split.append_line(&input[6..]);
    check_split(input, &split)
}

fn check_split(input: &[u8], split: &LineSplit) {
    assert_eq!(
        input,
        &*split.iter().fold(vec![], |acc, (lo, hi)| {
            let mut acc = acc.clone();
            acc.extend_from_slice(&input[lo..hi]);
            acc
        })
    );
}

#[test]
fn issue15() {
    diff_sequences_test_tokenized(
        &[
            (b"+      ", Added),
            (b"-", Keep),
            (b"    -", Removed),
            (b"01234;\r\n", Keep),
            (b"+      ", Added),
            (b"-", Keep),
            (b"    ", Removed),
            (b"-", Keep),
            (b"-", Removed),
            (b"abc;\r\n", Keep),
            (b"-    ", Removed),
            (b"+      ", Added),
            (b"--", Keep),
            (b"def;\r\n", Keep),
            (b"-    ", Removed),
            (b"+      ", Added),
            (b"--jkl;\r\n", Keep),
            (b"+      ", Added),
            (b"-", Keep),
            (b"    ", Removed),
            (b"-", Keep),
            (b"-", Removed),
            (b"poi;\r\n", Keep),
        ],
        b"-    -01234;\r\n-    --abc;\r\n-    --def;\r\n-    --jkl;\r\n-    --poi;\r\n",
        b"+      -01234;\r\n+      --abc;\r\n+      --def;\r\n+      --jkl;\r\n+      --poi;\r\n",
    )
}

#[test]
fn issue15_2() {
    diff_sequences_test_tokenized(
        &[
            (b"-", Removed),
            (b"+", Added),
            (b"        --include \'+ */\'", Keep),
            (b" ", Added),
            (b"\r\n", Keep),
        ],
        b"-        --include '+ */'\r\n",
        b"+        --include '+ */' \r\n",
    )
}
