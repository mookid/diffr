use crate::DiffKind::*;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use termcolor::{
    Color::{Green, Red},
    ColorChoice, ColorSpec, StandardStream,
};

trait Reader {
    fn next_line(&mut self, buffer: &mut Vec<u8>) -> io::Result<usize>;
}

impl<'a> Reader for BufRead {
    fn next_line(&mut self, buffer: &mut Vec<u8>) -> io::Result<usize> {
        self.read_until(b'\n', buffer)
    }
}

fn usage() -> ! {
    eprintln!("usage:");
    let process_name = std::env::args().next();
    let process_name = match process_name {
        Some(ref pn) => &pn[..],
        None => "<0>",
    };
    eprintln!("\t{}: read from stdin", process_name);
    eprintln!(
        "\t{} --input <filename>: read input from file",
        process_name
    );
    std::process::exit(1)
}

fn mk_reader<'a>(stdin: &'a io::Stdin) -> Box<BufRead + 'a> {
    let mut args = std::env::args().skip(1);
    match args.next().as_ref().map(|x| &**x) {
        Some("--input") => {
            if let Some(ref path) = args.next() {
                let file = match File::open(path) {
                    Ok(f) => f,
                    Err(e) => panic!("error opening file {}: {}", &path, e),
                };
                let stdin = BufReader::new(file);
                Box::new(stdin)
            } else {
                usage()
            }
        }
        Some(_) => usage(),
        None => Box::new(stdin.lock()),
    }
}

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut buffer = vec![];
    let mut hunk_buffer = HunkBuffer::new();
    let mut stdin = mk_reader(&stdin);
    let mut stdout = stdout.lock();

    let mut time_computing_diff_ms = 0;
    let start = std::time::SystemTime::now();

    // echo everything before the first diff hunk
    loop {
        stdin.read_until(b'\n', &mut buffer)?;
        if buffer.is_empty() || starts_hunk(&buffer) {
            break;
        }
        write!(stdout, "{}", String::from_utf8_lossy(&buffer))?;
        buffer.clear();
    }

    // process hunks
    loop {
        stdin.read_until(b'\n', &mut buffer)?;
        if buffer.is_empty() {
            break;
        }

        match first_after_escape(&buffer) {
            Some(b'+') => hunk_buffer.push_added(&buffer),
            Some(b'-') => hunk_buffer.push_removed(&buffer),
            _ => {
                let start = std::time::SystemTime::now();
                hunk_buffer.process(&mut stdout)?;
                hunk_buffer.clear();
                write!(stdout, "{}", String::from_utf8_lossy(&buffer))?;
                time_computing_diff_ms += start.elapsed().unwrap().as_millis();
            }
        }
        // dbg!(&String::from_utf8_lossy(&buffer));
        buffer.clear();
    }

    // flush remaining hunk
    hunk_buffer.process(&mut stdout)?;
    eprintln!("hunk processing time (ms): {}", time_computing_diff_ms);
    eprintln!(
        "total processing time (ms): {}",
        start.elapsed().unwrap().as_millis()
    );
    Ok(())
}

struct HunkBuffer {
    added_lines: Vec<u8>,
    removed_lines: Vec<u8>,
}

// Scan buf looking for target, returning the index of its first
// appearance.
fn index_of<It>(it: It, target: u8) -> Option<usize>
where
    It: std::iter::Iterator<Item = u8>,
{
    // let mut it = buf.iter().enumerate();
    let mut it = it.enumerate();
    loop {
        match it.next() {
            Some((index, c)) => {
                if c == target {
                    return Some(index);
                }
            }
            None => return None,
        }
    }
}

// the number of bytes until either the next escape code or the end of
// buf
fn skip_token(buf: &[u8]) -> usize {
    index_of(buf.iter().cloned(), b'\x1b').unwrap_or_else(|| buf.len())
}

fn add_raw_line(dst: &mut Vec<u8>, line: &[u8]) {
    let mut i = 0;
    let len = line.len();
    while i < len {
        i += skip_all_escape_code(&line[i..]);
        let tok_len = skip_token(&line[i..]);
        dst.extend_from_slice(&line[i..i + tok_len]);
        i += tok_len;
    }
}

impl HunkBuffer {
    fn new() -> Self {
        HunkBuffer {
            added_lines: vec![],
            removed_lines: vec![],
        }
    }

    fn clear(&mut self) {
        self.added_lines.clear();
        self.removed_lines.clear();
    }

    fn push_added(&mut self, line: &[u8]) {
        add_raw_line(&mut self.added_lines, line)
    }

    fn push_removed(&mut self, line: &[u8]) {
        add_raw_line(&mut self.removed_lines, line)
    }

    fn removed_lines(&self) -> &[u8] {
        &self.removed_lines
    }

    fn added_lines(&self) -> &[u8] {
        &self.added_lines
    }

    fn process<Stream>(&self, out: &mut Stream) -> io::Result<()>
    where
        Stream: termcolor::WriteColor,
    {
        let removed_words = tokenize(self.removed_lines());
        let added_words = tokenize(self.added_lines());
        // dbg!(added_words
        //      .map(|buf| String::from_utf8_lossy(buf))
        //      .collect::<Vec<_>>());

        let _diff = diff_sequences(&removed_words, &added_words);

        output(self.removed_lines(), Some(Red), out)?;
        output(self.added_lines(), Some(Green), out)?;
        Ok(())
    }
}

fn tokenize(src: &[u8]) -> Vec<&[u8]> {
    let mut tokens = vec![];
    let mut lo = 0;
    let mut it = src.iter().clone().enumerate();
    while let Some((hi, b)) = it.next() {
        if !is_alphanum(*b) {
            if lo < hi {
                tokens.push(&src[lo..hi]);
            }
            tokens.push(&src[hi..hi + 1]);
            lo = hi + 1
        }
    }
    if lo < src.len() {
        tokens.push(&src[lo..src.len()]);
    }
    tokens
}

type Point = (usize, usize);

#[derive(Clone, Debug)]
struct Diff {
    points: Vec<Point>,
}

fn last<T>(slice: &[T]) -> Option<&T> {
    slice.iter().next_back()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DiffKind {
    Keep,
    Added,
    Removed,
}

impl DiffKind {
    fn color(&self) -> Option<termcolor::Color> {
        match self {
            Keep => None,
            Added => Some(Green),
            Removed => Some(Red),
        }
    }
}

// The kind of diff and the size of the piece of diff
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DiffPathItem(DiffKind, usize, usize, usize);

impl DiffPathItem {
    fn kind(&self) -> DiffKind {
        self.0
    }

    fn len(&self) -> usize {
        self.1
    }

    fn start_removed(&self) -> usize {
        self.2
    }

    fn start_added(&self) -> usize {
        self.3
    }
}

struct Path<'a>(std::slice::Windows<'a, Point>);

impl<'a> Path<'a> {
    fn new(diff: &'a Diff) -> Self {
        Path(diff.points.windows(2))
    }
}

impl<'a> Iterator for Path<'a> {
    type Item = DiffPathItem;
    fn next(self: &mut Self) -> Option<Self::Item> {
        self.0.next().map(|pair| {
            let (x0, y0) = pair[0];
            let (x1, y1) = pair[1];
            let (kind, len) = match (x0 != x1, y0 != y1) {
                (true, true) => (Keep, x1 - x0),
                (false, true) => (Added, y1 - y0),
                (true, false) => (Removed, x1 - x0),
                (false, false) => panic!("invariant error: duplicate point"),
            };
            assert!(0 < len);
            DiffPathItem(kind, len as usize, x0, y0)
        })
    }
}

impl Diff {
    fn new(points: Vec<(usize, usize)>) -> Diff {
        Diff { points }
    }

    fn empty() -> Diff {
        Diff::new(vec![])
    }

    fn last_point(&self) -> &(usize, usize) {
        last(&self.points).unwrap_or(&(0, 0))
    }

    fn push(&mut self, x: usize, y: usize) {
        let mut it = self.points.iter_mut();
        let z2 = (x, y);
        let z1 = it.next_back();
        let z0 = it.next_back();
        match (z0, z1) {
            (Some(ref z0), Some(ref mut z1)) if aligned(z0, z1, &z2) => **z1 = z2,
            _ => self.points.push(z2),
        }
    }

    fn path(&self) -> Path {
        return Path::new(&self);
    }
}

fn aligned(z0: &Point, z1: &Point, z2: &Point) -> bool {
    fn vector(z0: &Point, z1: &Point) -> (i64, i64) {
        (z1.0 as i64 - z0.0 as i64, z1.1 as i64 - z0.1 as i64)
    }
    let v01 = vector(z0, z1);
    let v02 = vector(z0, z2);
    v01.0 * v02.1 == v01.1 * v02.0
}

fn diff_sequences(seq_a: &[&[u8]], seq_b: &[&[u8]]) -> Diff {
    let n = seq_a.len();
    let m = seq_b.len();
    let max = (n + m) as i64;
    let mut v = vec![Diff::empty(); max as usize * 2 + 1];
    let convert_index = |index| {
        assert!(-max <= index);
        assert!(index <= max);
        (index + max) as usize
    };
    for d in 0..max {
        for k in (-d..=d).step_by(2) {
            assert_eq!((k - d) % 2, 0);
            let mut path = {
                let vkp1: &Diff = &v[convert_index(k + 1)];
                if k == -d {
                    vkp1.clone()
                } else {
                    let vkm1 = &v[convert_index(k - 1)];
                    if k == d && vkm1.last_point().0 < vkp1.last_point().0 {
                        vkp1.clone()
                    } else {
                        let mut res = vkm1.clone();
                        let last_point = res.last_point();
                        res.push(last_point.0 + 1, last_point.1);
                        res
                    }
                }
            };
            let mut x = path.last_point().0;
            let mut y = (x as i64 - k) as usize;
            path.push(x, y);
            while x < n && y < m && seq_a[x] == seq_b[y] {
                x += 1;
                y += 1;
            }
            path.push(x, y);
            v[convert_index(k)] = path;
            if n <= x && m <= y {
                return v[convert_index(k)].clone();
            }
        }
    }
    Diff::new(vec![(0, 0), (0, m), (n, m)])
}

fn is_alphanum(b: u8) -> bool {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => true,
        _ => false,
    }
}

fn output<Stream>(buf: &[u8], color: Option<termcolor::Color>, out: &mut Stream) -> io::Result<()>
where
    Stream: termcolor::WriteColor,
{
    out.set_color(ColorSpec::new().set_fg(color))?;
    let whole_line = !buf.is_empty() && buf[buf.len() - 1] == b'\n';
    let buf = if whole_line {
        &buf[..buf.len() - 1]
    } else {
        buf
    };
    write!(out, "{}", String::from_utf8_lossy(buf))?;
    out.flush()?;
    out.reset()?;
    if whole_line {
        write!(out, "\n")?;
    }
    Ok(())
}

// Detect if the line marks the beginning of a hunk.
fn starts_hunk(buf: &[u8]) -> bool {
    first_after_escape(buf) == Some(b'@')
}

// Detect if the line starts with exactly one of the given bytes, after escape
// code bytes.
fn first_after_escape(buf: &[u8]) -> Option<u8> {
    let nbytes = skip_all_escape_code(&buf);
    buf.iter().skip(nbytes).cloned().next()
}

// Returns the number of bytes of escape code.
fn skip_all_escape_code(buf: &[u8]) -> usize {
    // Skip one sequence
    fn skip_escape_code(buf: &[u8]) -> Option<usize> {
        let mut it = buf.iter().cloned();
        if it.next()? == b'\x1b' && it.next()? == b'[' {
            // "\x1b[" + sequence body + "m" => 3 additional bytes
            Some(index_of(it, b'm')? + 3)
        } else {
            None
        }
    }
    let mut buf = buf;
    let mut sum = 0;
    while let Some(nbytes) = skip_escape_code(&buf) {
        buf = &buf[nbytes..];
        sum += nbytes
    }
    sum
}

#[cfg(test)]
fn string_of_bytes(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf).into()
}

#[cfg(test)]
fn to_strings(buf: &[&[u8]]) -> Vec<String> {
    mk_vec(buf.iter().map(|buf| string_of_bytes(buf)))
}

#[cfg(test)]
fn mk_vec<It, T>(it: It) -> Vec<T>
where
    It: Iterator<Item = T>,
{
    it.collect()
}

#[cfg(test)]
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

#[cfg(test)]
fn diff_sequences_test_edit(
    expected: &[(&[u8], DiffKind)],
    expected_added: &[(&[u8], DiffKind)],
    expected_removed: &[(&[u8], DiffKind)],
    seq_a: &[u8],
    seq_b: &[u8],
) {
    fn mk_tokens(buf: &[u8]) -> Vec<&[u8]> {
        (0..buf.len()).map(|i| &buf[i..i + 1]).collect()
    };

    let toks_a = mk_tokens(seq_a);
    let toks_b = mk_tokens(seq_b);

    let diff = &diff_sequences(&toks_a, &toks_b);
    let mut path = vec![];
    let mut path_added = vec![];
    let mut path_removed = vec![];

    for item in diff.path() {
        let kind = item.kind();
        if kind != Added {
            path_removed.push(item);
        }
        if kind != Removed {
            path_added.push(item);
        }
        path.push(item);
    }

    let concat = |item: DiffPathItem| {
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

    let output = mk_vec(&mut path.iter().cloned().map(concat));
    let output_added = mk_vec(path_added.iter().cloned().map(concat));
    let output_removed = mk_vec(path_removed.iter().cloned().map(concat));

    let output_added = compress_path(&output_added);
    let output_removed = compress_path(&output_removed);

    let output = mk_vec(output.iter().map(|(vec, c)| (&vec[..], c.clone())));
    let output_added = mk_vec(output_added.iter().map(|(vec, c)| (&vec[..], c.clone())));
    let output_removed = mk_vec(output_removed.iter().map(|(vec, c)| (&vec[..], c.clone())));

    assert_eq!(expected, &output[..]);
    assert_eq!(expected_added, &output_added[..]);
    assert_eq!(expected_removed, &output_removed[..]);
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
    diff_sequences_test_edit(
        &[
            (b"cb", Added),
            (b"ab", Keep),
            (b"a", Added),
            (b"c", Keep),
            (b"abba", Removed),
        ],
        &[(b"cb", Added), (b"ab", Keep), (b"a", Added), (b"c", Keep)],
        &[(b"abc", Keep), (b"abba", Removed)],
        b"abcabba",
        b"cbabac",
    )
}

#[test]
fn diff_sequences_test_2() {
    diff_sequences_test_edit(
        &[
            (b"x", Added),
            (b"a", Keep),
            (b"x", Added),
            (b"b", Keep),
            (b"xab", Added),
            (b"c", Keep),
            (b"y", Removed),
        ],
        &[
            (b"x", Added),
            (b"a", Keep),
            (b"x", Added),
            (b"b", Keep),
            (b"xab", Added),
            (b"c", Keep),
        ],
        &[(b"abc", Keep), (b"y", Removed)],
        b"abcy",
        b"xaxbxabc",
    )
}

#[test]
fn diff_sequences_test_3() {
    diff_sequences_test_edit(
        &[(b"defgh", Added), (b"abc", Removed)],
        &[(b"defgh", Added)],
        &[(b"abc", Removed)],
        b"abc",
        b"defgh",
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
