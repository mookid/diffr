use std::io::{self, BufRead, Write};
use termcolor::{
    Color::{Green, Red},
    ColorChoice, ColorSpec, StandardStream,
};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut buffer = vec![];
    let mut hunk_buffer = HunkBuffer::new();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();

    // echo everything before the first diff hunk
    loop {
        stdin.read_until(b'\n', &mut buffer)?;
        if buffer.is_empty() || starts_hunk(&buffer) {
            break;
        }
        // dbg!(&String::from_utf8_lossy(&buffer));
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
                hunk_buffer.process(&mut stdout)?;
                hunk_buffer.clear();
                write!(stdout, "{}", String::from_utf8_lossy(&buffer))?;
            }
        }
        // dbg!(&String::from_utf8_lossy(&buffer));
        buffer.clear();
    }

    // flush remaining hunk
    hunk_buffer.process(&mut stdout)?;
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
        fn tokenize(src: &[u8]) -> Vec<&[u8]> {
            src.split(is_whitespace).collect()
        }

        let removed_words = tokenize(self.removed_lines());
        let added_words = tokenize(self.added_lines());
        // dbg!(added_words
        //      .map(|buf| String::from_utf8_lossy(buf))
        //      .collect::<Vec<_>>());

        let _edit_scripts_len = diff_sequences(&removed_words, &added_words);

        output(self.removed_lines(), Red, out)?;
        output(self.added_lines(), Green, out)?;
        Ok(())
    }
}

// struct Diff {
//     removed: Vec<u8>,
//     added: Vec<u8>,
// }

type Diff = usize;

fn diff_sequences(seq_a: &[&[u8]], seq_b: &[&[u8]]) -> Diff {
    let n = seq_a.len();
    let m = seq_b.len();
    let max = (n + m) as i64;
    let mut v = vec![0; max as usize * 2 + 1];
    let convert_index = |index| {
        assert!(-max <= index);
        assert!(index <= max);
        (index + max) as usize
    };
    for d in 0..max {
        for k in -d..=d {
            if (k - d) % 2 != 0 {
                continue;
            }
            let mut x = {
                let vkm1 = v[convert_index(k - 1)];
                let vkp1 = v[convert_index(k + 1)];
                if k == -d || k == d && vkm1 < vkp1 {
                    vkp1
                } else {
                    vkm1 + 1
                }
            };
            let mut y = (x as i64 - k) as usize;
            while x < n && y < m && seq_a[x] == seq_b[y] {
                x += 1;
                y += 1;
            }
            v[convert_index(k)] = x;
            if n <= x && m <= y {
                return d as usize;
            }
        }
    }
    max as usize
}

fn is_whitespace(b: &u8) -> bool {
    match *b {
        b' ' | b'\n' | b'\t' => true,
        _ => false,
    }
}

fn output<Stream>(buf: &[u8], color: termcolor::Color, out: &mut Stream) -> io::Result<()>
where
    Stream: termcolor::WriteColor,
{
    out.set_color(ColorSpec::new().set_fg(Some(color)))?;
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

#[cfg(test)]
fn diff_sequences_test_edit_script_length(expected_esl: usize, seq_a: &[u8], seq_b: &[u8]) {
    fn mk_tokens(buf : &[u8]) -> Vec<&[u8]> {
        (0..buf.len()-1).map(|i| &buf[i..i+1]).collect()
    };
    assert_eq!(expected_esl, diff_sequences(&mk_tokens(seq_a), &mk_tokens(seq_b)))
}

#[test]
fn diff_sequences_test_1() {
    diff_sequences_test_edit_script_length(5, b"abcabba", b"cbabac")
}

#[test]
fn diff_sequences_test_2() {
    diff_sequences_test_edit_script_length(6, b"abcy", b"xaxbxabc")
}

#[test]
fn range_equality_test() {
    let range_a = [1,2,3];
    let range_b = [1,2,3];
    let range_c = [1,2,4];
    assert!(range_a == range_b);
    assert!(range_a != range_c);
}
