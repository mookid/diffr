use crate::AnsiTextFormatting::*;
use std::io::{self, BufRead, Write};
use termcolor::{
    Color::{Green, Red},
    ColorChoice, ColorSpec, StandardStream, WriteColor,
};

type Outstream = termcolor::StandardStream;

fn main() -> io::Result<()> {
    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    let mut buffer = vec![];
    let mut hunk_buffer = HunkBuffer::new();
    let stdin = io::stdin();
    let mut handle = stdin.lock();

    // echo everything before the first diff hunk
    loop {
        handle.read_until(b'\n', &mut buffer)?;
        if buffer.is_empty() || starts_hunk(&buffer) {
            break;
        }
        // dbg!(&String::from_utf8_lossy(&buffer));
        write!(stdout, "{}", String::from_utf8_lossy(&buffer));
        buffer.clear();
    }

    // process hunks
    loop {
        handle.read_until(b'\n', &mut buffer)?;
        if buffer.is_empty() {
            break;
        }

        match first_after_escape(&buffer) {
            Some(b'+') => hunk_buffer.push_added(&buffer),
            Some(b'-') => hunk_buffer.push_removed(&buffer),
            _ => {
                hunk_buffer.process(&mut stdout)?;
                hunk_buffer.clear();
                write!(stdout, "{}", String::from_utf8_lossy(&buffer));
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
fn index_of<'a, It>(it: It, target: u8) -> Option<usize>
where
    It: std::iter::Iterator<Item = &'a u8>,
{
    // let mut it = buf.iter().enumerate();
    let mut it = it.enumerate();
    loop {
        match it.next() {
            Some((index, c)) => {
                if *c == target {
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
    index_of(buf.iter(), b'\x1b').unwrap_or_else(|| buf.len())
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

    fn process(&self, out: &mut Outstream) -> io::Result<()> {
        output(self.removed_lines(), Red, out)?;
        output(self.added_lines(), Green, out)?;
        Ok(())
    }
}

fn output(buf: &[u8], color: termcolor::Color, out: &mut Outstream) -> io::Result<()> {
    out.set_color(ColorSpec::new().set_fg(Some(color)))?;
    let whole_line = !buf.is_empty() && buf[buf.len() - 1] == b'\n';
    let buf = if whole_line {
        &buf[..buf.len() - 1]
    } else {
        buf
    };
    write!(out, "{}", String::from_utf8_lossy(buf));
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

fn is_ansi_text_attribute(current: u8) -> Option<AnsiTextFormatting> {
    let result = match current {
        b'0' => Some(NormalDisplay),
        b'1' => Some(Bold),
        b'4' => Some(Underline),
        b'5' => Some(Blink),
        b'7' => Some(ReverseVideo),
        b'8' => Some(Invisible),
        _ => None,
    };
    dbg!((current, result));
    result
}

fn is_ansi_color_attribute(previous: u8, current: u8) -> Option<AnsiTextFormatting> {
    let result = match (previous, current) {
        // foreground colors
        (b'3', b'0') => Some(ForegroundBlack),
        (b'3', b'1') => Some(ForegroundRed),
        (b'3', b'2') => Some(ForegroundGreen),
        (b'3', b'3') => Some(ForegroundYellow),
        (b'3', b'4') => Some(ForegroundBlue),
        (b'3', b'5') => Some(ForegroundMagenta),
        (b'3', b'6') => Some(ForegroundCyan),
        (b'3', b'7') => Some(ForegroundWhite),
        // background colors
        (b'4', b'0') => Some(BackgroundBlack),
        (b'4', b'1') => Some(BackgroundRed),
        (b'4', b'2') => Some(BackgroundGreen),
        (b'4', b'3') => Some(BackgroundYellow),
        (b'4', b'4') => Some(BackgroundBlue),
        (b'4', b'5') => Some(BackgroundMagenta),
        (b'4', b'6') => Some(BackgroundCyan),
        (b'4', b'7') => Some(BackgroundWhite),
        _ => None,
    };
    dbg!((previous, current, result));
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AnsiTextFormatting {
    // attributes
    NormalDisplay,
    Bold,
    Underline,
    Blink,
    ReverseVideo,
    Invisible,
    // foreground colors
    ForegroundBlack,
    ForegroundRed,
    ForegroundGreen,
    ForegroundYellow,
    ForegroundBlue,
    ForegroundMagenta,
    ForegroundCyan,
    ForegroundWhite,
    // background colors
    BackgroundBlack,
    BackgroundRed,
    BackgroundGreen,
    BackgroundYellow,
    BackgroundBlue,
    BackgroundMagenta,
    BackgroundCyan,
    BackgroundWhite,
}

// Returns the number of bytes of escape code.
fn skip_all_escape_code(buf: &[u8]) -> usize {
    // Skip one sequence
    fn skip_escape_code(buf: &[u8]) -> Option<usize> {
        let mut it = buf.iter();
        if *it.next()? == b'\x1b' && *it.next()? == b'[' {
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

fn parse_escape_code_prefix(bytes: &[u8]) -> Option<Vec<AnsiTextFormatting>> {
    let it = &mut bytes.iter();
    let mut result = vec![];
    if *it.next()? == b'\x1b' && *it.next()? == b'[' {
        let mut previous = None;
        let mut completed_sequence = None;
        loop {
            match previous {
                None => previous = Some(*it.next()?),
                Some(b'm') => {
                    if let Some(value) = completed_sequence {
                        result.push(value);
                    }
                    return Some(result);
                }
                Some(b';') => {
                    // if a sequence is completed, register it
                    if let Some(value) = completed_sequence {
                        result.push(value);
                        completed_sequence = None
                    }
                    // if multiple ';' in a row, clear
                    previous = Some(*it.next()?);
                    if previous == Some(b';') {
                        result.clear();
                    }
                }
                Some(prevbyte) => {
                    let currbyte = *it.next()?;
                    dbg!((prevbyte, currbyte));
                    match currbyte {
                        b'm' => match completed_sequence {
                            None => {
                                let value = is_ansi_text_attribute(prevbyte)?;
                                result.push(value);
                                return Some(result);
                            }
                            Some(_) => return None,
                        },
                        b';' => match completed_sequence {
                            None => {
                                let value = is_ansi_text_attribute(prevbyte)?;
                                completed_sequence = Some(value);
                                previous = Some(b';')
                            }
                            Some(_) => return None,
                        },
                        _ => match completed_sequence {
                            None => {
                                let value = is_ansi_color_attribute(prevbyte, currbyte)?;
                                completed_sequence = Some(value);
                                previous = None
                            }
                            Some(_) => return None,
                        },
                    }
                }
            }
        }
    }
    None
}

#[test]
fn parse_color_escape_code_prefix_invalid() {
    assert_eq!(None, parse_escape_code_prefix(b"\x1c[32m"));
    assert_eq!(None, parse_escape_code_prefix(b"\x1b\\32m"));
    assert_eq!(None, parse_escape_code_prefix(b"\x1b[32"));
    assert_eq!(None, parse_escape_code_prefix(b"\x1b32m"));
    assert_eq!(None, parse_escape_code_prefix(b"\x1b[3233m"));
    assert_eq!(None, parse_escape_code_prefix(b"\x1b[32;433m"));
}

#[test]
fn parse_color_escape_code_prefix_simple_empty() {
    assert_eq!(Some(vec![]), parse_escape_code_prefix(b"\x1b[m"));
    assert_eq!(Some(vec![]), parse_escape_code_prefix(b"\x1b[;m"));
    assert_eq!(Some(vec![]), parse_escape_code_prefix(b"\x1b[;;m"));
}

#[test]
fn parse_color_escape_code_prefix_simple_cases() {
    assert_eq!(
        Some(vec![ForegroundGreen]),
        parse_escape_code_prefix(b"\x1b[32m")
    );
    assert_eq!(
        Some(vec![ForegroundRed]),
        parse_escape_code_prefix(b"\x1b[31m")
    );
    assert_eq!(Some(vec![Underline]), parse_escape_code_prefix(b"\x1b[4m"));
}

#[test]
fn parse_color_escape_code_reset_on_double_59() {
    assert_eq!(
        Some(vec![ForegroundRed, BackgroundGreen]),
        parse_escape_code_prefix(b"\x1b[31;42m")
    );
    assert_eq!(
        Some(vec![BackgroundGreen]),
        parse_escape_code_prefix(b"\x1b[31;;42m")
    );
    assert_eq!(
        Some(vec![ForegroundRed, Bold, BackgroundGreen]),
        parse_escape_code_prefix(b"\x1b[31;1;42m")
    );
    assert_eq!(
        Some(vec![Invisible, Bold, BackgroundGreen]),
        parse_escape_code_prefix(b"\x1b[8;1;42m")
    );
    assert_eq!(
        Some(vec![Bold, BackgroundGreen]),
        parse_escape_code_prefix(b"\x1b[8;;1;42m")
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
