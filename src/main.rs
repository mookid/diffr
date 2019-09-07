use atty::{is, Stream};
use std::io::{self, BufRead};
use std::time::SystemTime;
use termcolor::{
    Color,
    Color::{Green, Red, Rgb},
    ColorChoice, ColorSpec, StandardStream, WriteColor,
};

use diffr_lib::{DiffInput, HashedSpan, LineSplit, Snake, Tokenization};

mod cli_args;

#[derive(Debug)]
pub struct AppConfig {
    debug: bool,
    added_face: ColorSpec,
    refine_added_face: ColorSpec,
    removed_face: ColorSpec,
    refine_removed_face: ColorSpec,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            debug: false,
            added_face: color_spec(Some(Green), None, false),
            refine_added_face: color_spec(None, Some(Green), true),
            removed_face: color_spec(Some(Red), None, false),
            refine_removed_face: color_spec(None, Some(Red), true),
        }
    }
}

fn main() {
    let matches = cli_args::get_matches();
    if is(Stream::Stdin) {
        eprintln!("{}", matches.usage());
        std::process::exit(-1)
    }

    let mut config = AppConfig::default();
    config.debug = matches.is_present(cli_args::FLAG_DEBUG);

    if let Some(values) = matches.values_of(cli_args::FLAG_COLOR) {
        if let Err(err) = cli_args::parse_color_args(&mut config, values) {
            eprintln!("{}", err);
            std::process::exit(-1)
        }
    }

    match try_main(config) {
        Ok(()) => (),
        Err(ref err) if err.kind() == io::ErrorKind::BrokenPipe => (),
        Err(ref err) => {
            eprintln!("io error: {}", err);
            std::process::exit(-1)
        }
    }
}

fn now(debug: bool) -> Option<SystemTime> {
    if debug {
        Some(SystemTime::now())
    } else {
        None
    }
}

fn duration_ms(time: &Option<SystemTime>) -> u128 {
    if let Some(time) = time {
        if let Ok(elapsed) = time.elapsed() {
            elapsed.as_millis()
        } else {
            // some non monotonically increasing clock
            // this is a short period of time anyway,
            // let us map it to 0
            0
        }
    } else {
        0
    }
}

fn try_main(config: AppConfig) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut buffer = vec![];
    let mut hunk_buffer = HunkBuffer::default();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();
    let mut in_hunk = false;

    let mut time_computing_diff_ms = 0;
    let debug = config.debug;
    hunk_buffer.config = config;
    let start = now(debug);

    // process hunks
    loop {
        stdin.read_until(b'\n', &mut buffer)?;
        if buffer.is_empty() {
            break;
        }

        match (in_hunk, first_after_escape(&buffer)) {
            (true, Some(b'+')) => hunk_buffer.push_added(&buffer),
            (true, Some(b'-')) => hunk_buffer.push_removed(&buffer),
            (true, Some(b' ')) => add_raw_line(&mut hunk_buffer.lines, &buffer),
            (_, other) => {
                let start = now(debug);
                if in_hunk {
                    hunk_buffer.process(&mut stdout)?;
                }
                in_hunk = other == Some(b'@');
                output(&buffer, &ColorSpec::default(), &mut stdout)?;
                time_computing_diff_ms += duration_ms(&start);
            }
        }
        buffer.clear();
    }

    // flush remaining hunk
    hunk_buffer.process(&mut stdout)?;
    if debug {
        eprintln!("hunk processing time (ms): {}", time_computing_diff_ms);
        eprintln!("total processing time (ms): {}", duration_ms(&start));
    }
    Ok(())
}

fn color_spec(fg: Option<Color>, bg: Option<Color>, bold: bool) -> ColorSpec {
    let mut colorspec: ColorSpec = ColorSpec::default();
    colorspec.set_fg(fg);
    colorspec.set_bg(bg);
    colorspec.set_bold(bold);
    colorspec
}

#[derive(Default)]
struct HunkBuffer {
    v: Vec<isize>,
    diff_buffer: Vec<Snake>,
    added_tokens: Vec<HashedSpan>,
    removed_tokens: Vec<HashedSpan>,
    lines: LineSplit,
    config: AppConfig,
}

impl HunkBuffer {
    // Returns the number of completely printed snakes
    fn paint_line<Stream, Positions>(
        data: &[u8],
        &(data_lo, data_hi): &(usize, usize),
        no_highlight: &ColorSpec,
        highlight: &ColorSpec,
        shared: Positions,
        out: &mut Stream,
    ) -> io::Result<usize>
    where
        Stream: WriteColor,
        Positions: Iterator<Item = (usize, usize)>,
    {
        let mut y = data_lo;
        let mut nshared = 0;
        for (lo, hi) in shared {
            if hi <= data_lo {
                nshared += 1;
                continue;
            }
            // XXX: always highlight the leading +/- character
            let lo = lo.max(data_lo + 1);
            let hi = hi.min(data_hi);
            if hi <= lo {
                continue;
            }
            output(&data[y..lo], &highlight, out)?;
            output(&data[lo..hi], &no_highlight, out)?;
            y = hi;
        }
        if y < data_hi {
            output(&data[y..data_hi], &highlight, out)?;
        }
        Ok(nshared)
    }

    fn process<Stream>(&mut self, out: &mut Stream) -> io::Result<()>
    where
        Stream: WriteColor,
    {
        let Self {
            v,
            diff_buffer,
            added_tokens,
            removed_tokens,
            lines,
            config,
        } = self;
        let data = lines.data();
        let tokens = DiffInput {
            removed: Tokenization::new(lines.data(), removed_tokens),
            added: Tokenization::new(lines.data(), added_tokens),
        };
        diffr_lib::diff(&tokens, v, diff_buffer);
        let mut ishared_added = 0;
        let mut ishared_removed = 0;
        for (line_start, line_end) in lines.iter() {
            let first = data[line_start];
            match first {
                b'-' | b'+' => {
                    let is_plus = first == b'+';
                    let (nohighlight, highlight, toks, i) = if is_plus {
                        (
                            &config.added_face,
                            &config.refine_added_face,
                            &tokens.added,
                            &mut ishared_added,
                        )
                    } else {
                        (
                            &config.removed_face,
                            &config.refine_removed_face,
                            &tokens.removed,
                            &mut ishared_removed,
                        )
                    };
                    let shared = diff_buffer.iter().skip(*i).map(|s| {
                        let x0 = if is_plus { s.y0 } else { s.x0 };
                        let first = toks.nth_span(x0).lo;
                        let last = toks.nth_span(x0 + s.len - 1).hi;
                        (first, last)
                    });
                    *i += Self::paint_line(
                        toks.data(),
                        &(line_start, line_end),
                        &nohighlight,
                        &highlight,
                        shared,
                        out,
                    )?;
                }
                _ => output(&data[line_start..line_end], &ColorSpec::default(), out)?,
            }
        }
        lines.clear();
        added_tokens.clear();
        removed_tokens.clear();
        Ok(())
    }

    fn push_added(&mut self, line: &[u8]) {
        self.push_aux(line, true)
    }

    fn push_removed(&mut self, line: &[u8]) {
        self.push_aux(line, false)
    }

    fn push_aux(&mut self, line: &[u8], added: bool) {
        // XXX: don't tokenize the leading +/- character
        let ofs = self.lines.len() + 1;
        add_raw_line(&mut self.lines, line);
        diffr_lib::tokenize(
            &self.lines.data(),
            ofs,
            if added {
                &mut self.added_tokens
            } else {
                &mut self.removed_tokens
            },
        );
    }
}

fn add_raw_line(dst: &mut LineSplit, line: &[u8]) {
    let mut i = 0;
    let len = line.len();
    while i < len {
        i += skip_all_escape_code(&line[i..]);
        let tok_len = skip_token(&line[i..]);
        dst.append_line(&line[i..i + tok_len]);
        i += tok_len;
    }
}

fn output<Stream>(buf: &[u8], colorspec: &ColorSpec, out: &mut Stream) -> io::Result<()>
where
    Stream: WriteColor,
{
    if buf.is_empty() {
        return Ok(());
    }
    let ends_with_newline = buf.last().cloned() == Some(b'\n');
    let buf = if ends_with_newline {
        &buf[..buf.len() - 1]
    } else {
        buf
    };
    out.set_color(colorspec)?;
    out.write_all(&buf)?;
    out.reset()?;
    if ends_with_newline {
        out.write_all(b"\n")?;
    }
    Ok(())
}

/// Returns the number of bytes of escape code that start the slice.
fn skip_all_escape_code(buf: &[u8]) -> usize {
    // Skip one sequence
    fn skip_escape_code(buf: &[u8]) -> Option<usize> {
        if 2 <= buf.len() && &buf[..2] == b"\x1b[" {
            // "\x1b[" + sequence body + "m" => 3 additional bytes
            Some(index_of(&buf[2..], b'm')? + 3)
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

/// Returns the first byte of the slice, after skipping the escape
/// code bytes.
fn first_after_escape(buf: &[u8]) -> Option<u8> {
    let nbytes = skip_all_escape_code(&buf);
    buf.iter().skip(nbytes).cloned().next()
}

/// Scan the slice looking for the given byte, returning the index of
/// its first appearance.
fn index_of(buf: &[u8], target: u8) -> Option<usize> {
    let mut it = buf.iter().enumerate();
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

/// Computes the number of bytes until either the next escape code, or
/// the end of buf.
fn skip_token(buf: &[u8]) -> usize {
    match buf.len() {
        0 => 0,
        len => {
            for i in 0..buf.len() - 1 {
                if &buf[i..i + 2] == b"\x1b[" {
                    return i;
                }
            }
            len
        }
    }
}

#[cfg(test)]
mod test;

#[cfg(test)]
mod test_cli;
