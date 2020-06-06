use atty::{is, Stream};

use std::fmt::{Debug, Display, Error as FmtErr, Formatter};
use std::io::{self, BufRead, Write};
use std::iter::Peekable;
use std::time::SystemTime;
use termcolor::{
    Color,
    Color::{Green, Red, White},
    ColorChoice, ColorSpec, StandardStream, WriteColor,
};

use diffr_lib::optimize_partition;
use diffr_lib::DiffInput;
use diffr_lib::LineSplit;
use diffr_lib::Snake;
use diffr_lib::TokenMap;
use diffr_lib::Tokenization;

mod cli_args;

#[derive(Debug)]
pub struct AppConfig {
    debug: bool,
    line_numbers: bool,
    added_face: ColorSpec,
    refine_added_face: ColorSpec,
    removed_face: ColorSpec,
    refine_removed_face: ColorSpec,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            debug: false,
            line_numbers: false,
            added_face: color_spec(Some(Green), None, false),
            refine_added_face: color_spec(Some(White), Some(Green), true),
            removed_face: color_spec(Some(Red), None, false),
            refine_removed_face: color_spec(Some(White), Some(Red), true),
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
    config.line_numbers = matches.is_present(cli_args::FLAG_LINE_NUMBERS);

    if let Some(values) = matches.values_of(cli_args::FLAG_COLOR) {
        if let Err(err) = cli_args::parse_color_args(&mut config, values) {
            eprintln!("{}", err);
            std::process::exit(-1)
        }
    }

    let mut hunk_buffer = HunkBuffer::new(&config);
    match hunk_buffer.run() {
        Ok(()) => (),
        Err(ref err) if err.kind() == io::ErrorKind::BrokenPipe => (),
        Err(ref err) => {
            eprintln!("io error: {}", err);
            std::process::exit(-1)
        }
    }
}

fn now(do_timings: bool) -> Option<SystemTime> {
    if do_timings {
        Some(SystemTime::now())
    } else {
        None
    }
}

fn duration_ms_since(time: &Option<SystemTime>) -> u128 {
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

fn color_spec(fg: Option<Color>, bg: Option<Color>, bold: bool) -> ColorSpec {
    let mut colorspec: ColorSpec = ColorSpec::default();
    colorspec.set_fg(fg);
    colorspec.set_bg(bg);
    colorspec.set_bold(bold);
    colorspec
}

#[derive(Default)]
struct ExecStats {
    time_computing_diff_ms: u128,
    time_lcs_ms: u128,
    time_opt_lcs_ms: u128,
    total_time_ms: u128,
    program_start: Option<SystemTime>,
}

impl ExecStats {
    fn new(debug: bool) -> Self {
        ExecStats {
            time_computing_diff_ms: 0,
            time_lcs_ms: 0,
            time_opt_lcs_ms: 0,
            total_time_ms: 0,
            program_start: now(debug),
        }
    }

    /// Should we call SystemTime::now at all?
    fn do_timings(&self) -> bool {
        self.program_start.is_some()
    }

    fn stop(&mut self) {
        if self.do_timings() {
            self.total_time_ms = duration_ms_since(&self.program_start);
        }
    }

    fn report(&self) -> std::io::Result<()> {
        self.report_into(&mut std::io::stderr())
    }

    fn report_into<W>(&self, w: &mut W) -> std::io::Result<()>
    where
        W: std::io::Write,
    {
        const WORD_PADDING: usize = 35;
        const FIELD_PADDING: usize = 15;
        if self.do_timings() {
            let format_header = |name| format!("{} (ms)", name);
            let format_ratio = |dt: u128| {
                format!(
                    "({:3.3}%)",
                    100.0 * (dt as f64) / (self.total_time_ms as f64)
                )
            };
            let mut report = |name: &'static str, dt: u128| {
                writeln!(
                    w,
                    "{:>w$} {:>f$} {:>f$}",
                    format_header(name),
                    dt,
                    format_ratio(dt),
                    w = WORD_PADDING,
                    f = FIELD_PADDING,
                )
            };
            report("hunk processing time", self.time_computing_diff_ms)?;
            report("-- compute lcs", self.time_lcs_ms)?;
            report("-- optimize lcs", self.time_opt_lcs_ms)?;
            writeln!(
                w,
                "{:>w$} {:>f$}",
                format_header("total processing time"),
                self.total_time_ms,
                w = WORD_PADDING,
                f = FIELD_PADDING,
            )?;
        }
        Ok(())
    }
}

struct HunkBuffer<'a> {
    v: Vec<isize>,
    diff_buffer: Vec<Snake>,
    added_tokens: Vec<(usize, usize)>,
    removed_tokens: Vec<(usize, usize)>,
    line_number_info: Option<HunkHeader>,
    lines: LineSplit,
    config: &'a AppConfig,
    margin: Vec<u8>,
    warning_lines: Vec<usize>,
    hunk_line_number: usize,
    stats: ExecStats,
}

fn shared_spans(added_tokens: &Tokenization, diff_buffer: &Vec<Snake>) -> Vec<(usize, usize)> {
    let mut shared_spans = vec![];
    for snake in diff_buffer.iter() {
        for i in 0..snake.len {
            shared_spans.push(added_tokens.nth_span(snake.y0 + i));
        }
    }
    shared_spans
}

const MAX_MARGIN: usize = 41;

impl<'a> HunkBuffer<'a> {
    fn new(config: &'a AppConfig) -> Self {
        let debug = config.debug;
        HunkBuffer {
            v: vec![],
            diff_buffer: vec![],
            added_tokens: vec![],
            removed_tokens: vec![],
            line_number_info: None,
            lines: Default::default(),
            config,
            margin: vec![0; MAX_MARGIN],
            warning_lines: vec![],
            hunk_line_number: 0,
            stats: ExecStats::new(debug),
        }
    }

    // Returns the number of completely printed snakes
    fn paint_line<Stream, Positions>(
        data: &[u8],
        &(data_lo, data_hi): &(usize, usize),
        no_highlight: &ColorSpec,
        highlight: &ColorSpec,
        shared: &mut Peekable<Positions>,
        out: &mut Stream,
    ) -> io::Result<()>
    where
        Stream: WriteColor,
        Positions: Iterator<Item = (usize, usize)>,
    {
        let mut y = data_lo + 1;
        // XXX: skip leading token and leading spaces
        while y < data_hi && data[y].is_ascii_whitespace() {
            y += 1
        }
        let mut pending = (data_lo, y, false);
        let mut trailing_ws = ColorSpec::new();
        trailing_ws.set_bg(Some(Color::Red));
        let color = |h| if h { &highlight } else { &no_highlight };
        let mut output1 = |lo, hi, highlighted| -> std::io::Result<()> {
            if lo == hi {
                return Ok(());
            }
            let (lo1, hi1, highlighted1) = pending;
            let color = if &data[lo..hi] == b"\n"
                && data[lo1..hi1].iter().all(|b| b.is_ascii_whitespace())
            {
                &trailing_ws
            } else {
                color(highlighted1)
            };
            output(data, lo1, hi1, color, out)?;
            pending = (lo, hi, highlighted);
            Ok(())
        };
        // special case: all whitespaces
        if y == data_hi {
            output(data, data_lo, data_lo + 1, &no_highlight, out)?;
            output(data, data_lo + 1, data_hi, &trailing_ws, out)?;
            return Ok(());
        }

        while let Some((lo, hi)) = shared.peek() {
            if data_hi <= y {
                break;
            }
            let last_iter = data_hi <= *hi;
            let lo = (*lo).min(data_hi).max(y);
            let hi = (*hi).min(data_hi);
            if hi <= data_lo {
                shared.next();
                continue;
            }
            if hi < lo {
                continue;
            }
            output1(y, lo, true)?;
            output1(lo, hi, false)?;
            y = hi;
            if last_iter {
                break;
            } else {
                shared.next();
            }
        }
        output1(y, data_hi, true)?;
        let (lo1, hi1, highlighted1) = pending;
        output(data, lo1, hi1, color(highlighted1), out)?;
        Ok(())
    }

    fn process_with_stats<Stream>(&mut self, out: &mut Stream) -> io::Result<()>
    where
        Stream: WriteColor,
    {
        let start = now(self.stats.do_timings());
        let result = self.process(out);
        self.stats.time_computing_diff_ms += duration_ms_since(&start);
        result
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
            line_number_info,
            lines,
            config,
            margin,
            warning_lines,
            hunk_line_number,
            stats,
        } = self;
        let (mut current_line_minus, mut current_line_plus, margin, half_margin) =
            match line_number_info {
                Some(lni) => {
                    let full_margin = lni.width();
                    let half_margin = full_margin / 2;

                    // If line number is 0, the column is empty and
                    // shouldn't be printed
                    let margin_size = if lni.minus_range.0 == 0 || lni.plus_range.0 == 0 {
                        half_margin
                    } else {
                        full_margin
                    };
                    assert!(margin.len() >= margin_size);
                    (
                        lni.minus_range.0,
                        lni.plus_range.0,
                        &mut margin[..margin_size],
                        half_margin,
                    )
                }
                None => Default::default(),
            };
        let data = lines.data();
        let m = TokenMap::new(&mut [(removed_tokens.iter(), data), (added_tokens.iter(), data)]);
        let removed = Tokenization::new(data, removed_tokens, &m);
        let added = Tokenization::new(data, added_tokens, &m);
        let tokens = DiffInput::new(&added, &removed);
        let start = now(stats.do_timings());
        diffr_lib::diff(&tokens, v, diff_buffer);
        // TODO output the lcs directly out of `diff` instead
        let shared_spans = shared_spans(&added, &diff_buffer);
        let lcs = Tokenization::new(data, &shared_spans, &m);
        stats.time_lcs_ms += duration_ms_since(&start);
        let start = now(stats.do_timings());
        let normalized_lcs_added = optimize_partition(&added, &lcs);
        let normalized_lcs_removed = optimize_partition(&removed, &lcs);
        stats.time_opt_lcs_ms += duration_ms_since(&start);
        let mut shared_added = normalized_lcs_added.shared_segments(&added).peekable();
        let mut shared_removed = normalized_lcs_removed.shared_segments(&removed).peekable();
        let mut warnings = warning_lines.iter().peekable();
        let defaultspec = ColorSpec::default();

        for (i, (line_start, line_end)) in lines.iter().enumerate() {
            if let Some(&&nline) = warnings.peek() {
                if nline == i {
                    let w = &lines.data()[line_start..line_end];
                    output(w, 0, w.len(), &defaultspec, out)?;
                    warnings.next();
                    continue;
                }
            }
            let first = data[line_start];
            match first {
                b'-' | b'+' => {
                    let is_plus = first == b'+';
                    let (nohighlight, highlight, toks, shared) = if is_plus {
                        (
                            &config.added_face,
                            &config.refine_added_face,
                            tokens.added(),
                            &mut shared_added,
                        )
                    } else {
                        (
                            &config.removed_face,
                            &config.refine_removed_face,
                            tokens.removed(),
                            &mut shared_removed,
                        )
                    };

                    if config.line_numbers {
                        let mut margin_buf = &mut margin[..];
                        if is_plus {
                            if current_line_minus != 0 {
                                write!(margin_buf, "{:w$} ", ' ', w = half_margin)?;
                            }
                            write!(margin_buf, "{:w$}", current_line_plus, w = half_margin)?;
                            current_line_plus += 1;
                        } else {
                            write!(margin_buf, "{:w$}", current_line_minus, w = half_margin)?;
                            if current_line_plus != 0 {
                                write!(margin_buf, " {:w$}", ' ', w = half_margin)?;
                            }
                            current_line_minus += 1;
                        };
                        output(margin, 0, margin.len(), &nohighlight, out)?
                    }

                    Self::paint_line(
                        toks.data(),
                        &(line_start, line_end),
                        &nohighlight,
                        &highlight,
                        shared,
                        out,
                    )?;
                }
                _ => {
                    if config.line_numbers {
                        if current_line_minus != current_line_plus {
                            write!(out, "{:w$}", current_line_minus, w = half_margin)?;
                        } else {
                            write!(out, "{:w$}", ' ', w = half_margin)?;
                        }
                        write!(out, " {:w$}", current_line_plus, w = half_margin)?;
                    }
                    current_line_minus += 1;
                    current_line_plus += 1;
                    output(data, line_start, line_end, &defaultspec, out)?
                }
            }
        }
        assert!(warnings.peek() == None);
        lines.clear();
        added_tokens.clear();
        removed_tokens.clear();
        warning_lines.clear();
        *hunk_line_number = 0;
        Ok(())
    }

    fn push_added(&mut self, line: &[u8]) {
        self.push_aux(line, true)
    }

    fn push_removed(&mut self, line: &[u8]) {
        self.push_aux(line, false)
    }

    fn push_aux(&mut self, line: &[u8], added: bool) {
        let mut ofs = self.lines.len() + 1;
        add_raw_line(&mut self.lines, line);
        // XXX: skip leading token and leading spaces
        while ofs < line.len() && line[ofs].is_ascii_whitespace() {
            ofs += 1
        }
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

    fn run(&mut self) -> io::Result<()> {
        let stdin = io::stdin();
        let stdout = StandardStream::stdout(ColorChoice::Always);
        let mut buffer = vec![];
        let mut stdin = stdin.lock();
        let mut stdout = stdout.lock();
        let mut in_hunk = false;

        // process hunks
        loop {
            stdin.read_until(b'\n', &mut buffer)?;
            if buffer.is_empty() {
                break;
            }

            if in_hunk {
                self.hunk_line_number += 1;
            }
            match (in_hunk, first_after_escape(&buffer)) {
                (true, Some(b'+')) => self.push_added(&buffer),
                (true, Some(b'-')) => self.push_removed(&buffer),
                (true, Some(b' ')) => add_raw_line(&mut self.lines, &buffer),
                (true, Some(b'\\')) => {
                    add_raw_line(&mut self.lines, &buffer);
                    self.warning_lines.push(self.hunk_line_number - 1);
                }
                (_, other) => {
                    if in_hunk {
                        self.process_with_stats(&mut stdout)?;
                    }
                    in_hunk = other == Some(b'@');
                    if self.config.line_numbers && in_hunk {
                        self.line_number_info = parse_line_number(&buffer);
                    }
                    output(&buffer, 0, buffer.len(), &ColorSpec::default(), &mut stdout)?;
                }
            }
            buffer.clear();
        }

        // flush remaining hunk
        self.process_with_stats(&mut stdout)?;
        self.stats.stop();
        self.stats.report()?;
        Ok(())
    }
}

// TODO count whitespace characters as well here
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

fn output<Stream>(
    buf: &[u8],
    from: usize,
    to: usize,
    colorspec: &ColorSpec,
    out: &mut Stream,
) -> io::Result<()>
where
    Stream: WriteColor,
{
    let to = to.min(buf.len());
    if from >= to {
        return Ok(());
    }
    let buf = &buf[from..to];
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

#[derive(Default, PartialEq, Eq)]
struct HunkHeader {
    // range are (ofs,len) for the interval [ofs, ofs + len)
    minus_range: (usize, usize),
    plus_range: (usize, usize),
}

const WIDTH: [u64; 20] = [
    0,
    9,
    99,
    999,
    9999,
    99999,
    999999,
    9999999,
    99999999,
    999999999,
    9999999999,
    99999999999,
    999999999999,
    9999999999999,
    99999999999999,
    999999999999999,
    9999999999999999,
    99999999999999999,
    999999999999999999,
    9999999999999999999,
];

fn width1(x: u64) -> usize {
    let result = WIDTH.binary_search(&x);
    match result {
        Ok(i) | Err(i) => i,
    }
}

impl HunkHeader {
    fn new(minus_range: (usize, usize), plus_range: (usize, usize)) -> Self {
        HunkHeader {
            minus_range,
            plus_range,
        }
    }

    fn width(&self) -> usize {
        2 * width1((self.minus_range.0 + self.minus_range.1) as u64)
            .max(width1((self.plus_range.0 + self.plus_range.1) as u64))
            + 1
    }
}

impl Debug for HunkHeader {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtErr> {
        f.write_fmt(format_args!(
            "-{},{} +{},{}",
            self.minus_range.0, self.minus_range.1, self.plus_range.0, self.plus_range.1,
        ))
    }
}

impl Display for HunkHeader {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtErr> {
        Debug::fmt(&self, f)
    }
}

struct LineNumberParser<'a> {
    buf: &'a [u8],
    i: usize,
}

impl<'a> LineNumberParser<'a> {
    fn new(buf: &'a [u8]) -> Self {
        LineNumberParser { buf, i: 0 }
    }

    fn skip_escape_code(&mut self) {
        if self.i < self.buf.len() {
            let to_skip = skip_all_escape_code(&self.buf[self.i..]);
            self.i += to_skip;
        }
    }

    fn looking_at<M>(&mut self, matcher: M) -> bool
    where
        M: Fn(u8) -> bool,
    {
        self.skip_escape_code();
        self.i < self.buf.len() && matcher(self.buf[self.i])
    }

    fn read_digit(&mut self) -> Option<usize> {
        if self.looking_at(|x| x.is_ascii_digit()) {
            let cur = self.buf[self.i];
            self.i += 1;
            Some((cur - b'0') as usize)
        } else {
            None
        }
    }

    fn skip_whitespaces(&mut self) {
        while self.looking_at(|x| x.is_ascii_whitespace()) {
            self.i += 1;
        }
    }

    fn expect_multiple<M>(&mut self, matcher: M) -> Option<usize>
    where
        M: Fn(u8) -> bool,
    {
        self.skip_escape_code();
        let iorig = self.i;
        while self.looking_at(&matcher) {
            self.i += 1;
        }
        if self.i == iorig {
            None
        } else {
            Some(self.i - iorig)
        }
    }

    fn expect(&mut self, target: u8) -> Option<()> {
        if self.looking_at(|x| x == target) {
            self.i += 1;
            Some(())
        } else {
            None
        }
    }

    fn parse_usize(&mut self) -> Option<usize> {
        let mut res = 0usize;
        let mut any = false;
        while let Some(digit) = self.read_digit() {
            any = true;
            res = res.checked_mul(10)?;
            res = res.checked_add(digit)?;
        }
        if any {
            Some(res)
        } else {
            None
        }
    }

    fn parse_pair(&mut self) -> Option<(usize, usize)> {
        let p0 = self.parse_usize()?;
        if self.expect(b',').is_none() {
            return Some((p0, 1));
        }
        let p1 = self.parse_usize()?;
        Some((p0, p1))
    }

    fn parse_line_number(&mut self) -> Option<HunkHeader> {
        self.skip_whitespaces();
        self.expect_multiple(|x| x == b'@')?;
        self.expect_multiple(|x| x.is_ascii_whitespace())?;
        self.expect(b'-')?;
        let minus_range = self.parse_pair()?;
        self.expect_multiple(|x| x.is_ascii_whitespace())?;
        self.expect(b'+')?;
        let plus_range = self.parse_pair()?;
        self.expect_multiple(|x| x.is_ascii_whitespace())?;
        self.expect_multiple(|x| x == b'@')?;
        Some(HunkHeader::new(minus_range, plus_range))
    }
}

fn parse_line_number(buf: &[u8]) -> Option<HunkHeader> {
    LineNumberParser::new(&buf).parse_line_number()
}

#[cfg(test)]
mod test;

#[cfg(test)]
mod test_cli;
