use std::fmt::{Debug, Display, Error as FmtErr, Formatter};
use std::io::{self, BufRead, Write};
use std::iter::Peekable;
use std::time::SystemTime;
use termcolor::{
    Color::{self, Green, Red, Rgb},
    ColorChoice, ColorSpec, StandardStream, WriteColor,
};

use diffr_lib::*;

mod cli_args;
mod diffr_lib;

#[derive(Debug, Clone, Copy)]
pub enum LineNumberStyle {
    Compact,
    Aligned,
    Fixed(usize),
}

impl LineNumberStyle {
    fn min_width(&self) -> usize {
        match *self {
            LineNumberStyle::Compact | LineNumberStyle::Aligned => 0,
            LineNumberStyle::Fixed(w) => w,
        }
    }
}

#[derive(Debug)]
pub struct AppConfig {
    debug: bool,
    line_numbers_style: Option<LineNumberStyle>,
    added_face: ColorSpec,
    refine_added_face: ColorSpec,
    removed_face: ColorSpec,
    refine_removed_face: ColorSpec,
    large_diff_threshold: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        // The ANSI white is actually gray on many implementations. The actual white
        // that seem to work on all implementations is "bright white". `termcolor`
        // crate has no enum member for it, so we create it with Rgb.
        let bright_white = Rgb(255, 255, 255);
        AppConfig {
            debug: false,
            line_numbers_style: None,
            added_face: color_spec(Some(Green), None, false),
            refine_added_face: color_spec(Some(bright_white), Some(Green), true),
            removed_face: color_spec(Some(Red), None, false),
            refine_removed_face: color_spec(Some(bright_white), Some(Red), true),
            large_diff_threshold: 1000,
        }
    }
}

impl AppConfig {
    fn has_line_numbers(&self) -> bool {
        self.line_numbers_style.is_some()
    }

    fn line_numbers_aligned(&self) -> bool {
        if let Some(LineNumberStyle::Aligned) = self.line_numbers_style {
            return true;
        }
        false
    }
}

fn main() {
    let config = cli_args::parse_config();
    let mut hunk_buffer = HunkBuffer::new(config);
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

struct HunkBuffer {
    v: Vec<isize>,
    diff_buffer: Vec<Snake>,
    added_tokens: Vec<(usize, usize)>,
    removed_tokens: Vec<(usize, usize)>,
    line_number_info: Option<HunkHeader>,
    lines: LineSplit,
    config: AppConfig,
    margin: Vec<u8>,
    warning_lines: Vec<usize>,
    stats: ExecStats,
}

#[derive(Default)]
struct Margin<'a> {
    lino_minus: usize,
    lino_plus: usize,
    margin: &'a mut [u8],
    half_margin: usize,
}

const MARGIN_TAB_STOP: usize = 8;

impl<'a> Margin<'a> {
    fn new(header: &'a HunkHeader, margin: &'a mut [u8], config: &'a AppConfig) -> Self {
        let full_margin = header.width(config.line_numbers_style);
        let half_margin = full_margin / 2;

        // If line number is 0, the column is empty and
        // shouldn't be printed
        let margin_size = if header.minus_range.0 == 0 || header.plus_range.0 == 0 {
            half_margin
        } else {
            full_margin
        };
        assert!(margin.len() >= margin_size);
        Margin {
            lino_plus: header.plus_range.0,
            lino_minus: header.minus_range.0,
            margin: &mut margin[..margin_size],
            half_margin,
        }
    }

    fn write_margin_padding(&mut self, out: &mut impl WriteColor) -> io::Result<()> {
        if self.margin.len() % MARGIN_TAB_STOP != 0 {
            write!(out, "\t")?;
        }
        Ok(())
    }

    fn write_margin_changed(
        &mut self,
        is_plus: bool,
        config: &AppConfig,
        out: &mut impl WriteColor,
    ) -> io::Result<()> {
        let mut margin_buf = &mut self.margin[..];
        let color;
        if is_plus {
            color = &config.added_face;
            if self.lino_minus != 0 {
                write!(margin_buf, "{:w$} ", ' ', w = self.half_margin)?;
            }
            write!(margin_buf, "{:w$}", self.lino_plus, w = self.half_margin)?;
            self.lino_plus += 1;
        } else {
            color = &config.removed_face;
            write!(margin_buf, "{:w$}", self.lino_minus, w = self.half_margin)?;
            if self.lino_plus != 0 {
                write!(margin_buf, " {:w$}", ' ', w = self.half_margin)?;
            }
            self.lino_minus += 1;
        };
        output(self.margin, 0, self.margin.len(), color, out)?;
        if config.line_numbers_aligned() {
            self.write_margin_padding(out)?;
        }
        Ok(())
    }

    fn write_margin_context(
        &mut self,
        config: &AppConfig,
        out: &mut impl WriteColor,
    ) -> io::Result<()> {
        if self.lino_minus != self.lino_plus {
            write!(out, "{:w$}", self.lino_minus, w = self.half_margin)?;
        } else {
            write!(out, "{:w$}", ' ', w = self.half_margin)?;
        }
        write!(out, " {:w$}", self.lino_plus, w = self.half_margin)?;
        if config.line_numbers_aligned() {
            self.write_margin_padding(out)?;
        }
        self.lino_minus += 1;
        self.lino_plus += 1;
        Ok(())
    }
}

fn shared_spans(added_tokens: &Tokenization, diff_buffer: &[Snake]) -> Vec<(usize, usize)> {
    let mut shared_spans = vec![];
    for snake in diff_buffer.iter() {
        for i in 0..snake.len {
            shared_spans.push(added_tokens.nth_span(snake.y0 + i));
        }
    }
    shared_spans
}

const MAX_MARGIN: usize = 41;

impl HunkBuffer {
    fn new(config: AppConfig) -> Self {
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
            output(data, data_lo, data_lo + 1, no_highlight, out)?;
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
            stats,
        } = self;
        let mut margin = match line_number_info {
            Some(lni) => Margin::new(lni, margin, config),
            None => Default::default(),
        };
        let data = lines.data();
        let m = TokenMap::new(&mut [(removed_tokens.iter(), data), (added_tokens.iter(), data)]);
        let removed = Tokenization::new(data, removed_tokens, &m);
        let added = Tokenization::new(data, added_tokens, &m);
        let tokens = DiffInput::new(&added, &removed, config.large_diff_threshold);
        let start = now(stats.do_timings());
        diffr_lib::diff(&tokens, v, diff_buffer);
        // TODO output the lcs directly out of `diff` instead
        let shared_spans = shared_spans(&added, diff_buffer);
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

        for (i, range) in lines.iter().enumerate() {
            if let Some(&&nline) = warnings.peek() {
                if nline == i {
                    let w = &lines.data()[range.0..range.1];
                    output(w, 0, w.len(), &defaultspec, out)?;
                    warnings.next();
                    continue;
                }
            }
            let first = data[range.0];
            match first {
                b'-' | b'+' => {
                    let is_plus = first == b'+';
                    let (nhl, hl, toks, shared) = if is_plus {
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
                    if config.has_line_numbers() {
                        margin.write_margin_changed(is_plus, config, out)?
                    }
                    Self::paint_line(toks.data(), &range, nhl, hl, shared, out)?;
                }
                _ => {
                    if config.has_line_numbers() {
                        margin.write_margin_context(config, out)?
                    }
                    output(data, range.0, range.1, &defaultspec, out)?
                }
            }
        }
        assert!(warnings.peek().is_none());
        drop(shared_removed);
        drop(shared_added);
        lines.clear();
        added_tokens.clear();
        removed_tokens.clear();
        warning_lines.clear();
        Ok(())
    }

    fn push_added(&mut self, line: &[u8]) {
        self.push_aux(line, true)
    }

    fn push_removed(&mut self, line: &[u8]) {
        self.push_aux(line, false)
    }

    fn push_aux(&mut self, line: &[u8], added: bool) {
        // XXX: skip leading token
        let mut ofs = self.lines.len() + 1;
        add_raw_line(&mut self.lines, line);
        // get back the line sanitized from escape codes:
        let line = &self.lines.data()[ofs..];
        // skip leading spaces
        ofs += line
            .iter()
            .take_while(|ch| ch.is_ascii_whitespace())
            .count();
        diffr_lib::tokenize(
            self.lines.data(),
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
        let mut hunk_line_number = 0;

        // process hunks
        loop {
            stdin.read_until(b'\n', &mut buffer)?;
            if buffer.is_empty() {
                break;
            }

            let first = first_after_escape(&buffer);
            if in_hunk {
                hunk_line_number += 1;
                match first {
                    Some(b'+') => self.push_added(&buffer),
                    Some(b'-') => self.push_removed(&buffer),
                    Some(b' ') => add_raw_line(&mut self.lines, &buffer),
                    Some(b'\\') => {
                        add_raw_line(&mut self.lines, &buffer);
                        self.warning_lines.push(hunk_line_number - 1);
                    }
                    _ => {
                        self.process_with_stats(&mut stdout)?;
                        in_hunk = false;
                    }
                }
            }
            if !in_hunk {
                hunk_line_number = 0;
                in_hunk = first == Some(b'@');
                if self.config.has_line_numbers() && in_hunk {
                    self.line_number_info = parse_line_number(&buffer);
                }
                output(&buffer, 0, buffer.len(), &ColorSpec::default(), &mut stdout)?;
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
    out.write_all(buf)?;
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
    while let Some(nbytes) = skip_escape_code(buf) {
        buf = &buf[nbytes..];
        sum += nbytes
    }
    sum
}

/// Returns the first byte of the slice, after skipping the escape
/// code bytes.
fn first_after_escape(buf: &[u8]) -> Option<u8> {
    let nbytes = skip_all_escape_code(buf);
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

// TODO: extend to the multiple range case
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

fn width1(x: u64, st: Option<LineNumberStyle>) -> usize {
    let result = WIDTH.binary_search(&x);
    let result = match result {
        Ok(i) | Err(i) => i,
    };
    st.map(|style| style.min_width()).unwrap_or(0).max(result)
}

impl HunkHeader {
    fn new(minus_range: (usize, usize), plus_range: (usize, usize)) -> Self {
        HunkHeader {
            minus_range,
            plus_range,
        }
    }

    fn width(&self, st: Option<LineNumberStyle>) -> usize {
        let w1 = width1((self.minus_range.0 + self.minus_range.1) as u64, st);
        let w2 = width1((self.plus_range.0 + self.plus_range.1) as u64, st);
        2 * w1.max(w2) + 1
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

    fn expect_multiple_minus_ranges(&mut self) -> Option<(usize, usize)> {
        let next = |that: &mut Self| {
            that.expect(b'-')?;
            that.parse_pair()
        };
        let mut res = None;
        for i in 0.. {
            if i != 0 {
                self.expect_multiple(|x| x.is_ascii_whitespace())?;
            }
            match next(self) {
                next @ Some(_) => res = next,
                None => break,
            }
        }
        res
    }

    fn parse_line_number(&mut self) -> Option<HunkHeader> {
        self.skip_whitespaces();
        self.expect_multiple(|x| x == b'@')?;
        self.expect_multiple(|x| x.is_ascii_whitespace())?;
        let minus_range = self.expect_multiple_minus_ranges()?;
        self.expect(b'+')?;
        let plus_range = self.parse_pair()?;
        self.expect_multiple(|x| x.is_ascii_whitespace())?;
        self.expect_multiple(|x| x == b'@')?;
        Some(HunkHeader::new(minus_range, plus_range))
    }
}

fn parse_line_number(buf: &[u8]) -> Option<HunkHeader> {
    LineNumberParser::new(buf).parse_line_number()
}

#[cfg(test)]
mod tests_app;

#[cfg(test)]
mod tests_cli;
