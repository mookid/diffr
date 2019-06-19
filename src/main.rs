use std::convert::TryFrom;
use std::io::{self, BufRead};
use termcolor::{
    Color,
    Color::{Green, Red},
    ColorChoice, ColorSpec, StandardStream, WriteColor,
};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = StandardStream::stdout(ColorChoice::Always);
    let mut buffer = vec![];
    let mut hunk_buffer = HunkBuffer::default();
    let mut stdin = stdin.lock();
    let mut stdout = stdout.lock();

    let mut time_computing_diff_ms = 0;
    let start = std::time::SystemTime::now();

    // echo everything before the first diff hunk
    loop {
        stdin.read_until(b'\n', &mut buffer)?;
        if buffer.is_empty() || starts_hunk(&buffer) {
            break;
        }
        output(&buffer, &ColorSpec::default(), &mut stdout)?;
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
                output(&buffer, &ColorSpec::default(), &mut stdout)?;
                time_computing_diff_ms += start.elapsed().unwrap().as_millis();
            }
        }
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

#[derive(Debug, Default)]
pub struct DiffPair<T> {
    added: T,
    removed: T,
}

#[derive(Default)]
struct HunkBuffer {
    v: Vec<isize>,
    diff_buffer: Vec<Snake>,
    added_tokens: Vec<(usize, usize)>,
    removed_tokens: Vec<(usize, usize)>,
    added_lines: Vec<u8>,
    removed_lines: Vec<u8>,
}

fn color_spec(fg: Option<Color>, bg: Option<Color>, bold: bool) -> ColorSpec {
    let mut colorspec: ColorSpec = ColorSpec::default();
    colorspec.set_fg(fg);
    colorspec.set_bg(bg);
    colorspec.set_bold(bold);
    colorspec
}

impl HunkBuffer {
    fn paint_lines<Stream, Positions>(
        tokens: &Tokenization,
        color_no_highlight: &ColorSpec,
        color_highlight: &ColorSpec,
        positions: Positions,
        out: &mut Stream,
    ) -> io::Result<()>
    where
        Stream: WriteColor,
        Positions: Iterator<Item = (isize, isize)>,
    {
        let mut y = 0;
        for (ofs, len) in positions {
            for i in y..ofs {
                output(tokens.seq(i), &color_highlight, out)?;
            }
            for i in ofs..ofs + len {
                output(tokens.seq(i), &color_no_highlight, out)?;
            }
            y = ofs + len;
        }
        for i in y..to_isize(tokens.nb_tokens()) {
            output(tokens.seq(i), &color_highlight, out)?;
        }
        Ok(())
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
            added_lines,
            removed_lines,
        } = self;
        diff_buffer.clear();
        tokenize(removed_tokens, removed_lines);
        tokenize(added_tokens, added_lines);

        let tokens = Tokens {
            removed: Tokenization::new(removed_lines, removed_tokens),
            added: Tokenization::new(added_lines, added_tokens),
        };

        diff(&tokens, v, diff_buffer);
        Self::paint_lines(
            &tokens.removed,
            &color_spec(Some(Red), None, false),
            &color_spec(Some(Red), None, true),
            diff_buffer.iter().map(|s| (s.x0, s.len)),
            out,
        )?;
        Self::paint_lines(
            &tokens.added,
            &color_spec(Some(Green), None, false),
            &color_spec(Some(Green), None, true),
            diff_buffer.iter().map(|s| (s.y0, s.len)),
            out,
        )?;
        added_lines.clear();
        removed_lines.clear();
        Ok(())
    }

    fn push_added(&mut self, line: &[u8]) {
        add_raw_line(&mut self.added_lines, line)
    }

    fn push_removed(&mut self, line: &[u8]) {
        add_raw_line(&mut self.removed_lines, line)
    }
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

#[derive(Debug)]
pub struct Tokenization<'a> {
    data: &'a [u8],
    tokens: &'a [(usize, usize)],
    start_index: isize,
    one_past_end_index: isize,
}

impl<'a> Tokenization<'a> {
    fn new(data: &'a [u8], tokens: &'a [(usize, usize)]) -> Self {
        Tokenization {
            data,
            tokens,
            start_index: 0,
            one_past_end_index: to_isize(tokens.len()),
        }
    }

    fn split_at(&self, lo: isize, hi: isize) -> (Self, Self) {
        let start = self.start_index;
        let end = self.one_past_end_index;
        assert!(start <= lo);
        assert!(lo <= hi);
        assert!(hi <= end);
        (
            Tokenization {
                one_past_end_index: lo,
                ..*self
            },
            Tokenization {
                start_index: hi,
                ..*self
            },
        )
    }

    fn nb_tokens(&self) -> usize {
        to_usize(self.one_past_end_index - self.start_index)
    }

    fn seq(&self, index: isize) -> &[u8] {
        let (lo, hi) = self.tokens[to_usize(self.start_index + index)];
        &self.data[lo..hi]
    }
}

type Tokens<'a> = DiffPair<Tokenization<'a>>;

impl<'a> Tokens<'a> {
    fn split_at(&self, (x0, y0): (isize, isize), (x1, y1): (isize, isize)) -> (Self, Self) {
        let (removed1, removed2) = self.removed.split_at(x0, x1);
        let (added1, added2) = self.added.split_at(y0, y1);

        (
            DiffPair {
                added: added1,
                removed: removed1,
            },
            DiffPair {
                added: added2,
                removed: removed2,
            },
        )
    }

    fn n(&self) -> usize {
        self.removed.nb_tokens()
    }

    fn m(&self) -> usize {
        self.added.nb_tokens()
    }

    fn seq_a(&self, index: isize) -> &[u8] {
        &self.removed.seq(index)
    }

    fn seq_b(&self, index: isize) -> &[u8] {
        &self.added.seq(index)
    }
}

fn tokenize(tokens: &mut Vec<(usize, usize)>, src: &[u8]) {
    tokens.clear();
    let mut lo = 0;
    for (hi, b) in src.iter().enumerate() {
        if !is_alphanum(*b) {
            if lo < hi {
                tokens.push((lo, hi));
            }
            tokens.push((hi, hi + 1));
            lo = hi + 1
        }
    }
    if lo < src.len() {
        tokens.push((lo, src.len()));
    }
}

type Point = (usize, usize);

#[derive(Clone, Debug)]
struct Diff {
    points: Vec<Point>,
}

pub struct DiffTraversal<'a> {
    v: &'a mut [isize],
    max: usize,
    end: (isize, isize),
}

impl<'a> DiffTraversal<'a> {
    fn from_slice(input: &'a Tokens<'a>, v: &'a mut [isize], forward: bool, max: usize) -> Self {
        let start = (input.removed.start_index, input.added.start_index);
        let end = (
            input.removed.one_past_end_index,
            input.added.one_past_end_index,
        );
        assert!(max * 2 + 1 <= v.len());
        let (start, end) = if forward { (start, end) } else { (end, start) };
        let mut res = DiffTraversal { v, max, end };
        if max != 0 {
            *res.v_mut(1) = start.0 - input.removed.start_index
        }
        res
    }

    fn from_vector(
        input: &'a Tokens<'a>,
        v: &'a mut Vec<isize>,
        forward: bool,
        max: usize,
    ) -> Self {
        v.resize(max * 2 + 1, 0);
        Self::from_slice(input, v, forward, max)
    }

    fn v(&self, index: isize) -> isize {
        self.v[to_usize(index + to_isize(self.max))]
    }

    fn v_mut(&mut self, index: isize) -> &mut isize {
        &mut self.v[to_usize(index + to_isize(self.max))]
    }
}

fn diff_sequences_kernel_forward(
    input: &Tokens,
    ctx: &mut DiffTraversal,
    d: usize,
) -> Option<usize> {
    let n = to_isize(input.n());
    let m = to_isize(input.m());
    assert!(d < ctx.max);
    let d = to_isize(d);
    for k in (-d..=d).step_by(2) {
        let mut x = if k == -d || k != d && ctx.v(k - 1) < ctx.v(k + 1) {
            ctx.v(k + 1)
        } else {
            ctx.v(k - 1) + 1
        };
        let mut y = x - k;
        while x < n && y < m && input.seq_a(x) == input.seq_b(y) {
            x += 1;
            y += 1;
        }
        *ctx.v_mut(k) = x;
        if ctx.end == (x, y) {
            return Some(to_usize(d));
        }
    }
    None
}

fn diff_sequences_kernel_backward(
    input: &Tokens,
    ctx: &mut DiffTraversal,
    d: usize,
) -> Option<usize> {
    let n = to_isize(input.n());
    let m = to_isize(input.m());
    let delta = n - m;
    assert!(d < ctx.max);
    let d = to_isize(d);
    for k in (-d..=d).step_by(2) {
        let mut x = if k == -d || k != d && ctx.v(k + 1) < ctx.v(k - 1) {
            ctx.v(k + 1)
        } else {
            ctx.v(k - 1) + 1
        };
        let mut y = x - (k + delta);
        while 0 < x && 0 < y && input.seq_a(x - 1) == input.seq_b(y - 1) {
            x -= 1;
            y -= 1;
        }
        *ctx.v_mut(k) = x - 1;
        if ctx.end == (x, y) {
            return Some(to_usize(d));
        }
    }
    None
}

#[derive(Clone, Debug, Default)]
pub struct Snake {
    x0: isize,
    y0: isize,
    len: isize,
    d: isize,
}

impl Snake {
    fn from(mut self, x0: isize, y0: isize) -> Self {
        self.x0 = x0;
        self.y0 = y0;
        self
    }

    fn len(mut self, len: isize) -> Self {
        self.len = len;
        self
    }

    fn d(mut self, d: isize) -> Self {
        self.d = d;
        self
    }

    fn recenter(mut self, dx: isize, dy: isize) -> Self {
        self.x0 += dx;
        self.y0 += dy;
        self
    }
}

fn diff_sequences_kernel_bidirectional(
    input: &Tokens,
    ctx_fwd: &mut DiffTraversal,
    ctx_bwd: &mut DiffTraversal,
    d: usize,
) -> Option<Snake> {
    let n = to_isize(input.n());
    let m = to_isize(input.m());
    let delta = n - m;
    let odd = delta % 2 != 0;
    assert!(d < ctx_fwd.max);
    assert!(d < ctx_bwd.max);
    let d = to_isize(d);
    for k in (-d..=d).step_by(2) {
        let mut x = if k == -d || k != d && ctx_fwd.v(k - 1) < ctx_fwd.v(k + 1) {
            ctx_fwd.v(k + 1)
        } else {
            ctx_fwd.v(k - 1) + 1
        };
        let mut y = x - k;
        let (x0, y0) = (x, y);
        while x < n && y < m && input.seq_a(x) == input.seq_b(y) {
            x += 1;
            y += 1;
        }
        if odd && (k - delta).abs() <= d - 1 && x > ctx_bwd.v(k - delta) {
            return Some(Snake::default().from(x0, y0).len(x - x0).d(2 * d - 1));
        }
        *ctx_fwd.v_mut(k) = x;
    }
    for k in (-d..=d).step_by(2) {
        let mut x = if k == -d || k != d && ctx_bwd.v(k + 1) < ctx_bwd.v(k - 1) {
            ctx_bwd.v(k + 1)
        } else {
            ctx_bwd.v(k - 1) + 1
        };
        let mut y = x - (k + delta);
        let x1 = x;
        while 0 < x && 0 < y && input.seq_a(x - 1) == input.seq_b(y - 1) {
            x -= 1;
            y -= 1;
        }
        if !odd && (k + delta).abs() <= d && x - 1 < ctx_fwd.v(k + delta) {
            return Some(Snake::default().from(x, y).len(x1 - x).d(2 * d));
        }
        *ctx_bwd.v_mut(k) = x - 1;
    }
    None
}

pub fn diff_sequences_simple_forward(input: &Tokens, v: &mut Vec<isize>) -> usize {
    diff_sequences_simple(input, v, true)
}

pub fn diff_sequences_simple_backward(input: &Tokens, v: &mut Vec<isize>) -> usize {
    diff_sequences_simple(input, v, false)
}

fn diff_sequences_simple(input: &Tokens, v: &mut Vec<isize>, forward: bool) -> usize {
    let max_result = input.n() + input.m();
    let ctx = &mut DiffTraversal::from_vector(input, v, forward, max_result);
    (0..max_result)
        .filter_map(|d| {
            if forward {
                diff_sequences_kernel_forward(input, ctx, d)
            } else {
                diff_sequences_kernel_backward(input, ctx, d)
            }
        })
        .next()
        .unwrap_or(max_result)
}

fn diff(input: &Tokens, v: &mut Vec<isize>, dst: &mut Vec<Snake>) {
    let n = input.n() as isize;
    fn trivial_diff(tok: &Tokenization) -> bool {
        tok.one_past_end_index <= tok.start_index
    }

    if trivial_diff(&input.removed) || trivial_diff(&input.added) {
        return;
    }

    let snake = diff_sequences_bidirectional_snake(input, v);
    let &Snake { x0, y0, len, d } = &snake;
    if 1 < d {
        let (input1, input2) = input.split_at((x0, y0), (x0 + len, y0 + len));
        diff(&input1, v, dst);
        if len != 0 {
            dst.push(snake);
        }
        diff(&input2, v, dst);
    } else {
        let SplittingPoint { sp, dx, dy } = find_splitting_point(&input);
        let x0 = input.removed.start_index;
        let y0 = input.added.start_index;
        if sp != 0 {
            dst.push(Snake::default().from(x0, y0).len(sp));
        }
        let len = n - sp - dx;
        if len != 0 {
            dst.push(Snake::default().from(x0 + sp + dx, y0 + sp + dy).len(len));
        }
    }
}

struct SplittingPoint {
    sp: isize,
    dx: isize,
    dy: isize,
}

// Find the splitting point when two sequences differ by one element.
fn find_splitting_point(input: &Tokens) -> SplittingPoint {
    let n = input.n() as isize;
    let m = input.m() as isize;
    let (short, long, nb_tokens, dx, dy) = if n < m {
        (&input.removed, &input.added, n, 0, 1)
    } else if m < n {
        (&input.added, &input.removed, m, 1, 0)
    } else {
        (&input.added, &input.removed, m, 0, 0)
    };
    let mut sp = nb_tokens;
    for i in 0..nb_tokens {
        if long.seq(i) != short.seq(i) {
            sp = i;
            break;
        }
    }
    SplittingPoint { sp, dx, dy }
}

pub fn diff_sequences_bidirectional(input: &Tokens, v: &mut Vec<isize>) -> usize {
    if input.n() + input.m() == 0 {
        return 0;
    }
    to_usize(diff_sequences_bidirectional_snake(input, v).d)
}

fn diff_sequences_bidirectional_snake(input: &Tokens, v: &mut Vec<isize>) -> Snake {
    let max = (input.n() + input.m() + 1) / 2 + 1;
    let iter_len = 2 * max + 1;
    v.resize(2 * iter_len, 0);

    let (v1, v2) = v.split_at_mut(iter_len);
    let ctx_fwd = &mut DiffTraversal::from_slice(input, v1, true, max);
    let ctx_bwd = &mut DiffTraversal::from_slice(input, v2, false, max);
    (0..max)
        .filter_map(|d| diff_sequences_kernel_bidirectional(input, ctx_fwd, ctx_bwd, d))
        .next()
        .expect("snake not found")
        .recenter(input.removed.start_index, input.added.start_index)
}

fn is_alphanum(b: u8) -> bool {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => true,
        _ => false,
    }
}

fn output<Stream>(buf: &[u8], colorspec: &ColorSpec, out: &mut Stream) -> io::Result<()>
where
    Stream: WriteColor,
{
    let mut i = 0;
    let len = buf.len();
    let ends_with_newline = len != 0 && buf[len - 1] == b'\n';
    let (buf, len) = if ends_with_newline {
        (&buf[..len - 1], len - 1)
    } else {
        (&buf[..], len)
    };
    while i < len {
        out.set_color(colorspec)?;
        let buf = &buf[i..];
        let hi = index_of(buf.iter().cloned(), b'\n').unwrap_or(buf.len() - 1) + 1;
        out.write_all(&buf[..hi])?;
        i += hi;
    }
    out.reset()?;
    if ends_with_newline {
        out.write_all(b"\n")?;
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

fn to_isize(input: usize) -> isize {
    isize::try_from(input).unwrap()
}

fn to_usize(input: isize) -> usize {
    usize::try_from(input).unwrap()
}

#[cfg(test)]
mod test;
