use clap::{App, Arg};
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::fmt::Display;
use std::fmt::{Error as FmtErr, Formatter};
use std::hash::Hasher;
use std::io::{self, BufRead};
use std::str::FromStr;
use std::time::SystemTime;
use termcolor::{
    Color,
    Color::{Green, Red},
    ColorChoice, ColorSpec, StandardStream, WriteColor,
};

const ABOUT: &str = "
diffr adds word-level diff on top of unified diffs.
word-level diff information is displayed using text attributes.";

const USAGE: &str = "
    diffr reads from standard input and write to standard output.

    Typical usage is for interactive use of diff:
    diff -u <file1> <file2> | diffr
    git show | diffr";

const TEMPLATE: &str = "\
{bin} {version}
{author}
{about}

USAGE:{usage}

OPTIONS:
{unified}";

const FLAG_DEBUG: &str = "--debug";
const FLAG_COLOR: &str = "--colors";

#[derive(Debug)]
struct AppConfig {
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

#[derive(Debug, Clone, Copy)]
enum FaceName {
    Added,
    RefineAdded,
    Removed,
    RefineRemoved,
}

impl EnumString for FaceName {
    fn data() -> &'static [(Self, &'static str)] {
        use FaceName::*;
        &[
            (Added, "added"),
            (RefineAdded, "refine-added"),
            (Removed, "removed"),
            (RefineRemoved, "refine-removed"),
        ]
    }
}

impl Display for FaceName {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtErr> {
        use FaceName::*;
        match self {
            Added => write!(f, "added"),
            RefineAdded => write!(f, "refine-added"),
            Removed => write!(f, "removed"),
            RefineRemoved => write!(f, "refine-removed"),
        }
    }
}

impl FaceName {
    fn get_face_mut<'a, 'b>(&'a self, config: &'b mut AppConfig) -> &'b mut ColorSpec {
        use FaceName::*;
        match self {
            Added => &mut config.added_face,
            RefineAdded => &mut config.refine_added_face,
            Removed => &mut config.removed_face,
            RefineRemoved => &mut config.refine_removed_face,
        }
    }
}

// custom parsing of Option<Color>
struct ColorOpt(Option<Color>);

impl FromStr for ColorOpt {
    type Err = ArgParsingError;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        if input == "none" {
            Ok(ColorOpt(None))
        } else {
            match input.parse() {
                Ok(color) => Ok(ColorOpt(Some(color))),
                Err(err) => Err(ArgParsingError::Color(format!("{}", err))),
            }
        }
    }
}

trait EnumString: Copy {
    fn data() -> &'static [(Self, &'static str)];
}

fn join<'a, It>(it: It, sep: &'a str) -> String
where
    It: Iterator<Item = &'a str>,
{
    let mut res = String::new();
    let mut first = true;
    for value in it {
        if first {
            first = false;
        } else {
            res += sep;
        }
        res += value
    }
    res
}

fn tryparse<T>(input: &str) -> Result<T, String>
where
    T: EnumString + 'static,
{
    T::data()
        .iter()
        .find(|p| p.1 == input)
        .map(|&p| p.0)
        .ok_or_else(|| {
            format!(
                "got '{}', expected {}",
                input,
                join(T::data().iter().map(|p| p.1), "|")
            )
        })
}

#[derive(Debug, Clone, Copy)]
enum AttributeName {
    Foreground,
    Background,
    Bold,
    NoBold,
    Intense,
    NoIntense,
    Underline,
    NoUnderline,
    Reset,
}

impl EnumString for AttributeName {
    fn data() -> &'static [(Self, &'static str)] {
        use AttributeName::*;
        &[
            (Foreground, "foreground"),
            (Background, "background"),
            (Bold, "bold"),
            (NoBold, "nobold"),
            (Intense, "intense"),
            (NoIntense, "nointense"),
            (Underline, "underline"),
            (NoUnderline, "nounderline"),
            (Reset, "none"),
        ]
    }
}

#[derive(Debug)]
enum ArgParsingError {
    FaceName(String),
    AttributeName(String),
    Color(String),
    MissingValue(FaceName),
    Unknown,
}

impl Display for ArgParsingError {
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtErr> {
        match self {
            ArgParsingError::FaceName(err) => write!(f, "unexpected face name: {}", err),
            ArgParsingError::AttributeName(err) => write!(f, "unexpected attribute name: {}", err),
            ArgParsingError::Color(err) => write!(f, "unexpected color value: {}", err),
            ArgParsingError::MissingValue(face_name) => write!(
                f,
                "error parsing color: missing color value for face '{}'",
                face_name
            ),
            ArgParsingError::Unknown => write!(f, "Internal error"),
        }
    }
}

impl FromStr for FaceName {
    type Err = ArgParsingError;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        tryparse(input).map_err(ArgParsingError::FaceName)
    }
}

impl FromStr for AttributeName {
    type Err = ArgParsingError;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        tryparse(input).map_err(ArgParsingError::AttributeName)
    }
}

fn main() {
    let matches = App::new("diffr")
        .version("0.1.0")
        .author("Nathan Moreau <nathan.moreau@m4x.org>")
        .about(ABOUT)
        .usage(USAGE)
        .template(TEMPLATE)
        .arg(Arg::with_name(FLAG_DEBUG).long(FLAG_DEBUG).hidden(true))
        .arg(
            Arg::with_name(FLAG_COLOR)
                .long(FLAG_COLOR)
                .value_name("COLOR_SPEC")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1)
                .help("Configure color settings.")
                .long_help(
                    "Configure color settings for console ouput.

There are four faces to customize:
+----------------+--------------+----------------+
|  line prefix   |      +       |       -        |
+----------------+--------------+----------------+
| common segment |    added     |    removed     |
| unique segment | refine-added | refine-removed |
+----------------+--------------+----------------+

The customization allows
- to change the foreground or background color;
- to set or unset the attributes 'bold', 'intense', 'underline';
- to clear all attributes.

Customization is done passing a color_spec argument.
This flag may be provided multiple times.

The syntax is the following:

color_spec = face-name + ':' + attributes
attributes = <empty>
           | attribute + ':' + attributes
attribute  = ('foreground' | 'background') + ':' + color
           | (<empty> | 'no') + ('bold' | 'intense' | 'underline')
           | 'none'
color      = 'none'
           | [0-255]
           | [0-255] + ',' + [0-255] + ',' + [0-255]
           | ('black', 'blue', 'green', 'red',
              'cyan', 'magenta', 'yellow', 'white')

For example, the color_spec

    'refine-added:background:blue:bold'

sets the color of unique added segments with
a blue background, written with a bold font.",
                ),
        )
        .get_matches();

    let mut config = AppConfig::default();

    if let Some(values) = matches.values_of(FLAG_COLOR) {
        if let Err(err) = parse_color_args(&mut config, values) {
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

fn parse_color_args<'a, Values>(
    config: &mut AppConfig,
    values: Values,
) -> Result<(), ArgParsingError>
where
    Values: Iterator<Item = &'a str>,
{
    for value in values {
        let mut pieces = value.split(':');
        if let Some(piece) = pieces.next() {
            let face_name = piece.parse::<FaceName>()?;
            parse_color_attributes(config, pieces, face_name)?;
        }
    }
    Ok(())
}

fn ignore<T>(_: T) {}

fn parse_color_attributes<'a, Values>(
    config: &mut AppConfig,
    mut values: Values,
    face_name: FaceName,
) -> Result<(), ArgParsingError>
where
    Values: Iterator<Item = &'a str>,
{
    use AttributeName::*;
    let face = face_name.get_face_mut(config);
    while let Some(value) = values.next() {
        let attribute_name = value.parse::<AttributeName>()?;
        match attribute_name {
            Foreground | Background => {
                if let Some(value) = values.next() {
                    let ColorOpt(color) = value.parse::<ColorOpt>()?;
                    match attribute_name {
                        Foreground => face.set_fg(color),
                        Background => face.set_bg(color),
                        _ => return Err(ArgParsingError::Unknown),
                    };
                } else {
                    return Err(ArgParsingError::MissingValue(face_name));
                }
            }
            Bold => ignore(face.set_bold(true)),
            NoBold => ignore(face.set_bold(false)),
            Intense => ignore(face.set_intense(true)),
            NoIntense => ignore(face.set_intense(false)),
            Underline => ignore(face.set_underline(true)),
            NoUnderline => ignore(face.set_underline(false)),
            Reset => *face = Default::default(),
        }
    }
    Ok(())
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

#[derive(Debug, Default)]
pub struct DiffPair<T> {
    added: T,
    removed: T,
}

#[derive(Default)]
struct HunkBuffer {
    v: Vec<isize>,
    diff_buffer: Vec<Snake>,
    added_tokens: Vec<HashedSliceRef>,
    removed_tokens: Vec<HashedSliceRef>,
    lines: LineSplit,
    config: AppConfig,
}

fn color_spec(fg: Option<Color>, bg: Option<Color>, bold: bool) -> ColorSpec {
    let mut colorspec: ColorSpec = ColorSpec::default();
    colorspec.set_fg(fg);
    colorspec.set_bg(bg);
    colorspec.set_bold(bold);
    colorspec
}

impl HunkBuffer {
    // Returns the number of printed shared tokens
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
            nshared += 1;
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
        let tokens = Tokens {
            removed: Tokenization::new(&lines.data, removed_tokens),
            added: Tokenization::new(&lines.data, added_tokens),
        };
        diff(&tokens, v, diff_buffer);
        let data = &lines.data;
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
                    let data = toks.data;
                    let shared = diff_buffer
                        .iter()
                        .skip(*i)
                        // .filter(|s| s.len != 0)
                        .map(|s| {
                            let x0 = if is_plus { s.y0 } else { s.x0 };
                            let (first, _) = toks.seq_index(x0);
                            let (_, last) = toks.seq_index(x0 + s.len - 1);
                            (first.max(line_start), last.min(line_end))
                        })
                        // .filter(|(first, last)| first < last)
                        .take_while(|xy| xy.0 <= line_end);
                    *i += Self::paint_line(
                        &data,
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
        let ofs = self.lines.len();
        add_raw_line(&mut self.lines, line);
        tokenize(
            ofs,
            if added {
                &mut self.added_tokens
            } else {
                &mut self.removed_tokens
            },
            &self.lines.data[ofs..],
        );
    }
}

// Scan buf looking for target, returning the index of its first
// appearance.
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

#[derive(Debug)]
pub struct Tokenization<'a> {
    data: &'a [u8],
    tokens: &'a [HashedSliceRef],
    start_index: isize,
    one_past_end_index: isize,
}

impl<'a> Tokenization<'a> {
    fn new(data: &'a [u8], tokens: &'a [HashedSliceRef]) -> Self {
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

    fn seq(&self, index: isize) -> HashedSlice {
        let (lo, hi, h) = &self.tokens[to_usize(self.start_index + index)];
        HashedSlice {
            hash: *h,
            data: &self.data[*lo..*hi],
        }
    }

    fn seq_index(&self, index: isize) -> (usize, usize) {
        let (lo, hi, _) = self.tokens[to_usize(self.start_index + index)];
        (lo, hi)
    }
}

type Tokens<'a> = DiffPair<Tokenization<'a>>;

#[derive(PartialEq, Debug)]
struct HashedSlice<'a> {
    hash: u64,
    data: &'a [u8],
}

fn hash_slice(data: &[u8]) -> u64 {
    let mut s = DefaultHasher::new();
    s.write(data);
    s.finish()
}

type HashedSliceRef = (usize, usize, u64);

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

    fn seq_a(&self, index: isize) -> HashedSlice {
        self.removed.seq(index)
    }

    fn seq_b(&self, index: isize) -> HashedSlice {
        self.added.seq(index)
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum TokenKind {
    Other,
    Word,
    Spaces,
}

fn tokenize(ofs: usize, tokens: &mut Vec<HashedSliceRef>, src: &[u8]) {
    let mut push = |lo: usize, hi: usize| {
        if lo < hi {
            tokens.push((ofs + lo, ofs + hi, hash_slice(&src[lo..hi])))
        }
    };
    let mut lo = 0;
    let mut kind = TokenKind::Other;
    for (hi, b) in src.iter().enumerate() {
        let oldkind = kind;
        kind = classify_byte(*b);
        if kind != oldkind || oldkind == TokenKind::Other {
            push(lo, hi);
            lo = hi
        }
    }
    push(lo, src.len());
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

pub fn diff(input: &Tokens, v: &mut Vec<isize>, dst: &mut Vec<Snake>) {
    dst.clear();
    diff_rec(input, v, dst)
}

fn diff_rec(input: &Tokens, v: &mut Vec<isize>, dst: &mut Vec<Snake>) {
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
        diff_rec(&input1, v, dst);
        if len != 0 {
            dst.push(snake);
        }
        diff_rec(&input2, v, dst);
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

fn classify_byte(b: u8) -> TokenKind {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => TokenKind::Word,
        b'\t' | b' ' => TokenKind::Spaces,
        _ => TokenKind::Other,
    }
}

fn output<Stream>(buf: &[u8], colorspec: &ColorSpec, out: &mut Stream) -> io::Result<()>
where
    Stream: WriteColor,
{
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

fn to_isize(input: usize) -> isize {
    isize::try_from(input).unwrap()
}

fn to_usize(input: isize) -> usize {
    usize::try_from(input).unwrap()
}

#[derive(Debug, Default)]
struct LineSplit {
    data: Vec<u8>,
    line_lens: Vec<usize>,
}

impl LineSplit {
    fn iter(&self) -> LineSplitIter {
        LineSplitIter {
            line_split: &self,
            index: 0,
            start_of_slice: 0,
        }
    }

    fn append_line(&mut self, line: &[u8]) {
        if self.data.last().cloned() == Some(b'\n') {
            self.line_lens.push(line.len());
        } else {
            match self.line_lens.last_mut() {
                Some(len) => *len += line.len(),
                None => self.line_lens.push(line.len()),
            }
        }
        self.data.extend_from_slice(line)
    }

    fn clear(&mut self) {
        self.data.clear();
        self.line_lens.clear();
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

struct LineSplitIter<'a> {
    line_split: &'a LineSplit,
    start_of_slice: usize,
    index: usize,
}

impl<'a> Iterator for LineSplitIter<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let &mut LineSplitIter {
            line_split: LineSplit { data: _, line_lens },
            index,
            start_of_slice,
        } = self;
        if index < line_lens.len() {
            let len = line_lens[index];
            self.start_of_slice += len;
            self.index += 1;
            Some((start_of_slice, start_of_slice + len))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test;

#[cfg(test)]
mod test_cli;
