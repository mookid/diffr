use super::AppConfig;
use super::LineNumberStyle;

use std::fmt::Display;
use std::fmt::Error as FmtErr;
use std::fmt::Formatter;
use std::io::Write;
use std::iter::Peekable;
use std::process;
use std::str::FromStr;

use termcolor::Color;
use termcolor::ColorSpec;
use termcolor::ParseColorError;

const FLAG_DEBUG: &str = "--debug";
const FLAG_HTML: &str = "--html";
const FLAG_COLOR: &str = "--colors";
const FLAG_LINE_NUMBERS: &str = "--line-numbers";

const BIN_NAME: &str = env!("CARGO_PKG_NAME");
const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = include_str!("../assets/usage.txt");
const HELP_SHORT: &str = include_str!("../assets/h.txt");
const HELP_LONG: &str = include_str!("../assets/help.txt");

fn show_version() -> ! {
    eprintln!("{} {}", BIN_NAME, VERSION);
    process::exit(0);
}

#[derive(Debug, Clone, Copy)]
enum FaceName {
    Added,
    RefineAdded,
    Removed,
    RefineRemoved,
}

fn missing_arg(arg: impl std::fmt::Display) -> ! {
    eprintln!("option requires an argument: '{}'", arg);
    process::exit(2);
}

fn interpolate(s: &str) -> String {
    s.replace("$VERSION", VERSION)
}

fn usage(code: i32) -> ! {
    let txt = interpolate(USAGE);
    let _ = std::io::stderr().write(txt.as_bytes());
    process::exit(code);
}

fn help(long: bool) -> ! {
    let txt = if long { HELP_LONG } else { HELP_SHORT };
    let txt = interpolate(txt);
    let _ = std::io::stdout().write(txt.as_bytes());
    process::exit(0);
}

impl EnumString for FaceName {
    fn data() -> &'static [(&'static str, Self)] {
        use FaceName::*;
        &[
            ("added", Added),
            ("refine-added", RefineAdded),
            ("removed", Removed),
            ("refine-removed", RefineRemoved),
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
    fn get_face_mut<'a, 'b>(&'a self, config: &'b mut super::AppConfig) -> &'b mut ColorSpec {
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
                Err(err) => Err(ArgParsingError::Color(err)),
            }
        }
    }
}

trait EnumString: Copy {
    fn data() -> &'static [(&'static str, Self)];
}

fn tryparse<T>(input: &str) -> Result<T, String>
where
    T: EnumString + 'static,
{
    T::data()
        .iter()
        .find(|p| p.0 == input)
        .map(|&p| p.1)
        .ok_or_else(|| {
            format!(
                "got '{}', expected {}",
                input,
                T::data().iter().map(|p| p.0).collect::<Vec<_>>().join("|")
            )
        })
}

#[derive(Debug, Clone, Copy)]
struct LineNumberStyleOpt(LineNumberStyle);

impl EnumString for LineNumberStyleOpt {
    fn data() -> &'static [(&'static str, Self)] {
        use LineNumberStyle::*;
        &[
            ("aligned", LineNumberStyleOpt(Aligned)),
            ("compact", LineNumberStyleOpt(Compact)),
        ]
    }
}

#[derive(Debug, Clone, Copy)]
enum FaceColor {
    Foreground,
    Background,
}

#[derive(Debug, Clone, Copy)]
enum AttributeName {
    Color(FaceColor),
    Italic(bool),
    Bold(bool),
    Intense(bool),
    Underline(bool),
    Reset,
}

impl EnumString for AttributeName {
    fn data() -> &'static [(&'static str, Self)] {
        use AttributeName::*;
        &[
            ("foreground", Color(FaceColor::Foreground)),
            ("background", Color(FaceColor::Background)),
            ("italic", Italic(true)),
            ("noitalic", Italic(false)),
            ("bold", Bold(true)),
            ("nobold", Bold(false)),
            ("intense", Intense(true)),
            ("nointense", Intense(false)),
            ("underline", Underline(true)),
            ("nounderline", Underline(false)),
            ("none", Reset),
        ]
    }
}

#[derive(Debug)]
enum ArgParsingError {
    FaceName(String),
    AttributeName(String),
    Color(ParseColorError),
    MissingValue(FaceName),
    LineNumberStyle(String),
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
            ArgParsingError::LineNumberStyle(err) => {
                write!(f, "unexpected line number style: {}", err)
            }
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

impl FromStr for LineNumberStyleOpt {
    type Err = ArgParsingError;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        tryparse(input).map_err(ArgParsingError::LineNumberStyle)
    }
}

fn ignore<T>(_: T) {}

fn parse_line_number_style<'a>(
    config: &mut AppConfig,
    value: Option<&'a str>,
) -> Result<(), ArgParsingError> {
    let style = if let Some(style) = value {
        style.parse::<LineNumberStyleOpt>()?.0
    } else {
        LineNumberStyle::Compact
    };
    config.line_numbers_style = Some(style);
    Ok(())
}

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
            Color(kind) => {
                if let Some(value) = values.next() {
                    let ColorOpt(color) = value.parse::<ColorOpt>()?;
                    match kind {
                        FaceColor::Foreground => face.set_fg(color),
                        FaceColor::Background => face.set_bg(color),
                    };
                } else {
                    return Err(ArgParsingError::MissingValue(face_name));
                }
            }
            Italic(italic) => ignore(face.set_italic(italic)),
            Bold(bold) => ignore(face.set_bold(bold)),
            Intense(intense) => ignore(face.set_intense(intense)),
            Underline(underline) => ignore(face.set_underline(underline)),
            Reset => *face = Default::default(),
        }
    }
    Ok(())
}

fn parse_color_arg(value: &str, config: &mut AppConfig) -> Result<(), ArgParsingError> {
    let mut pieces = value.split(':');
    Ok(if let Some(piece) = pieces.next() {
        let face_name = piece.parse::<FaceName>()?;
        parse_color_attributes(config, pieces, face_name)?;
    })
}

fn die_error<TRes>(result: Result<TRes, ArgParsingError>) -> bool {
    if let Err(err) = result {
        eprintln!("{}", err);
        process::exit(-1);
    }
    true
}

fn color(config: &mut AppConfig, args: &mut Peekable<impl Iterator<Item = String>>) -> bool {
    if let Some(spec) = args.peek() {
        let parse_result = parse_color_arg(&spec, config);
        args.next();
        die_error(parse_result)
    } else {
        missing_arg(FLAG_COLOR)
    }
}

fn line_numbers(config: &mut AppConfig, args: &mut Peekable<impl Iterator<Item = String>>) -> bool {
    let spec = if let Some(spec) = args.peek() {
        if spec.starts_with("-") { // next option
            parse_line_number_style(config, None)
        } else {
            let parse_result = parse_line_number_style(config, Some(&*spec));
            args.next();
            parse_result
        }
    } else {
        parse_line_number_style(config, None)
    };
    die_error(spec)
}

fn html(config: &mut AppConfig, args: &mut Peekable<impl Iterator<Item = String>>) -> bool {
    config.html = true;
    true
}

fn debug(config: &mut AppConfig, args: &mut Peekable<impl Iterator<Item = String>>) -> bool {
    config.debug = true;
    true
}

fn bad_arg(arg: &str) -> ! {
    eprintln!("bad argument: '{}'", arg);
    usage(2);
}

fn parse_options(
    config: &mut AppConfig,
    args: &mut Peekable<impl Iterator<Item = String>>,
) -> bool {
    if let Some(arg) = args.next() {
        match &arg[..] {
            // generic flags
            "-h" | "--help" => help(&arg[..] == "--help"),
            "-V" | "--version" => show_version(),

            // documented flags
            FLAG_COLOR => color(config, args),
            FLAG_LINE_NUMBERS => line_numbers(config, args),

            // hidden flags
            FLAG_DEBUG => debug(config, args),
            FLAG_HTML => html(config, args),

            arg => bad_arg(arg),
        }
    } else {
        false
    }
}

pub fn parse_config() -> AppConfig {
    let mut config = AppConfig::default();
    let mut args = std::env::args().skip(1).peekable();
    while parse_options(&mut config, &mut args) {}

    if atty::is(atty::Stream::Stdin) {
        usage(-1);
    }
    config
}
