use super::AppConfig;

use clap::App;
use clap::AppSettings;
use clap::Arg;
use clap::ArgMatches;

use termcolor::Color;
use termcolor::ColorSpec;
use termcolor::ParseColorError;

use std::borrow::Cow;
use std::fmt::Display;
use std::fmt::{Error as FmtErr, Formatter};
use std::process::Command;
use std::process::Stdio;
use std::str::FromStr;

const ABOUT: &str = "
diffr adds word-level diff on top of unified diffs.
word-level diff information is displayed using text attributes.";

const USAGE: &str = "\
diffr reads from standard input and write to standard output.

    Typical usage is for interactive use of diff:
    diff -u <file1> <file2> | diffr
    git show | diffr";

const FLAG_DEBUG: &str = "--debug";
const FLAG_COLOR: &str = "--colors";
const FLAG_LINE_NUMBERS: &str = "--line-numbers";
const FLAG_NO_GITCONFIG: &str = "--no-gitconfig";

#[derive(Debug, Clone, Copy)]
enum FaceName {
    Added,
    RefineAdded,
    Removed,
    RefineRemoved,
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

fn parse_color_arg(config: &mut AppConfig, value: &str) -> Result<(), ArgParsingError> {
    let mut pieces = value.split(':');
    if let Some(piece) = pieces.next() {
        let face_name = piece.parse::<FaceName>()?;
        parse_color_attributes(config, pieces, face_name)?;
    }
    Ok(())
}

fn parse_color_args<'a, Values>(
    config: &mut AppConfig,
    values: Values,
) -> Result<(), ArgParsingError>
where
    Values: Iterator<Item = &'a str>,
{
    for value in values {
        parse_color_arg(config, value)?;
    }
    Ok(())
}

fn app() -> App<'static, 'static> {
    App::new("diffr")
        .setting(AppSettings::UnifiedHelpMessage)
        .version("0.1.4")
        .author("Nathan Moreau <nathan.moreau@m4x.org>")
        .about(ABOUT)
        .usage(USAGE)
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
attributes = attribute
           | attribute + ':' + attributes
attribute  = ('foreground' | 'background') + ':' + color
           | (<empty> | 'no') + font-flag
           | 'none'
font-flag  = 'italic'
           | 'bold'
           | 'intense'
           | 'underline'
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
        .arg(
            Arg::with_name(FLAG_LINE_NUMBERS)
                .long(FLAG_LINE_NUMBERS)
                .multiple(true)
                .help("Display line numbers."),
        )
}

fn app_cli() -> App<'static, 'static> {
    app()
        .arg(Arg::with_name(FLAG_DEBUG).long(FLAG_DEBUG).hidden(true))
        .arg(
            Arg::with_name(FLAG_NO_GITCONFIG)
                .long(FLAG_NO_GITCONFIG)
                .help("Ignore settings from the git configuration."),
        )
}

fn die(message: impl Display) -> ! {
    eprintln!("{}", message);
    std::process::exit(-1);
}

fn parse_config_from_cli(config: &mut AppConfig, matches: &ArgMatches<'static>) {
    if matches.is_present(FLAG_DEBUG) {
        config.debug = true;
    }
    if matches.is_present(FLAG_LINE_NUMBERS) {
        config.line_numbers = true;
    }
    if let Some(values) = matches.values_of(FLAG_COLOR) {
        if let Err(err) = parse_color_args(config, values) {
            die(&err);
        }
    }
}

fn remove_prefix<'a>(data: &'a [u8], prefix: &'a [u8]) -> Option<&'a [u8]> {
    if data.starts_with(prefix) {
        Some(&data[prefix.len()..])
    } else {
        None
    }
}

fn parse_config_from_dotgitconfig(config: &mut AppConfig) {
    // run `git config --list` from the same working directory
    let mut cmd = Command::new("git");
    cmd.stdout(Stdio::piped());
    cmd.args(&["config", "--list", "--null"]);
    let stdout = match cmd.spawn() {
        Err(_) => {
            die("could not spawn git process");
        }
        Ok(child) => match child.wait_with_output() {
            Ok(output) => output.stdout,
            Err(_) => die("error waiting for git process output"),
        },
    };

    fn kvp<'a>(value: &'a [u8]) -> Result<(Cow<'a, str>, Cow<'a, str>), &str> {
        let mut pieces = value.split(|b| b == &b'\n');
        let key = pieces.next().ok_or("missing key")?;
        let value = pieces.next().ok_or("missing value")?;
        if pieces.next().is_some() {
            Err("too many values")
        } else {
            Ok((String::from_utf8_lossy(key), String::from_utf8_lossy(value)))
        }
    }
    let kvp = |value| match kvp(value) {
        Err(err) => die(format!("invalid key-value pair: {}", err)),
        Ok(res) => res,
    };
    let add_flag = |mut args: Vec<String>, (ref k, ref v)| {
        if k == "line-numbers" {
            args.push(format!("--line-numbers"));
        } else {
            // let clap emit an error message
            args.push(format!("--{}", k));
            args.push(format!("{}", v));
        }
        args
    };
    let args = stdout
        .split(|b| b == &0)
        .filter_map(|kvp| remove_prefix(kvp, b"diffr."))
        .map(kvp)
        .fold(vec!["diffr".to_string()], add_flag);
    let matches = app().get_matches_from(&args);
    parse_config_from_cli(config, &matches);
}

pub fn parse_config() -> AppConfig {
    let matches = app_cli().get_matches();
    if atty::is(atty::Stream::Stdin) {
        die(matches.usage());
    }

    let mut config = AppConfig::default();
    if !matches.is_present(FLAG_NO_GITCONFIG) {
        parse_config_from_dotgitconfig(&mut config);
    }
    parse_config_from_cli(&mut config, &matches);
    config
}
