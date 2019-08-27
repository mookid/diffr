use super::AppConfig;
use clap::{App, AppSettings, Arg, ArgMatches};
use std::fmt::Display;
use std::fmt::{Error as FmtErr, Formatter};
use std::str::FromStr;
use termcolor::{Color, ColorSpec, ParseColorError};

const ABOUT: &str = "
diffr adds word-level diff on top of unified diffs.
word-level diff information is displayed using text attributes.";

const USAGE: &str = "\
diffr reads from standard input and write to standard output.

    Typical usage is for interactive use of diff:
    diff -u <file1> <file2> | diffr
    git show | diffr";

pub const FLAG_DEBUG: &str = "--debug";
pub const FLAG_COLOR: &str = "--colors";

#[derive(Debug, Clone, Copy)]
pub enum FaceName {
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
pub enum ArgParsingError {
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
            Bold(bold) => ignore(face.set_bold(bold)),
            Intense(intense) => ignore(face.set_intense(intense)),
            Underline(underline) => ignore(face.set_underline(underline)),
            Reset => *face = Default::default(),
        }
    }
    Ok(())
}

pub fn parse_color_args<'a, Values>(
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

pub fn get_matches() -> ArgMatches<'static> {
    App::new("diffr")
        .setting(AppSettings::UnifiedHelpMessage)
        .version("0.1.1")
        .author("Nathan Moreau <nathan.moreau@m4x.org>")
        .about(ABOUT)
        .usage(USAGE)
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
        .get_matches()
}
