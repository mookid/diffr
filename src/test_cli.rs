use std::path::PathBuf;
use std::process::{Command, Stdio};
use StringTest::*;

enum StringTest {
    Empty,
    AtLeast(&'static str),
    Exactly(&'static str),
}

fn quote_or_empty(msg: &str) -> String {
    if msg.is_empty() {
        "<empty>".to_owned()
    } else {
        format!("\"{}\"", msg)
    }
}

impl StringTest {
    fn test(&self, actual: &str, prefix: &str) {
        match self {
            Empty => assert!(
                actual.is_empty(),
                format!(
                    "{}: expected empty, got\n\n{}",
                    quote_or_empty(prefix),
                    quote_or_empty(actual)
                )
            ),
            AtLeast(exp) => assert!(
                actual.contains(exp),
                format!(
                    "{}: expected at least\n\n{}\n\ngot\n\n{}",
                    prefix,
                    quote_or_empty(exp),
                    quote_or_empty(actual)
                )
            ),
            Exactly(exp) => assert!(
                actual.trim() == exp.trim(),
                format!(
                    "{}: expected\n\n{}\n\ngot\n\n{}",
                    prefix,
                    quote_or_empty(exp),
                    quote_or_empty(actual)
                )
            ),
        }
    }
}

struct ProcessTest {
    args: &'static [&'static str],
    out: StringTest,
    err: StringTest,
    is_success: bool,
}

fn diffr_path() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("target");
    dir.push("debug");
    dir.push("diffr");
    dir
}

fn test_cli(descr: ProcessTest) {
    let mut cmd = Command::new(diffr_path());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    cmd.stdin(Stdio::piped());
    for arg in descr.args {
        cmd.arg(&*arg);
    }
    let child = cmd.spawn().expect("spawn");
    let output = child.wait_with_output().expect("wait_with_output");
    fn string_of_status(code: bool) -> &'static str {
        if code {
            "success"
        } else {
            "failure"
        }
    };
    assert!(
        descr.is_success == output.status.success(),
        format!(
            "unexpected status: expected {} got {}",
            string_of_status(descr.is_success),
            string_of_status(output.status.success()),
        )
    );
    descr
        .out
        .test(&String::from_utf8_lossy(&output.stdout), "stdout");
    descr
        .err
        .test(&String::from_utf8_lossy(&output.stderr), "stderr");
}

#[test]
fn debug_flag() {
    test_cli(ProcessTest {
        args: &["--debug"],
        out: Empty,
        err: AtLeast("hunk processing time (ms):"),
        is_success: true,
    })
}

#[test]
fn color_invalid_face_name() {
    test_cli(ProcessTest {
        args: &["--colors", "notafacename"],
        out: Empty,
        err: Exactly("unexpected face name: got 'notafacename', expected added|refine-added|removed|refine-removed"),
        is_success: false,
    })
}

#[test]
fn color_only_face_name() {
    test_cli(ProcessTest {
        args: &["--colors", "added"],
        out: Empty,
        err: Exactly(""),
        is_success: true,
    })
}

#[test]
fn color_invalid_attribute_name() {
    test_cli(ProcessTest {
        args: &["--colors", "added:bar"],
        out: Empty,
        err: Exactly("unexpected attribute name: got 'bar', expected foreground|background|bold|nobold|intense|nointense|underline|nounderline|none"),
        is_success: false,
    })
}

#[test]
fn color_invalid_color_value_name() {
    test_cli(ProcessTest {
        args: &["--colors", "added:foreground:baz"],
        out: Empty,
        err: Exactly("unexpected color value: unrecognized color name 'baz'. Choose from: black, blue, green, red, cyan, magenta, yellow, white"),
        is_success: false,
    })
}

#[test]
fn color_invalid_color_value_ansi() {
    test_cli(ProcessTest {
        args: &["--colors", "added:foreground:777"],
        out: Empty,
        err: AtLeast("unexpected color value: unrecognized ansi256 color number"),
        is_success: false,
    })
}

#[test]
fn color_invalid_color_value_rgb() {
    test_cli(ProcessTest {
        args: &["--colors", "added:foreground:0,0,777"],
        out: Empty,
        err: AtLeast("unexpected color value: unrecognized RGB color triple"),
        is_success: false,
    })
}

#[test]
fn color_invalid_color_not_done() {
    test_cli(ProcessTest {
        args: &["--colors", "added:foreground"],
        out: Empty,
        err: Exactly("error parsing color: missing color value for face 'added'"),
        is_success: false,
    })
}

#[test]
fn color_ok() {
    test_cli(ProcessTest {
        args: &["--colors", "added:foreground:0"],
        out: Empty,
        err: Exactly(""),
        is_success: true,
    })
}

#[test]
fn color_ok_multiple() {
    test_cli(ProcessTest {
        args: &[
            "--colors",
            "added:foreground:0",
            "--colors",
            "removed:background:red",
        ],
        out: Empty,
        err: Exactly(""),
        is_success: true,
    })
}
