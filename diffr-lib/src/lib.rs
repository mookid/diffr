//! Algorithms to compute diffs.
//!
//! This module implements various algorithms described in E. Myers
//! paper: [An O(ND) Difference Algorithm and Its
//! Variations](http://www.xmailserver.org/diff2.pdf).
//!
//! The main entrypoint is `diff`, which allows to compute the longest
//! common subsequence between two sequences of byte slices.

use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::fmt::Debug;
use std::fmt::{Error as FmtErr, Formatter};
use std::hash::{Hash, Hasher};

/// A span of bytes and a hash of the content it refers.
#[derive(Clone, Copy, Debug)]
pub struct HashedSpan {
    pub lo: usize,
    pub hi: usize,
    pub hash: u64,
}

/// A wrapper around a token, optimized for equality comparison.
#[derive(PartialEq, Eq)]
pub struct HashedSlice<'a> {
    pub hash: u64,
    pub data: &'a [u8],
}

impl<'a> Debug for HashedSlice<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtErr> {
        let data_pp = String::from_utf8_lossy(self.data);
        f.debug_tuple("HashedSlice").field(&data_pp).finish()
    }
}

impl<'a> Hash for HashedSlice<'a> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        h.write_u64(self.hash)
    }
}

/// A tokenized slice of bytes.
pub struct Tokenization<'a> {
    data: &'a [u8],
    tokens: &'a [HashedSpan],
    start_index: isize,
    one_past_end_index: isize,
}

impl<'a> Debug for Tokenization<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtErr> {
        let Self {
            data,
            tokens,
            start_index,
            one_past_end_index,
        } = self;
        let data_pp = String::from_utf8_lossy(data);
        let tokens_pp = tokens[to_usize(*start_index)..to_usize(*one_past_end_index)]
            .iter()
            .map(|sref| String::from_utf8_lossy(&data[sref.lo..sref.hi]))
            .collect::<Vec<_>>();
        f.debug_struct("Tokenization")
            .field("data", &data_pp)
            .field("tokens", &tokens_pp)
            .finish()
    }
}

impl<'a> Tokenization<'a> {
    pub fn new(data: &'a [u8], tokens: &'a [HashedSpan]) -> Self {
        Tokenization {
            data,
            tokens,
            start_index: 0,
            one_past_end_index: to_isize(tokens.len()),
        }
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Split `self` in two tokenizations:
    /// * the first one from the start to `lo`;
    /// * the second one from `hi` to the end.
    pub fn split_at(&self, lo: isize, hi: isize) -> (Self, Self) {
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

    /// Get `self`'s number of tokens.
    pub fn nb_tokens(&self) -> usize {
        to_usize(self.one_past_end_index - self.start_index)
    }

    /// Get `self`'s `n`th token.
    pub fn nth_token(&self, n: isize) -> HashedSlice {
        let HashedSpan { lo, hi, hash } = self.nth_span(n);
        HashedSlice {
            hash,
            data: &self.data[lo..hi],
        }
    }

    /// Get the span corresponding to `self`'s `n`th token.
    pub fn nth_span(&self, n: isize) -> HashedSpan {
        self.tokens[to_usize(self.start_index + n)]
    }
}

/// A pair of [Tokenization]s to compare.
#[derive(Debug)]
pub struct DiffInput<'a> {
    pub added: Tokenization<'a>,
    pub removed: Tokenization<'a>,
}

impl<'a> DiffInput<'a> {
    fn split_at(&self, (x0, y0): (isize, isize), (x1, y1): (isize, isize)) -> (Self, Self) {
        let (removed1, removed2) = self.removed.split_at(x0, x1);
        let (added1, added2) = self.added.split_at(y0, y1);

        (
            DiffInput {
                added: added1,
                removed: removed1,
            },
            DiffInput {
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
        self.removed.nth_token(index)
    }

    fn seq_b(&self, index: isize) -> HashedSlice {
        self.added.nth_token(index)
    }
}

/// Hash the given bytes.
fn hash_slice(data: &[u8]) -> u64 {
    let mut s = DefaultHasher::new();
    s.write(data);
    s.finish()
}

struct DiffTraversal<'a> {
    v: &'a mut [isize],
    max: usize,
    end: (isize, isize),
}

impl<'a> DiffTraversal<'a> {
    fn from_slice(input: &'a DiffInput<'a>, v: &'a mut [isize], forward: bool, max: usize) -> Self {
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
        input: &'a DiffInput<'a>,
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
    input: &DiffInput,
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
    input: &DiffInput,
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

/// A wrapper around a vector of bytes that keeps track of end of lines.
#[derive(Debug, Default)]
pub struct LineSplit {
    data: Vec<u8>,
    line_lengths: Vec<usize>,
}

impl LineSplit {
    pub fn iter(&self) -> LineSplitIter {
        LineSplitIter {
            line_split: &self,
            index: 0,
            start_of_slice: 0,
        }
    }

    pub fn data<'a>(&'a self) -> &'a [u8] {
        &self.data
    }

    pub fn append_line(&mut self, line: &[u8]) {
        if self.data.last() == Some(&b'\n') {
            self.line_lengths.push(line.len());
        } else {
            match self.line_lengths.last_mut() {
                Some(len) => *len += line.len(),
                None => self.line_lengths.push(line.len()),
            }
        }
        self.data.extend_from_slice(line)
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.line_lengths.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

pub struct LineSplitIter<'a> {
    line_split: &'a LineSplit,
    start_of_slice: usize,
    index: usize,
}

impl<'a> Iterator for LineSplitIter<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let &mut LineSplitIter {
            line_split:
                LineSplit {
                    data: _,
                    line_lengths,
                },
            index,
            start_of_slice,
        } = self;
        if index < line_lengths.len() {
            let len = line_lengths[index];
            self.start_of_slice += len;
            self.index += 1;
            Some((start_of_slice, start_of_slice + len))
        } else {
            None
        }
    }
}

/// A pair of spans with the same content in two different slices.
#[derive(Clone, Debug, Default)]
pub struct Snake {
    /// The start of the span in the removed bytes.
    pub x0: isize,

    /// The start of the span in the added bytes.
    pub y0: isize,

    /// The length of the span.
    pub len: isize,
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
}

fn diff_sequences_kernel_bidirectional(
    input: &DiffInput,
    ctx_fwd: &mut DiffTraversal,
    ctx_bwd: &mut DiffTraversal,
    d: usize,
) -> Option<(Snake, isize)> {
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
            return Some((Snake::default().from(x0, y0).len(x - x0), 2 * d - 1));
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
            return Some((Snake::default().from(x, y).len(x1 - x), 2 * d));
        }
        *ctx_bwd.v_mut(k) = x - 1;
    }
    None
}

/// Compute the length of the edit script for `input`.
/// This is the forward version.
pub fn diff_sequences_simple_forward(input: &DiffInput, v: &mut Vec<isize>) -> usize {
    diff_sequences_simple(input, v, true)
}

/// Compute the length of the edit script for `input`.
/// This is the backward version.
pub fn diff_sequences_simple_backward(input: &DiffInput, v: &mut Vec<isize>) -> usize {
    diff_sequences_simple(input, v, false)
}

fn diff_sequences_simple(input: &DiffInput, v: &mut Vec<isize>, forward: bool) -> usize {
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

/// Compute the longest common subsequence for `input` into `dst`.
pub fn diff(input: &DiffInput, v: &mut Vec<isize>, dst: &mut Vec<Snake>) {
    dst.clear();
    diff_rec(input, v, dst)
}

fn diff_rec(input: &DiffInput, v: &mut Vec<isize>, dst: &mut Vec<Snake>) {
    let n = to_isize(input.n());
    fn trivial_diff(tok: &Tokenization) -> bool {
        tok.one_past_end_index <= tok.start_index
    }

    if trivial_diff(&input.removed) || trivial_diff(&input.added) {
        return;
    }

    let (snake, d) = diff_sequences_bidirectional_snake(input, v);
    let &Snake { x0, y0, len } = &snake;
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
fn find_splitting_point(input: &DiffInput) -> SplittingPoint {
    let n = to_isize(input.n());
    let m = to_isize(input.m());
    let (short, long, nb_tokens, dx, dy) = if n < m {
        (&input.removed, &input.added, n, 0, 1)
    } else if m < n {
        (&input.added, &input.removed, m, 1, 0)
    } else {
        (&input.added, &input.removed, m, 0, 0)
    };
    let mut sp = nb_tokens;
    for i in 0..nb_tokens {
        if long.nth_token(i) != short.nth_token(i) {
            sp = i;
            break;
        }
    }
    SplittingPoint { sp, dx, dy }
}

/// Compute the length of the edit script for `input`.
/// This is the bidirectional version.
pub fn diff_sequences_bidirectional(input: &DiffInput, v: &mut Vec<isize>) -> usize {
    if input.n() + input.m() == 0 {
        return 0;
    }
    to_usize(diff_sequences_bidirectional_snake(input, v).1)
}

fn diff_sequences_bidirectional_snake(input: &DiffInput, v: &mut Vec<isize>) -> (Snake, isize) {
    let max = (input.n() + input.m() + 1) / 2 + 1;
    let iter_len = 2 * max + 1;
    v.resize(2 * iter_len, 0);

    let (v1, v2) = v.split_at_mut(iter_len);
    let ctx_fwd = &mut DiffTraversal::from_slice(input, v1, true, max);
    let ctx_bwd = &mut DiffTraversal::from_slice(input, v2, false, max);
    let mut result = (0..max)
        .filter_map(|d| diff_sequences_kernel_bidirectional(input, ctx_fwd, ctx_bwd, d))
        .next()
        .expect("snake not found");
    result.0.x0 += input.removed.start_index;
    result.0.y0 += input.added.start_index;
    result
}

fn to_isize(input: usize) -> isize {
    isize::try_from(input).unwrap()
}

fn to_usize(input: isize) -> usize {
    usize::try_from(input).unwrap()
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum TokenKind {
    Other,
    Word,
    Spaces,
}

/// Tokenize data from `src` from the position `ofs` into `tokens`.
pub fn tokenize(src: &[u8], ofs: usize, tokens: &mut Vec<HashedSpan>) {
    let mut push = |lo: usize, hi: usize| {
        if lo < hi {
            tokens.push(HashedSpan {
                lo,
                hi,
                hash: hash_slice(&src[lo..hi]),
            })
        }
    };
    let mut lo = ofs;
    let mut kind = TokenKind::Other;
    for hi in ofs..src.len() {
        let oldkind = kind;
        kind = classify_byte(src[hi]);
        if kind != oldkind || oldkind == TokenKind::Other {
            push(lo, hi);
            lo = hi
        }
    }
    push(lo, src.len());
}

fn classify_byte(b: u8) -> TokenKind {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' => TokenKind::Word,
        b'\t' | b' ' => TokenKind::Spaces,
        _ => TokenKind::Other,
    }
}

mod best_projection;
pub use crate::best_projection::optimize_partition;
pub use crate::best_projection::{NormalizationResult, SharedSegments};

#[cfg(test)]
mod test;
