use std::collections::hash_map::Entry::*;
use std::collections::HashMap;
use std::convert::TryFrom;

use crate::TokenId;
use crate::Tokenization;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy, Hash)]
struct Coord {
    next_lcs: usize,
    next_seq: usize,
}

#[derive(Debug)]
struct Context {
    seq_index: HashMap<TokenId, Vec<usize>>,
}

impl Context {
    fn new<'a>(seq: &'a Tokenization<'a>, lcs: &'a Tokenization<'a>) -> Self {
        let mut seq_index = HashMap::new();
        for v in lcs.tokens() {
            match seq_index.entry(*v) {
                Occupied(_) => (),
                Vacant(e) => {
                    e.insert(vec![]);
                }
            }
        }
        for (i, v) in seq.tokens().iter().enumerate() {
            match seq_index.entry(*v) {
                Occupied(e) => {
                    e.into_mut().push(i);
                }
                Vacant(_) => (),
            }
        }
        Context { seq_index }
    }

    fn get_indexes(&self, tok: TokenId, min_value: usize) -> &[usize] {
        match self.seq_index.get(&tok) {
            Some(values) => {
                let min_idx = match values.binary_search(&min_value) {
                    Ok(i) | Err(i) => i,
                };
                &values[min_idx..]
            }
            None => &[],
        }
    }
}

/// The result of `optimize_partition`. This is mostly used by `shared_segments`.
#[derive(Debug)]
pub struct NormalizationResult {
    pub path: Vec<isize>,
    pub starts_with_shared: bool,
}

impl NormalizationResult {
    /// The shared segments between both inputs of `optimize_partition`.
    /// The `seq` argument is the longest of the two inputs.
    pub fn shared_segments<'a>(&'a self, seq: &'a Tokenization) -> SharedSegments<'a> {
        SharedSegments::new(self, seq)
    }
}

fn snake_len(seq: &Tokenization, lcs: &Tokenization, start_lcs: usize, start_seq: usize) -> usize {
    let lcs_len = lcs.nb_tokens() - start_lcs;
    let seq_len = seq.nb_tokens() - start_seq;
    let max_snake_len = lcs_len.min(seq_len);
    let mut snake_len = 0;
    let seq = &seq.tokens()[start_seq..start_seq + max_snake_len];
    let lcs = &lcs.tokens()[start_lcs..start_lcs + max_snake_len];

    while snake_len < max_snake_len && lcs[snake_len] == seq[snake_len] {
        snake_len += 1
    }
    snake_len
}

/// Minimize the number of elements when partitioning `seq` according to `lcs`.
/// `lcs` is a subsequence of `seq`.
pub fn optimize_partition(seq: &Tokenization, lcs: &Tokenization) -> NormalizationResult {
    let context = Context::new(&seq, &lcs);
    let root = Coord {
        next_lcs: 0,
        next_seq: 0,
    };
    let target = Coord {
        next_lcs: lcs.nb_tokens(),
        next_seq: seq.nb_tokens(),
    };
    let mut frontier = vec![root];
    let mut new_frontier = vec![];
    let mut prev = HashMap::new();
    let mut found_seq = None;
    while !frontier.is_empty() && found_seq == None {
        new_frontier.clear();
        for &coord in frontier.iter() {
            if coord.next_lcs == target.next_lcs {
                found_seq = Some(coord.next_seq);
                if coord.next_seq == target.next_seq {
                    break;
                } else {
                    // TODO do something more clever here
                    continue;
                }
            }
            let start_lcs = coord.next_lcs;
            let lcs_len = lcs.nb_tokens() - start_lcs;
            let mut last_enqueued_snake_len = 0;
            for start_seq in
                context.get_indexes(lcs.nth_token(to_isize(coord.next_lcs)), coord.next_seq)
            {
                if start_seq + lcs_len > seq.nb_tokens() {
                    break;
                }
                let snake_len = 1 + snake_len(&seq, &lcs, start_lcs + 1, start_seq + 1);
                let next_coord = Coord {
                    next_lcs: start_lcs + snake_len,
                    next_seq: start_seq + snake_len,
                };
                if last_enqueued_snake_len < snake_len || next_coord == target {
                    if next_coord.next_lcs == target.next_lcs
                        && (next_coord.next_seq == target.next_seq || found_seq == None)
                    {
                        found_seq = Some(next_coord.next_seq);
                    }
                    match prev.entry(next_coord) {
                        Occupied(_) => continue,
                        Vacant(e) => e.insert(coord),
                    };
                    new_frontier.push(next_coord);
                    last_enqueued_snake_len = snake_len;
                }
            }
        }
        std::mem::swap(&mut frontier, &mut new_frontier)
    }

    let target = found_seq.map(|next_seq| Coord {
        next_lcs: lcs.nb_tokens(),
        next_seq,
    });
    let mut path = vec![];
    let mut starts_with_shared = false;
    let mut coord = target.as_ref();
    let mut seq = seq.nb_tokens();
    let mut lcs = lcs.nb_tokens();
    while let Some(&coord_content) = coord {
        let next_seq = coord_content.next_seq;
        let next_lcs = coord_content.next_lcs;
        let snake_len = lcs - next_lcs;
        push_if_not_last(&mut path, to_isize(seq - snake_len));
        starts_with_shared = !push_if_not_last(&mut path, to_isize(next_seq));

        coord = prev.get(&coord_content);

        seq = next_seq;
        lcs = next_lcs;
    }
    path.reverse();
    NormalizationResult {
        path,
        starts_with_shared,
    }
}

fn push_if_not_last(v: &mut Vec<isize>, val: isize) -> bool {
    let should_push = v.last() != Some(&val);
    if should_push {
        v.push(val);
    }
    should_push
}

fn to_isize(input: usize) -> isize {
    isize::try_from(input).unwrap()
}

/// The shared segments between both inputs of `optimize_partition`.
pub struct SharedSegments<'a> {
    index: usize,
    normalization: &'a Vec<isize>,
    seq: &'a Tokenization<'a>,
}

impl<'a> SharedSegments<'a> {
    fn new(normalization: &'a NormalizationResult, seq: &'a Tokenization) -> Self {
        SharedSegments {
            index: if normalization.starts_with_shared {
                0
            } else {
                1
            },
            normalization: &normalization.path,
            seq,
        }
    }
}

impl<'a> Iterator for SharedSegments<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index + 1 < self.normalization.len() {
            let prev = self.normalization[self.index];
            let curr = self.normalization[self.index + 1];
            let from = self.seq.nth_span(prev).0;
            let to = self.seq.nth_span(curr - 1).1;
            self.index += 2;
            Some((from, to))
        } else {
            None
        }
    }
}
