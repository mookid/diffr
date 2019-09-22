## diffr_lib

This crate implements various algorithms described in E. Myers
paper: [An O(ND) Difference Algorithm and Its
Variations](http://www.xmailserver.org/diff2.pdf).

The main entrypoint is `diff`, which allows to compute the longest
common subsequence between two sequences of byte slices.

Note that the current API is not stabilized yet.

### Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
diffr-lib="*"
```

```rust
use diffr_lib::{diff, DiffInput, HashedSpan, Tokenization};
use std::collections::HashSet;

fn main() {
    fn line(lo: usize, hi: usize) -> HashedSpan {
        HashedSpan { lo, hi, hash: 0 }
    }

    let old_data = b"I need to buy apples.\n\
I need to run the laundry.\n\
I need to wash the dog.\n\
I need to get the car detailed.";
    let old_tokens = vec![line(0, 21), line(22, 48), line(49, 72), line(73, 104)];

    let new_data = b"I need to buy apples.\n\
I need to do the laundry.\n\
I need to wash the car.\n\
I need to get the dog detailed.";
    let new_tokens = vec![line(0, 21), line(22, 47), line(48, 71), line(72, 103)];

    let input = DiffInput {
        added: Tokenization::new(old_data, &old_tokens),
        removed: Tokenization::new(new_data, &new_tokens),
    };
    let mut namespace = vec![]; // memory used during the diff algorithm
    let mut shared_blocks = vec![]; // results
    diff(&input, &mut namespace, &mut shared_blocks);

    let mut old_shared = HashSet::new();
    let mut new_shared = HashSet::new();

    println!("LCS:");
    for shared_block in &shared_blocks {
        for line_idx in 0..shared_block.len as usize {
            let old_line_index = shared_block.x0 as usize + line_idx;
            let old_line = old_tokens[old_line_index];
            old_shared.insert(old_line_index);
            let new_line_index = shared_block.y0 as usize + line_idx;
            new_shared.insert(new_line_index);
            println!(
                "\t{}",
                String::from_utf8_lossy(&old_data[old_line.lo..old_line.hi])
            );
        }
    }

    println!("unique to old data: ");
    for i in 0..old_tokens.len() {
        if !old_shared.contains(&i) {
            let old_line = old_tokens[i];
            println!(
                "\t{}",
                String::from_utf8_lossy(&old_data[old_line.lo..old_line.hi])
            );
        }
    }

    println!("unique to new data: ");
    for i in 0..new_tokens.len() {
        if !new_shared.contains(&i) {
            let new_line = new_tokens[i];
            println!(
                "\t{}",
                String::from_utf8_lossy(&new_data[new_line.lo..new_line.hi])
            );
        }
    }
}
```
