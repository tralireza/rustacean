---
name: lcct
description: Commit a LeetCode solution with cargo test output
---

# lcct — LeetCode Commit

## Title Format

```
<Category> <ProblemNumber><Difficulty> <Problem Name>
```

- **Category**: Module/package name (e.g., `Math`, `String`, `Array`, `DP`, `Priority Queue`)
- **Difficulty**: `m` (medium), `h` (hard), or omitted for easy
- **Example**: `Math 1980m Find Unique Binary String`

## Body

Full output of `cargo test` for the solution, captured with `--nocapture`:

```bash
source ~/.cargo/env && cargo test -p <package> test_<problem_number> -- --nocapture 2>&1
```

The body includes:
1. `Finished` and `Running` lines from cargo
2. Test execution output (debug prints, assertions)
3. Test result summary

## Scaffolding

When creating scaffolding code (struct, impl, test), always run `rustfmt` on the files afterward:

```bash
source ~/.cargo/env && rustfmt <file_path>
```

This ensures consistent formatting and avoids linter-triggered reformats.

## Staging — only the new solution + its test

The commit MUST contain exactly two hunks: the new `SolNNN` block in `<package>/src/lib.rs` and the new `test_NNN` function in `<package>/src/tests.rs`. Any other diff in those files (formatting drift, edits to other tests/structs, unrelated work) MUST be excluded.

Procedure (run in order):

1. **Inspect**: `git diff -- <package>/src/lib.rs <package>/src/tests.rs` — read every hunk.
2. **Clean case** — diff contains only the new `SolNNN` block + new `test_NNN` function:
   - `git add <package>/src/lib.rs <package>/src/tests.rs`
3. **Mixed case** — diff contains unrelated hunks too:
   - Build a patch with only the wanted hunks: write a `.patch` file containing the `diff --git` header plus exactly the hunks for the new struct/impl and the new test function.
   - Apply it to the index: `git apply --cached <file>.patch`
   - Do NOT `git add` the whole file in this case.
4. **Verify**: `git diff --staged` — output must show ONLY the new struct/impl and the new test function. If anything else appears, `git restore --staged <file>` and redo step 3.
5. **Commit** using the command below.

Forbidden: `git add -A`, `git add .`, `git commit -a`, staging an entire file when other hunks are present.

## Commit Command

```bash
git commit -m "$(cat <<'EOF'
<Category> <Number><Difficulty> <Problem Name>

<full cargo test output>
EOF
)"
```

## Author

```
Alireza <26859725+tralireza@users.noreply.github.com>
```

## Example

```
Math 1980m Find Unique Binary String

    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.08s
     Running unittests src/lib.rs (target/debug/deps/math-0b3a49576d88c877)

running 1 test
* ["01", "10"]
:: {"11", "00"}
test tests::test_1980 ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 29 filtered out; finished in 0.00s
```
