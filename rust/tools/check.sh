#!/usr/bin/env bash
# Workspace check before every push to `rust`: all crates and targets build, all tests pass (tests needing oracle
# dumps or a GPU skip themselves). Install as a git hook: ln -sf ../../rust/tools/check.sh .git/hooks/pre-push
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/rust"
export PATH="$HOME/.cargo/bin:$PATH"
# Gentle on a shared machine by default: CHECK_NICE (default 10) and CARGO_BUILD_JOBS (default 4) are honoured;
# CHECK_NICE=0 runs un-niced.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
N=(nice -n "${CHECK_NICE:-10}")
"${N[@]}" cargo build --release --workspace --all-targets --exclude bayenv-py
"${N[@]}" cargo check --release -p bayenv-py --all-targets
"${N[@]}" cargo test --release --workspace --exclude bayenv-py -q
echo "check.sh: OK"
