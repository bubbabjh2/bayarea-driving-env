#!/usr/bin/env bash
# Workspace check before every push to `rust`: all crates and targets build, all tests pass (tests needing oracle
# dumps or a GPU skip themselves). Install as a git hook: ln -sf ../../rust/tools/check.sh .git/hooks/pre-push
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/rust"
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --workspace --all-targets --exclude bayenv-py
cargo check --release -p bayenv-py --all-targets
cargo test --release --workspace --exclude bayenv-py -q
echo "check.sh: OK"
