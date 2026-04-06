#!/usr/bin/env bash
set -euo pipefail

TEMPLATE_DIR="docs/templates"
OUTPUT_DIR="docs"

run_and_replace() {
    local tag="$1"
    shift
    local cmd=("$@")

    echo "Running [${tag}]: ${cmd[*]}" >&2

    local result
    result=$("${cmd[@]}" 2>/dev/null \
        | sed -n '/^=\{10,\}/,/^-\{10,\}$/p')

    awk -v tag="<!-- BENCH:${tag} -->" -v replacement="$result" '
        index($0, tag) { print replacement; next }
        { print }
    ' "$OUTPUT" > "${OUTPUT}.tmp" && mv "${OUTPUT}.tmp" "$OUTPUT"
}

BENCHMARKS=(
    # Standard
    "annoy:euclidean:gaussian:32|cargo run --example gridsearch_annoy --release -- --distance euclidean"
    "annoy:cosine:gaussian:32|cargo run --example gridsearch_annoy --release -- --distance cosine"
    "annoy:cosine:correlated:32|cargo run --example gridsearch_annoy --release -- --distance euclidean --data correlated"
    "annoy:cosine:correlated:32|cargo run --example gridsearch_annoy --release -- --distance euclidean --data lowrank"
    "annoy:cosine:correlated:32|cargo run --example gridsearch_annoy --release -- --distance euclidean --data lowrank --dim 128"

    # # Binary
    # "binary:correlated:256|cargo run --example gridsearch_binary --release --features binary -- --dim 256 --data correlated --n-samples 50000"
    # "rabitq:lowrank:512|cargo run --example gridsearch_rabitq --release --features binary -- --dim 512 --data lowrank --n-samples 50000 --intrinsic-dim 128"
    # # ...
)

for entry in "${BENCHMARKS[@]}"; do
    tag="${entry%%|*}"
    cmd="${entry#*|}"
    # shellcheck disable=SC2086
    run_and_replace "$tag" $cmd
done
