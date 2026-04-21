#!/usr/bin/env bash
set -euo pipefail

TEMPLATE_DIR="docs/templates"
OUTPUT_DIR="docs"

usage() {
    echo "Usage: $0 --kind <standard|gpu|binary|quantised> [--dry-run]"
    exit 1
}

KIND=""
DRY_RUN=false

while [ $# -gt 0 ]; do
    case "$1" in
        --kind)    KIND="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *)         usage ;;
    esac
done

[ -z "$KIND" ] && usage

TEMPLATE="${TEMPLATE_DIR}/benchmarks_${KIND}.md.tmpl"
OUTPUT="${OUTPUT_DIR}/benchmarks_${KIND}.md"

if [ ! -f "$TEMPLATE" ]; then
    echo "Template not found: $TEMPLATE" >&2
    exit 1
fi

cp "$TEMPLATE" "$OUTPUT"

run_and_replace() {
    local tag="$1"
    shift
    local cmd=("$@")

    if $DRY_RUN; then
        echo "[dry-run] Would run [${tag}]: ${cmd[*]}" >&2
        return
    fi

    echo "Running [${tag}]: ${cmd[*]}" >&2
    local result_file
    result_file=$(mktemp)
    local err_file
    err_file=$(mktemp)

    "${cmd[@]}" 2>"$err_file" \
        | sed -n '/^=\{10,\}/,$p' > "$result_file" || true

    if [ ! -s "$result_file" ]; then
        echo "WARNING: no table output for ${tag}" >&2
        if [ -s "$err_file" ]; then
            echo "--- stderr ---" >&2
            tail -5 "$err_file" >&2
            echo "--- end ---" >&2
        fi
        rm -f "$result_file" "$err_file"
        return
    fi

    rm -f "$err_file"

    perl -pi -e "
        if (/<!-- BENCH:\Q${tag}\E -->/) {
            open my \$fh, '<', '${result_file}' or die;
            local \$/;
            \$_ = <\$fh>;
            close \$fh;
        }
    " "$OUTPUT"

    rm -f "$result_file"
}

case "$KIND" in
    standard)
        BENCHMARKS=(
            # annoy
            "annoy:euclidean:gaussian:32|cargo run --example gridsearch_annoy --release -- --distance euclidean"
            "annoy:cosine:gaussian:32|cargo run --example gridsearch_annoy --release -- --distance cosine"
            "annoy:euclidean:correlated:32|cargo run --example gridsearch_annoy --release -- --distance euclidean --data correlated"
            "annoy:euclidean:lowrank:32|cargo run --example gridsearch_annoy --release -- --distance euclidean --data lowrank"
            "annoy:euclidean:lowrank:128|cargo run --example gridsearch_annoy --release -- --distance euclidean --data lowrank --dim 128"

            # balltree
            "balltree:euclidean:gaussian:32|cargo run --example gridsearch_balltree --release -- --distance euclidean"
            "balltree:cosine:gaussian:32|cargo run --example gridsearch_balltree --release -- --distance cosine"
            "balltree:euclidean:correlated:32|cargo run --example gridsearch_balltree --release -- --distance euclidean --data correlated"
            "balltree:euclidean:lowrank:32|cargo run --example gridsearch_balltree --release -- --distance euclidean --data lowrank"
            "balltree:euclidean:lowrank:128|cargo run --example gridsearch_balltree --release -- --distance euclidean --data lowrank --dim 128"

            # hnsw
            "hnsw:euclidean:gaussian:32|cargo run --example gridsearch_hnsw --release -- --distance euclidean"
            "hnsw:cosine:gaussian:32|cargo run --example gridsearch_hnsw --release -- --distance cosine"
            "hnsw:euclidean:correlated:32|cargo run --example gridsearch_hnsw --release -- --distance euclidean --data correlated"
            "hnsw:euclidean:lowrank:32|cargo run --example gridsearch_hnsw --release -- --distance euclidean --data lowrank"
            "hnsw:euclidean:lowrank:128|cargo run --example gridsearch_hnsw --release -- --distance euclidean --data lowrank --dim 128"

            # ivf
            "ivf:euclidean:gaussian:32|cargo run --example gridsearch_ivf --release -- --distance euclidean"
            "ivf:cosine:gaussian:32|cargo run --example gridsearch_ivf --release -- --distance cosine"
            "ivf:euclidean:correlated:32|cargo run --example gridsearch_ivf --release -- --distance euclidean --data correlated"
            "ivf:euclidean:lowrank:32|cargo run --example gridsearch_ivf --release -- --distance euclidean --data lowrank"
            "ivf:euclidean:lowrank:128|cargo run --example gridsearch_ivf --release -- --distance euclidean --data lowrank --dim 128"

            # kd_forest
            "kd_forest:euclidean:gaussian:32|cargo run --example gridsearch_kd_forest --release -- --distance euclidean"
            "kd_forest:cosine:gaussian:32|cargo run --example gridsearch_kd_forest --release -- --distance cosine"
            "kd_forest:euclidean:correlated:32|cargo run --example gridsearch_kd_forest --release -- --distance euclidean --data correlated"
            "kd_forest:euclidean:lowrank:32|cargo run --example gridsearch_kd_forest --release -- --distance euclidean --data lowrank"
            "kd_forest:euclidean:lowrank:128|cargo run --example gridsearch_kd_forest --release -- --distance euclidean --data lowrank --dim 128"

            # kmknn
            "kmknn:euclidean:gaussian:32|cargo run --example gridsearch_kmknn --release -- --distance euclidean"
            "kmknn:cosine:gaussian:32|cargo run --example gridsearch_kmknn --release -- --distance cosine"
            "kmknn:euclidean:correlated:32|cargo run --example gridsearch_kmknn --release -- --distance euclidean --data correlated"
            "kmknn:euclidean:lowrank:32|cargo run --example gridsearch_kmknn --release -- --distance euclidean --data lowrank"
            "kmknn:euclidean:lowrank:128|cargo run --example gridsearch_kmknn --release -- --distance euclidean --data lowrank --dim 128"

            # lsh
            "lsh:euclidean:gaussian:32|cargo run --example gridsearch_lsh --release -- --distance euclidean"
            "lsh:cosine:gaussian:32|cargo run --example gridsearch_lsh --release -- --distance cosine"
            "lsh:euclidean:correlated:32|cargo run --example gridsearch_lsh --release -- --distance euclidean --data correlated"
            "lsh:euclidean:lowrank:32|cargo run --example gridsearch_lsh --release -- --distance euclidean --data lowrank"
            "lsh:euclidean:lowrank:128|cargo run --example gridsearch_lsh --release -- --distance euclidean --data lowrank --dim 128"

            # nndescent
            "nndescent:euclidean:gaussian:32|cargo run --example gridsearch_nndescent --release -- --distance euclidean"
            "nndescent:cosine:gaussian:32|cargo run --example gridsearch_nndescent --release -- --distance cosine"
            "nndescent:euclidean:correlated:32|cargo run --example gridsearch_nndescent --release -- --distance euclidean --data correlated"
            "nndescent:euclidean:lowrank:32|cargo run --example gridsearch_nndescent --release -- --distance euclidean --data lowrank"
            "nndescent:euclidean:lowrank:128|cargo run --example gridsearch_nndescent --release -- --distance euclidean --data lowrank --dim 128"

             # vamana
             "vamana:euclidean:gaussian:32|cargo run --example gridsearch_vamana --release -- --distance euclidean"
             "vamana:cosine:gaussian:32|cargo run --example gridsearch_vamana --release -- --distance cosine"
             "vamana:euclidean:correlated:32|cargo run --example gridsearch_vamana --release -- --distance euclidean --data correlated"
             "vamana:euclidean:lowrank:32|cargo run --example gridsearch_vamana --release -- --distance euclidean --data lowrank"
             "vamana:euclidean:lowrank:128|cargo run --example gridsearch_vamana --release -- --distance euclidean --data lowrank --dim 128"
        )
        ;;
    gpu)
        BENCHMARKS=(
            # standard patterns
            "gpu:euclidean:gaussian:32|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean"
            "gpu:cosine:gaussian:32|cargo run --example gridsearch_gpu --release --features gpu -- --distance cosine"
            "gpu:euclidean:correlated:32|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data correlated"
            "gpu:euclidean:lowrank:32|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data lowrank"
            "gpu:euclidean:lowrank:128|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data lowrank --dim 128"

            # CPU IVF baselines at larger sizes
            "ivf:euclidean:lowrank:64:250000|cargo run --example gridsearch_ivf --release -- --distance euclidean --data lowrank --n-samples 250000 --dim 64"
            "ivf:euclidean:lowrank:128:250000|cargo run --example gridsearch_ivf --release -- --distance euclidean --data lowrank --n-samples 250000 --dim 128"
            "ivf:euclidean:lowrank:64:500000|cargo run --example gridsearch_ivf --release -- --distance euclidean --data lowrank --n-samples 500000 --dim 64"
            "ivf:euclidean:lowrank:128:500000|cargo run --example gridsearch_ivf --release -- --distance euclidean --data lowrank --n-samples 500000 --dim 128"

            # GPU IVF at larger sizes
            "gpu:euclidean:lowrank:64:250000|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data lowrank --n-samples 250000 --dim 64"
            "gpu:euclidean:lowrank:128:250000|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data lowrank --n-samples 250000 --dim 128"
            "gpu:euclidean:lowrank:64:500000|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data lowrank --n-samples 500000 --dim 64"
            "gpu:euclidean:lowrank:128:500000|cargo run --example gridsearch_gpu --release --features gpu -- --distance euclidean --data lowrank --n-samples 500000 --dim 128"

            # CAGRA

            # standard patterns
            "cagra:euclidean:gaussian:32|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean"
            "cagra:cosine:gaussian:32|cargo run --example gridsearch_cagra --release --features gpu -- --distance cosine"
            "cagra:euclidean:correlated:32|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --data correlated"
            "cagra:euclidean:lowrank:32|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --data lowrank"
            "cagra:euclidean:lowrank:128|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --data lowrank --dim 128"

            # CAGRA at larger sizes
            "cagra:euclidean:lowrank:64:250000|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 250000 --dim 64"
            "cagra:euclidean:lowrank:128:250000|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 250000 --dim 128"
            "cagra:euclidean:lowrank:64:500000|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 500000 --dim 64"
            "cagra:euclidean:lowrank:128:500000|cargo run --example gridsearch_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 500000 --dim 128"

            # CAGRA kNN
            "cagra_knn:euclidean:lowrank:32:250000|cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 250000"
            "cagra_knn:euclidean:lowrank:64:250000|cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 250000 --dim 64"
            "cagra_knn:euclidean:lowrank:32:500000|cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 500000"
            "cagra_knn:euclidean:lowrank:64:500000|cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 500000 --dim 64"
            "cagra_knn:euclidean:lowrank:32:1000000|cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 1000000"
            "cagra_knn:euclidean:lowrank:64:1000000|cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 1000000 --dim 64"
            "cagra_knn:euclidean:lowrank:32:2500000|cargo run --example knn_comparison_cagra --release --features gpu -- --distance euclidean --data lowrank --n-samples 2500000"
        )
        ;;
    binary)
        BENCHMARKS=(
            "binary:euclidean:correlated:256:50000|cargo run --example gridsearch_binary --release --features binary -- --data correlated --n-samples 50000 --dim 256"
            "binary:euclidean:lowrank:256:50000|cargo run --example gridsearch_binary --release --features binary -- --data lowrank --n-samples 50000 --dim 256 --intrinsic-dim 32"
            "binary:euclidean:quantisation:256:50000|cargo run --example gridsearch_binary --release --features binary -- --data quantisation --n-samples 50000 --dim 256 --n-clusters 50"

            "binary:euclidean:correlated:512:50000|cargo run --example gridsearch_binary --release --features binary -- --data correlated --n-samples 50000 --dim 512"
            "binary:euclidean:lowrank:512:50000|cargo run --example gridsearch_binary --release --features binary -- --data lowrank --n-samples 50000 --dim 512 --intrinsic-dim 64"
            "binary:euclidean:quantisation:512:50000|cargo run --example gridsearch_binary --release --features binary -- --data quantisation --n-samples 50000 --dim 512 --n-clusters 50"

            "binary:euclidean:correlated:1024:50000|cargo run --example gridsearch_binary --release --features binary -- --data correlated --n-samples 50000 --dim 1024"
            "binary:euclidean:lowrank:1024:50000|cargo run --example gridsearch_binary --release --features binary -- --data lowrank --n-samples 50000 --dim 1024 --intrinsic-dim 128"
            "binary:euclidean:quantisation:1024:50000|cargo run --example gridsearch_binary --release --features binary -- --data quantisation --n-samples 50000 --dim 1024 --n-clusters 50"

            "rabitq:euclidean:correlated:256:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data correlated --n-samples 50000 --dim 256"
            "rabitq:euclidean:lowrank:256:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data lowrank --n-samples 50000 --dim 256 --intrinsic-dim 32"
            "rabitq:euclidean:quantisation:256:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data quantisation --n-samples 50000 --dim 256 --n-clusters 50"

            "rabitq:euclidean:correlated:512:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data correlated --n-samples 50000 --dim 512"
            "rabitq:euclidean:lowrank:512:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data lowrank --n-samples 50000 --dim 512 --intrinsic-dim 64"
            "rabitq:euclidean:quantisation:512:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data quantisation --n-samples 50000 --dim 512 --n-clusters 50"

            "rabitq:euclidean:correlated:1024:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data correlated --n-samples 50000 --dim 1024"
            "rabitq:euclidean:lowrank:1024:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data lowrank --n-samples 50000 --dim 1024 --intrinsic-dim 128"
            "rabitq:euclidean:quantisation:1024:50000|cargo run --example gridsearch_rabitq --release --features binary -- --data quantisation --n-samples 50000 --dim 1024 --n-clusters 50"
        )
        ;;
    quantised)
        BENCHMARKS=(
            "bf16:euclidean:gaussian:32|cargo run --example gridsearch_bf16 --release --features quantised -- --distance euclidean"
            "bf16:cosine:gaussian:32|cargo run --example gridsearch_bf16 --release --features quantised -- --distance cosine"
            "bf16:euclidean:correlated:32|cargo run --example gridsearch_bf16 --release --features quantised -- --distance euclidean --data correlated"
            "bf16:euclidean:lowrank:32|cargo run --example gridsearch_bf16 --release --features quantised -- --distance euclidean --data lowrank"
            "bf16:euclidean:lowrank:128|cargo run --example gridsearch_bf16 --release --features quantised -- --distance euclidean --data lowrank --dim 128"

            "sq8:euclidean:gaussian:32|cargo run --example gridsearch_sq8 --release --features quantised -- --distance euclidean"
            "sq8:cosine:gaussian:32|cargo run --example gridsearch_sq8 --release --features quantised -- --distance cosine"
            "sq8:euclidean:correlated:32|cargo run --example gridsearch_sq8 --release --features quantised -- --distance euclidean --data correlated"
            "sq8:euclidean:lowrank:32|cargo run --example gridsearch_sq8 --release --features quantised -- --distance euclidean --data lowrank"
            "sq8:euclidean:lowrank:128|cargo run --example gridsearch_sq8 --release --features quantised -- --distance euclidean --data lowrank --dim 128"

            "pq:euclidean:correlated:128:50000|cargo run --example gridsearch_pq --release --features quantised -- --data correlated --n-samples 50000 --dim 128"
            "pq:euclidean:lowrank:128:50000|cargo run --example gridsearch_pq --release --features quantised -- --data lowrank --n-samples 50000 --dim 128"
            "pq:euclidean:quantisation:128:50000|cargo run --example gridsearch_pq --release --features quantised -- --data quantisation --n-samples 50000 --dim 128 --n-clusters 50"

            "pq:euclidean:correlated:256:50000|cargo run --example gridsearch_pq --release --features quantised -- --data correlated --n-samples 50000 --dim 256"
            "pq:euclidean:lowrank:256:50000|cargo run --example gridsearch_pq --release --features quantised -- --data lowrank --n-samples 50000 --dim 256 --intrinsic-dim 32"
            "pq:euclidean:quantisation:256:50000|cargo run --example gridsearch_pq --release --features quantised -- --data quantisation --n-samples 50000 --dim 256 --n-clusters 50"

            "pq:euclidean:correlated:512:50000|cargo run --example gridsearch_pq --release --features quantised -- --data correlated --n-samples 50000 --dim 512"
            "pq:euclidean:lowrank:512:50000|cargo run --example gridsearch_pq --release --features quantised -- --data lowrank --n-samples 50000 --dim 512 --intrinsic-dim 64"
            "pq:euclidean:quantisation:512:50000|cargo run --example gridsearch_pq --release --features quantised -- --data quantisation --n-samples 50000 --dim 512 --n-clusters 50"

            "opq:euclidean:correlated:128:50000|cargo run --example gridsearch_opq --release --features quantised -- --data correlated --n-samples 50000 --dim 128"
            "opq:euclidean:lowrank:128:50000|cargo run --example gridsearch_opq --release --features quantised -- --data lowrank --n-samples 50000 --dim 128"
            "opq:euclidean:quantisation:128:50000|cargo run --example gridsearch_opq --release --features quantised -- --data quantisation --n-samples 50000 --dim 128 --n-clusters 50"

            "opq:euclidean:correlated:256:50000|cargo run --example gridsearch_opq --release --features quantised -- --data correlated --n-samples 50000 --dim 256"
            "opq:euclidean:lowrank:256:50000|cargo run --example gridsearch_opq --release --features quantised -- --data lowrank --n-samples 50000 --dim 256 --intrinsic-dim 32"
            "opq:euclidean:quantisation:256:50000|cargo run --example gridsearch_opq --release --features quantised -- --data quantisation --n-samples 50000 --dim 256 --n-clusters 50"

            "opq:euclidean:correlated:512:50000|cargo run --example gridsearch_opq --release --features quantised -- --data correlated --n-samples 50000 --dim 512"
            "opq:euclidean:lowrank:512:50000|cargo run --example gridsearch_opq --release --features quantised -- --data lowrank --n-samples 50000 --dim 512 --intrinsic-dim 64"
            "opq:euclidean:quantisation:512:50000|cargo run --example gridsearch_opq --release --features quantised -- --data quantisation --n-samples 50000 --dim 512 --n-clusters 50"

        )
        ;;
    *)
        echo "Unknown kind: $KIND" >&2
        usage
        ;;
esac

for entry in "${BENCHMARKS[@]}"; do
    tag="${entry%%|*}"
    cmd="${entry#*|}"
    # shellcheck disable=SC2086
    run_and_replace "$tag" $cmd
done

echo "Generated: $OUTPUT"
