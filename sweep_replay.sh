#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# agent_id is now read from each fixture's eigenflux_profile.agent_id
COMMON="--source replay --model qwen-plus --workers 16"

run() {
    local label="$1"; shift
    echo "======== [$label] ========"
    python3 -m proactive_eval simulate $COMMON "$@" 2>&1 | grep -E "DONE|items|Experiment:|Error"
    echo ""
}

# Group 1: Baseline + Alpha/Beta sweep
group1() {
    run "baseline (default)"
    run "pure-keyword (a=0 b=1)" --alpha 0.0 --beta 1.0
    run "pure-semantic (a=1 b=0)" --alpha 1.0 --beta 0.0
    run "balanced (a=0.5 b=0.5)" --alpha 0.5 --beta 0.5
}

# Group 2: Gamma + Delta sweep
group2() {
    run "high-freshness (g=0.5)" --gamma 0.5
    run "no-freshness (g=0)" --gamma 0.0
    run "high-diversity (d=0.5)" --delta 0.5
    run "no-diversity (d=0)" --delta 0.0
}

# Group 3: Min-relevance + Exploration + Recall sweep
group3() {
    run "no-filter (min=0)" --min-relevance-score 0.0
    run "strict-filter (min=0.5)" --min-relevance-score 0.5
    run "exploration=5" --exploration-slots 5
    run "exploration=15" --exploration-slots 15
    run "small-kw-recall (50)" --keyword-recall-size 50
    run "large-kw-recall (500)" --keyword-recall-size 500
    run "large-knn (k=200)" --enable-knn-recall --knn-recall-k 200
}

# Run 3 groups in parallel (max ~32 concurrent LLM requests)
group1 &
group2 &
group3 &
wait

echo "======== ALL DONE ========"
