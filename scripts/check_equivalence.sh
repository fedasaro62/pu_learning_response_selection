#!/usr/bin/env bash
# Compares the first-batch loss at seed 0 against the losses logged by the previous campaign.
# A match confirms model init, data order and alpha line up. All six rows always run and the
# observed value is always printed, so a mismatch shows the full picture (no `set -e` on purpose).
# Needs a GPU. Usage: scripts/check_equivalence.sh [python_interpreter]
set -uo pipefail

PYTHON="${1:-/nfs/home/dasaro/research/innonext-BERT_FP/.venv/bin/python}"

run() {
    local task="$1" objective="$2" prior="$3" expected="$4"
    local got err
    err=$(mktemp)
    got=$("$PYTHON" -u FineTuning/Response_selection.py \
              --task "$task" --rPU 0.8 --post-train positive_only \
              --objective "$objective" --prior "$prior" --seed 0 \
              --max-batches 1 --is_training 2>"$err" \
          | grep -oP 'Batch\[0\] - loss: \K[0-9.]+' | head -1)

    if [[ "$got" == "$expected" ]]; then
        printf '  OK   %-12s %-18s %s\n' "$task" "${objective}/${prior}" "$got"
        rm -f "$err"
    else
        printf '  FAIL %-12s %-18s expected %s, got %s\n' "$task" "${objective}/${prior}" "$expected" "${got:-<none>}"
        [[ -z "$got" ]] && { echo "       stderr tail:"; tail -n 5 "$err" | sed 's/^/       /'; }
        rm -f "$err"
        return 1
    fi
}

matched=0
run ubuntu     pu         exact 1.188617 && ((matched++))
run ubuntu     pu         zero  1.269248 && ((matched++))
run ubuntu     supervised exact 0.551310 && ((matched++))
run e_commerce pu         exact 1.622780 && ((matched++))
run e_commerce pu         zero  1.487587 && ((matched++))
run e_commerce supervised exact 0.705367 && ((matched++))

echo "$matched of 6 matched"
[[ $matched -eq 6 ]]
