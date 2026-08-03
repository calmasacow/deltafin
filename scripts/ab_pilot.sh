#!/bin/zsh
# Sustained interleaved A/B for PILOT speculative expert prefetch.
#
# Decode wall-clock on this machine swings 20%+ between identical runs, and
# short runs cannot resolve the effect sizes involved. This harness trades
# time for resolution: long fixed-work runs, arms interleaved inside each
# iteration, and arm order coin-flipped per iteration so a systematic
# first/second-position advantage cannot masquerade as an arm difference.
#
# Every run generates exactly the same tokens, so the generated text must be
# byte-identical across all runs and all arms. That is checked, not assumed:
# speculative prefetch may only change *when* bytes are read, never which
# experts the router selects, so any divergence is a correctness alarm and
# invalidates the timing comparison.
#
#   ./experiments/ab_pilot.sh -n 5              # overnight: 5 A/B pairs
#   ./experiments/ab_pilot.sh -n 1              # one pair, then quit
#   ./experiments/ab_pilot.sh --smoke-only      # preflight only
#
set -u
setopt NO_NOMATCH 2>/dev/null || true
typeset -F SECONDS

REPO=${0:A:h:h}
cd $REPO || { print -u2 "cannot enter repository root $REPO"; exit 1 }

ITERATIONS=1
TOKENS=400
SMOKE_TOKENS=8
GAP=90
ARMS=speculation
PROMPT_FILE=""
SEED=""
OUTDIR=""
SKIP_SMOKE=0
SMOKE_ONLY=0

usage() {
  cat <<'USAGE'
ab_pilot.sh - sustained interleaved A/B for PILOT speculative prefetch

  -n N        A/B iterations; each runs both arms once (default 1)
  -t N        tokens generated per timed run (default 400)
  -p FILE     prompt file for timed runs (default: built-in long prompt)
  -a ARMS     comparison preset (default speculation):
                speculation  legacy full prefetch  vs  all prefetch suppressed
                             -> measures what speculation is worth at all
                gate         legacy full prefetch  vs  adaptive gate defaults
                             -> measures what the per-layer governor adds
  -g SECS     cooldown between runs, lets the machine settle (default 90)
  -s SEED     seed for the arm-order coin flips (default: clock; recorded)
  -o DIR      output directory (default bench-results/ab-pilot-<stamp>)
  --smoke-tokens N   tokens for the preflight run (default 8)
  --skip-smoke       skip the preflight
  --smoke-only       run the preflight and exit
  -h          this help
USAGE
}

while (( $# )); do
  case $1 in
    -n) ITERATIONS=$2; shift 2 ;;
    -t) TOKENS=$2; shift 2 ;;
    -p) PROMPT_FILE=$2; shift 2 ;;
    -a) ARMS=$2; shift 2 ;;
    -g) GAP=$2; shift 2 ;;
    -s) SEED=$2; shift 2 ;;
    -o) OUTDIR=$2; shift 2 ;;
    --smoke-tokens) SMOKE_TOKENS=$2; shift 2 ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --smoke-only) SMOKE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) print -u2 "unknown option: $1"; usage; exit 2 ;;
  esac
done

# --- arms ------------------------------------------------------------------
# Suppression arms use warmup=1 so the tested regime covers nearly the whole
# run instead of the governor spending its first samples replaying legacy.
case $ARMS in
  speculation)
    A_LABEL=spec-on;  A_ENV=(K3_PILOT_GATE=off)
    B_LABEL=spec-off; B_ENV=(K3_PILOT_GATE=on K3_PILOT_GATE_THRESHOLD=0.99 K3_PILOT_GATE_WARMUP=1)
    ;;
  gate)
    A_LABEL=legacy;   A_ENV=(K3_PILOT_GATE=off)
    B_LABEL=gated;    B_ENV=(K3_PILOT_GATE=on K3_PILOT_GATE_WARMUP=8)
    ;;
  *) print -u2 "unknown arms preset: $ARMS (want speculation|gate)"; exit 2 ;;
esac

# --smoke-only lives inside the preflight block, so skipping the preflight
# would silently fall through into a full timed run -- exactly the surprise an
# overnight harness must not spring.
if (( SMOKE_ONLY && SKIP_SMOKE )); then
  print -u2 -r -- "--smoke-only and --skip-smoke contradict each other"; exit 2
fi
if (( ITERATIONS < 1 || TOKENS < 1 )); then
  print -u2 "iterations and tokens must both be at least 1"; exit 2
fi

BIN=$REPO/target/release/deltafin
[[ -x $BIN ]] || { print -u2 "missing release binary: $BIN
build it with: cargo build --release -p deltafin"; exit 1 }

STAMP=$(date +%Y%m%d-%H%M%S)
: ${OUTDIR:=$REPO/bench-results/ab-pilot-$STAMP}
mkdir -p $OUTDIR || exit 1
MANIFEST=$OUTDIR/manifest.tsv
[[ -f $MANIFEST ]] || print "iteration\tposition\tarm\tstarted\twall_s\tinternal_s\ttokens\texit\tsha\tlog" > $MANIFEST

# Archive the exact prompt beside the results: a timing number without the
# prompt that produced it is not reproducible.
if [[ -z $PROMPT_FILE ]]; then
  PROMPT_FILE=$OUTDIR/prompt.txt
  cat > $PROMPT_FILE <<'PROMPT'
The following is an engineering note on streaming mixture-of-experts models
from local storage.

Modern sparse models route each token to a small subset of experts, so the
weights actually needed for one forward pass are a tiny fraction of the whole
checkpoint. That makes them attractive to run on a single machine: rather than
holding every parameter resident, a runtime can keep the dense trunk in memory
and stream only the routed experts from disk. The catch is that the router
decides which experts are needed at the last possible moment, so the read
cannot begin until the layer that needs it is already executing. Decode speed
then collapses to a simple ratio: the bytes that miss cache, divided by the
bandwidth of the device holding them.

Three levers move that ratio. The first is the size of each expert on disk,
which quantization and entropy coding attack directly. The second is the hit
rate of whatever cache sits in front of storage, which pinning and admission
policy attack. The third is overlap: issuing reads early enough that they land
while the machine is still computing something else, which speculative
prefetch attacks by predicting the routing decision before it is made. Only
the third leaves the arithmetic of the first two untouched, which is why it is
both attractive and sharply bounded, and understanding that boundary is worth
some care.

Consider what a prefetcher can actually win. If a layer's experts take longer
to read than the previous layer takes to compute, then perfect prediction only
hides part of the read, and the remainder still stalls the pipeline. The
ceiling is therefore not the accuracy of the predictor but the amount of
compute available to hide behind. A runtime that is entirely storage-bound has
almost no window, and a runtime that is entirely compute-bound has no need for
the prefetcher in the first place. The interesting regime is in between, and
locating a given machine within it requires measurement rather than intuition.

Continue the note by working through the following in detail: how one would
measure the available overlap window on a real system without a profiler; why
per-layer prediction accuracy varies with attention architecture and what that
implies for hybrid models; how to decide whether an adaptive policy is worth
its complexity; and what a principled experiment would look like end to end,
including the statistical traps that make small effects appear and disappear.
PROMPT
fi
[[ -r $PROMPT_FILE ]] || { print -u2 "cannot read prompt file: $PROMPT_FILE"; exit 1 }
PROMPT=$(<$PROMPT_FILE)

[[ -z $SEED ]] && SEED=$(date +%s)
RANDOM=$SEED

# --- preflight -------------------------------------------------------------
stray=$(pgrep -fl "deltafin run" 2>/dev/null | grep -v pgrep)
if [[ -n $stray ]]; then
  print -u2 "WARNING: another deltafin run is active; timings will be contaminated:"
  print -u2 "$stray"
  print -u2 "abort with Ctrl-C within 10s, or continue anyway"
  sleep 10
fi

runs_total=$(( ITERATIONS * 2 ))
est_per_run=$(( TOKENS * 11 ))
est_total=$(( runs_total * (est_per_run + GAP) ))
print "=== PILOT A/B ==="
print "repository   $REPO"
print "binary       $BIN"
print "arms         A=$A_LABEL [${A_ENV[*]}]"
print "             B=$B_LABEL [${B_ENV[*]}]"
print "prompt       $PROMPT_FILE ($(wc -w < $PROMPT_FILE | tr -d ' ') words)"
print "iterations   $ITERATIONS  ($runs_total timed runs of $TOKENS tokens)"
print "cooldown     ${GAP}s between runs"
print "seed         $SEED  (arm order is coin-flipped per iteration)"
print "output       $OUTDIR"
print "rough ETA    $(( est_total / 3600 ))h $(( (est_total % 3600) / 60 ))m at ~11s/token; prefill adds to this"
print ""

INTERRUPTED=0
trap 'INTERRUPTED=1; print -u2 "\ninterrupted; finishing manifest and stopping"; exit 130' INT TERM

# run_one <iteration> <position> <label> <tokens> <prompt> <env...>
run_one() {
  local iter=$1 position=$2 label=$3 ntokens=$4 prompt=$5; shift 5
  local log=$OUTDIR/run-i${iter}-p${position}-${label}.log
  local started=$(date +%H:%M:%S)
  print -n "[$(date +%H:%M:%S)] iter $iter pos $position $label ... "
  local t0=$SECONDS
  env "$@" $BIN run --prompt "$prompt" --max-new $ntokens --stats > $log 2>&1
  local rc=$?
  local wall=$(printf '%.1f' $(( SECONDS - t0 )))
  # The binary's own cumulative counter excludes process startup; the shell's
  # wall clock includes it. Record both, prefer the internal one downstream.
  local internal=$(grep -oE 'elapsed=[0-9.]+' $log | tail -1 | cut -d= -f2)
  local tokens=$(grep -oE 'generated=[0-9]+' $log | tail -1 | cut -d= -f2)
  local sha=$(grep -vE '^\[' $log | tr -d '\n' | shasum | cut -c1-16)
  : ${internal:=0}; : ${tokens:=0}
  print "rc=$rc wall=${wall}s internal=${internal}s tokens=$tokens sha=$sha"
  print "$iter\t$position\t$label\t$started\t$wall\t$internal\t$tokens\t$rc\t$sha\t${log:t}" >> $MANIFEST
  return $rc
}

# --- smoke: same prompt through A then B, verified identical ---------------
if (( ! SKIP_SMOKE )); then
  # The preflight proves both arms start, finish, and agree on output. It is
  # deliberately too short to engage the governor (suppression needs a few
  # forward passes to accumulate samples), so `suppressed=0` here is expected
  # and is not a failure.
  print -r -- "--- preflight: $SMOKE_TOKENS tokens through both arms ---"
  smoke_prompt="The capital of France is"
  run_one 0 1 $A_LABEL $SMOKE_TOKENS $smoke_prompt "${A_ENV[@]}" || {
    print -u2 "preflight FAILED for arm $A_LABEL; see $OUTDIR"; exit 1 }
  sleep 5
  run_one 0 2 $B_LABEL $SMOKE_TOKENS $smoke_prompt "${B_ENV[@]}" || {
    print -u2 "preflight FAILED for arm $B_LABEL; see $OUTDIR"; exit 1 }
  a_sha=$(awk -F'\t' -v l=$A_LABEL '$1==0 && $3==l {print $9}' $MANIFEST | tail -1)
  b_sha=$(awk -F'\t' -v l=$B_LABEL '$1==0 && $3==l {print $9}' $MANIFEST | tail -1)
  if [[ $a_sha != $b_sha ]]; then
    print -u2 "PREFLIGHT ABORT: arms produced different text ($a_sha vs $b_sha)."
    print -u2 "Prefetch must never change output. Investigate before timing anything."
    exit 1
  fi
  print "preflight OK: both arms generated identical text ($a_sha)"
  print ""
  (( SMOKE_ONLY )) && { print -r -- "--smoke-only: stopping here"; exit 0 }
  sleep $GAP
fi

# --- timed iterations ------------------------------------------------------
for (( i = 1; i <= ITERATIONS; i++ )); do
  if (( RANDOM % 2 )); then first=A; else first=B; fi
  print -r -- "--- iteration $i/$ITERATIONS (coin flip: $first first) ---"
  if [[ $first == A ]]; then
    run_one $i 1 $A_LABEL $TOKENS $PROMPT "${A_ENV[@]}"
    sleep $GAP
    run_one $i 2 $B_LABEL $TOKENS $PROMPT "${B_ENV[@]}"
  else
    run_one $i 1 $B_LABEL $TOKENS $PROMPT "${B_ENV[@]}"
    sleep $GAP
    run_one $i 2 $A_LABEL $TOKENS $PROMPT "${A_ENV[@]}"
  fi
  (( i < ITERATIONS )) && sleep $GAP
  # Analyze after every iteration so an interrupted overnight run still has a
  # readable verdict for the pairs it completed.
  python3 $REPO/scripts/ab_pilot_analyze.py $OUTDIR > $OUTDIR/analysis.txt 2>&1
done

print ""
print -r -- "=== complete: $OUTDIR ==="
python3 $REPO/scripts/ab_pilot_analyze.py $OUTDIR | tee $OUTDIR/analysis.txt
