#!/bin/bash
# How self-play throughput responds to being split up, on a box to itself.
#
# One arena leaves most of the machine idle, so the question is how many to run
# at once and how many encoding threads each should get. The last two runs ask
# a different question: what three v4 opponents cost against three v3 ones,
# since v4's observation carries an expected-value solve that v3's does not and
# the trainee's own seat is the same either way.
set -u
cd /root/Mortal/mortal
OUT=/root/bench.log
PY=/root/venv/bin/python
GAMES=${GAMES:-400}
V4=logs/v4/best.pth
V3=logs/baseline.pth

run() { # arenas rayon champion
    MORTAL_CFG=config.online.toml $PY /root/selfplay_bench.py \
        --arenas "$1" --rayon "$2" --champion "$3" \
        --games "$GAMES" --device cuda:1 --weights $V4 2>&1 | grep -a '^arenas' >> $OUT
}

echo "=== $(date -u +%H:%M:%S) self-play, $GAMES games a run, $(nproc) cpus" >> $OUT
echo "--- splitting the box, v4 opponents" >> $OUT
run 1 56 $V4
run 1 28 $V4
run 2 14 $V4
run 4 7  $V4
run 6 5  $V4
run 4 14 $V4
echo "--- the same, against the v3 bot" >> $OUT
run 1 56 $V3
run 4 7  $V3
echo "=== $(date -u +%H:%M:%S) done" >> $OUT
