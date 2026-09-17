#!/bin/bash
# How the v4 run is doing, at a glance.
LOG=/root/Mortal/mortal/logs/v4/train.log
echo "== process: $(pgrep -fc '[t]rain.py') train.py processes, tensorboard $(pgrep -fc '[t]ensorboard')"
echo "== steps"
grep -a "total steps" "$LOG" | tail -1 | sed 's/^.*INFO *//'
# Steps/s over the last few save windows, from their elapsed times.
tr '\r' '\n' < "$LOG" | grep -aoE '400/400 \[[0-9:]+<' | grep -oE '\[[0-9:]+' | tr -d '[' | uniq | tail -5 \
    | awk -F: '{ s = (NF == 3) ? $1*3600 + $2*60 + $3 : $1*60 + $2; printf "%.2f steps/s  ", 400 / s } END { print "" }'
echo "== evaluations (vs v3 baseline; 2.5 = even)"
grep -aE "avg rank|avg pt|new record" "$LOG" | sed 's/^.*INFO *//' | tail -9
echo "== problems"
grep -aE "Traceback|diverged|Error|Killed" "$LOG" | tail -3
echo "== gpus"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "== disk: $(df -h / | tail -1 | awk '{print $4 " free of " $2}')"
