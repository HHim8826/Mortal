#!/bin/bash
# vast.ai runs this on every container start. The box rebooted at 12:43 UTC on
# 2026-09-12 and nothing came back: the watchdog is an ordinary background
# process and a restart kills it, so two hours were spent idle with the
# checkpoint sitting at step 559,600. The watchdog restarts training; this
# restarts the watchdog.
#
# It goes first and in the background because entrypoint.sh does not return --
# it stays in the foreground streaming the portal's logs. The delay is for the
# GPUs and the filesystem to be there before training asks for them.
#
# /root/watchdog.off still stops everything: the watchdog reads it every minute.
(
    sleep 90
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) container started; bringing the watchdog up" >> /root/watchdog.log
    setsid nohup bash /root/watchdog.sh >/dev/null 2>&1 </dev/null
) &

entrypoint.sh
