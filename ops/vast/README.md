# Rented-box operations for the v4 runs

These scripts ran the v4 offline and online training on rented vast.ai boxes
from September 11 to September 13, 2026. They lived only in `/root` on those
boxes, and the boxes are gone. They were rebuilt from the session transcript
that wrote them, patches included, and are kept here so the next box starts
from what worked instead of from memory.

## What they assume

The scripts are copied to `/root` on a box and run from there. They hard-code
that layout, so read this before you run them anywhere else.

- The repository is cloned at `/root/Mortal`, with a virtual environment at
  `/root/venv`.
- The box has two GPUs. Offline uses both through `torchrun`; online puts the
  trainer on `cuda:0` and the self-play workers on `cuda:1`.
- Offline runs from `mortal/config.vast.toml` into `mortal/logs/v4`, and online
  runs from `mortal/config.online.toml` into `mortal/logs/v4o`.
- `/root/phase` holds `offline` or `online`, and `/root/stop_at_step` holds the
  step where offline stops.
- Checkpoints are backed up to the private Hugging Face repository
  `hhim8826/mortal4-0911`. `backup_hf.py` refuses to upload if the repository
  is public.

## Scripts

Each script does one job. The table groups them by when you use them.

| Script | Use |
| --- | --- |
| `setup_new_box.sh` | Installs PyTorch, builds libriichi, clones the fork, and downloads the dataset. Safe to run again. |
| `onstart.sh` | vast.ai runs it on every container start; it brings the watchdog back after a reboot. |
| `launch_v4.sh` | Starts or resumes offline training and TensorBoard on `127.0.0.1:6007`. |
| `restart_v4.sh` | Restarts offline between two saves, after proving the last checkpoint loads. |
| `stop_after_save.sh` | Stops offline right after a whole checkpoint, to hand the run to another box. |
| `status.sh` | Shows offline progress, evaluations, and problems on one screen. |
| `preflight_online.py` | Checks everything about the switch to online that can be checked without starting it. |
| `switch_to_online.sh` | Stops offline at its gate, seeds online from `best_ema.pth`, and launches it. Use `--dry-run` first. |
| `launch_online.sh` | Starts the server, the trainer, and the workers, or whichever of them is down. |
| `online_status.sh` | Shows online progress, including the self-play rank against the frozen start. |
| `watchdog.sh` | Restarts whichever phase is current when it crashes or stops saving. `touch /root/watchdog.off` pauses it. |
| `test_watchdog.sh` | Runs the watchdog's decisions against made-up logs; no process is started or stopped. |
| `watch_tick.sh` | Prints only new alerts and measurements, for calling on a timer. |
| `backup_hf.py` | Copies checkpoints to Hugging Face every 2 hours, only after they load. |
| `selfplay_bench.py`, `bench_grid.sh` | Measure self-play throughput by games in flight, arenas, and opponent version. |
| `eval_perf.py` | Samples CPU, thread, and GPU use during an evaluation. |

`ab_checkpoints.py` is not here. `mortal/evaluate.py` replaces it: it plays any
checkpoints on fixed, named sets of walls and compares them paired.

## Checking a change

You can test the watchdog's logic anywhere with bash, because the test sources
only its functions and points them at a temporary tree.

```bash
bash ops/vast/test_watchdog.sh
```

Every shell script must keep LF line endings. A CR once broke `watchdog.sh` on
the box, and `.gitattributes` now forces LF for `ops/**/*.sh`.

## Known limits

The rebuild matches the transcript, not the boxes, so a few things can differ
from what last ran.

- A change made on a box by hand and never shown in the transcript is missing.
  Two spot checks passed: `restart_v4.sh` has the same size in bytes as the
  box's copy, and `launch_online.sh`, before the three later patches, matches
  the copy the box printed on September 12 line for line.
- `test_watchdog.sh` passes 22 of 22 checks, the same result it gave on the
  box.
- `restart_v4.sh` copies a freshly built `libriichi.so` into place before it
  relaunches, because it was written for a libriichi update.
- Throughput numbers in the comments come from a 2x RTX 4070 Ti SUPER box with
  56 usable CPUs. Measure again on a different box.

## Next steps

Before the next rented run, adapt the paths and GPU layout to that box, run
`bash -n` on every script, and run `test_watchdog.sh`.
