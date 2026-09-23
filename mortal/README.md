# The `mortal` directory

This directory holds the Python side of Mortal. The modules that come from
upstream stay flat at the top, exactly where upstream keeps them, and the code
this fork adds lives in subpackages. You run everything from this directory.

## Layout

The top level is upstream's layout plus two shared modules. Everything else is
grouped by what it's for.

| Path | What it holds |
| --- | --- |
| `*.py` at the top | Upstream's modules (`train.py`, `model.py`, `player.py`, `server.py`, `client.py`, and the rest), plus `distributed.py` and `rollout.py`, which the trainer and the data loader import. |
| `config*.toml` | Configurations. Select one with `MORTAL_CFG`. |
| `evaluation/` | Paired test play on fixed wall sets (`evaluate`), two checkpoints on fresh walls (`compare_checkpoints`), and held-out validation (`validate_supervised`). |
| `policy/` | The policy head and the policy-gradient line: `train_policy`, `sharpen_policy`, `train_ppo`, and `train_awr`. |
| `research/` | One-off measurements behind the phase 3 audit: the `deviation_*` replays and `advantage_signal`. |
| `tenpai/` | The model that reads the other three hands: its data, its net, and `train_tenpai`. |
| `riichi_lab/` | A client that plays Mortal against other bots on RiichiLab (riichi.dev). |
| `tests/` | Local tests. They're never committed: `.gitignore` excludes `test_*.py`. |

Keeping upstream's modules in place means that `from model import Brain` works
everywhere and that merging upstream doesn't conflict with this fork's moves.

## Running a script

Upstream's scripts run as files, and the scripts in subpackages run as
modules. Both run from this directory, which puts the top-level modules on the
import path.

```bash
cd mortal
MORTAL_CFG=config.v4.toml python train.py
python -m evaluation.evaluate sets
python -m evaluation.compare_checkpoints --seeds 8000 --key 0xb0a7 a.pth b.pth
python -m riichi_lab --ranked
```

> **Note:** Don't run a subpackage script by its file path, for example
> `python evaluation/evaluate.py`. That puts `evaluation/` on the import path
> instead of this directory, and the script can't import `model` or `config`.

## Running the tests

The tests import modules the same way the scripts do, so you run them from this
directory too.

```bash
cd mortal
python -m unittest discover -s tests
```
