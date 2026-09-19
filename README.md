# TabPFN 3.5 for Super Mario Bros.

A RAM-based imitation policy using the successful trajectories you select with
`data_selection.py`. The pipeline prepares a compact context table, fits an
explicit TabPFN **v3.5** classifier, saves it, and predicts controller actions
from live NES RAM. You manage the held-out level yourself; no split is generated.

`train` performs TabPFN context fitting, **not gradient fine-tuning or reinforcement
learning**. Successful demonstrations do not establish that the resulting policy
can complete a new level or recover after errors. Measure that with live rollouts.

## Setup

The project uses Python 3.14 and TabPFN package 9.x (model version 3.5).
From this directory:

```powershell
uv sync
uv run tfm4mario doctor
```

The existing local environment was found to have CPU-only PyTorch. CPU is suitable
for a tiny integration check; for larger contexts and gameplay latency, use an
NVIDIA GPU and a CUDA-enabled PyTorch build matching your machine. Install PyTorch
following its [official selector](https://pytorch.org/get-started/locally/), then
run `doctor` again. Do not assume CUDA is working just because a GPU is installed.
If using a manually configured environment, `python -m tfm4mario` is equivalent to
the installed `tfm4mario` command. The same CLI works on a Linux server.

TabPFN downloads its pretrained checkpoint on first fit. Its 3.5 weights have a
separate non-commercial license; check the
[official model instructions](https://github.com/PriorLabs/TabPFN#basic-usage).
**One-time setup:** log in at [Prior Labs](https://ux.priorlabs.ai), accept the
license in the Licenses tab, then obtain your API key from the Account page.
For non-interactive training, set `TABPFN_TOKEN` in your process environment.
Do not save credentials in `config.toml` or commit them. An interactive terminal
may guide you through authentication instead. Our local real-fit check reached
this license gate, so checkpoint-backed fitting and gameplay have not yet been
validated in this workspace.
For offline execution, set `train.model_path` in `config.toml` to your local
`tabpfn-v3.5-20260909.safetensors` file.
This must be the v3.5 checkpoint. A fitted archive omits the foundation weights,
and records its checkpoint path. When switching machines/operating systems,
copy the context and refit on the destination (or reproduce the checkpoint path).
Fitted artifacts require the same TabPFN package version as the training process.

## Prepare your context

**Edit `config.toml` for all routine settings.** Shared `[paths]` connect the
preparation output to training and the fitted model to inference automatically.
`[runtime]` sets the device and seed; each command has its own settings section.
Paths in TOML are relative to that TOML file. Unknown keys, invalid choices and
invalid value types fail immediately. CLI flags are optional overrides; use
`--config another.toml` for a different experiment. `--no-render` overrides a
configured `render = true`.

```powershell
python data_selection.py
uv run tfm4mario prepare
```

Remove your held-out level from `processed_data` before preparation, as planned.
The existing selector is unchanged and copies all winning folders; it neither
excludes a test level nor clears old output. Point `--data` at your actual selected
directory if different (set `paths.selected_data`). Inputs are read-only and existing output artifacts are
never overwritten.

Preparation selects winning trajectories by default, sorts frame numbers
numerically within each episode, and samples candidate rows reproducibly across
the selected data. `--max-rows` is a cap before removing non-controllable states;
the final count can be lower. Uniform reservoir sampling retains the natural
action proportions. `--stride 4` reduces context redundancy; it does **not** cause
the live emulator to skip frames. Inspect the returned action counts, especially
for rare jump/climb/pipe combinations, before scaling up.

The `.npz` includes `X`, `y`, named feature metadata, episode/level IDs, source and
target paths, frame numbers, and a source-pair hash. IDs and paths are provenance,
not model inputs. `numpy.load(path, allow_pickle=False)` reads it. No image
decoding, Torch, weights, or emulator is required for preparation.

### RAM features

The shared offline/live extractor produces **202 features**:

| Group | Features |
|---|---|
| Mario (17) | Screen-relative x, y, position within a tile, signed raw x/y speed, movement state/direction, size, power-up, swimming/crouching, collision flags, area type, injury/star timers |
| Objects (42) | Six nearest active object slots, each with presence, type, state, relative x/y and signed raw x/y speed |
| Terrain (143) | 13 rows × 11 columns of metatile IDs, spanning 3 tiles behind to 7 ahead of Mario |

Object coordinates are relative to Mario; the terrain window follows him.
The terrain circular buffer alternates between addresses `0x500` and `0x5D0`.
Tiles outside the current viewport and absent object fields are missing values,
which TabPFN supports. Metatile/object IDs are marked categorical. Nonzero tiles
are **not** all treated as solid: pipes, coins, platforms, water and other tile
types need different behavior. Speeds are signed register values, not claimed to
be calibrated pixels per frame. Object types share slots with platforms and
power-ups, so the feature names intentionally say “object.”

The extractor excludes controller registers, score, frame counter, level/world
IDs, outcome, and absolute level progress. It currently does not add history,
fireball-specific slots or an exhaustive terrain physics model. Feature schema
version and ordered names are checked when loading a policy or evaluation table.

### Dataset repairs and action timing

RAM is in binary `tEXt` trailers **after IEND**, and the recorded chunk length is
wrong when CR bytes were expanded. The reader follows the correction in
[dataset issue #4](https://github.com/rafaelcp/smbdataset/issues/4): `CR LF -> CR`,
then requires exactly 2048 bytes. It verifies BP1/action and OUTCOME against the
filename. `--ram-encoding raw` is only for already repaired files.

Gameplay button bits follow the corrected dataset mapping in
[issue #2](https://github.com/rafaelcp/smbdataset/issues/2): A=128, up=64, left=32,
B=16, right=4, down=2. START/SELECT remain ambiguous in that discussion; those
labels and opposing directions are excluded, not silently remapped. The model
predicts whole observed button combinations, preserving simultaneous run/jump.
The adapter translates to nes-py's different controller bit order.

**Alignment is an explicit assumption:** default `--label-offset 1` pairs
RAM at frame `f` with the recorded action at `f+1`, treating each RAM snapshot as
post-action state. The collector's capture timing has not been established from
the dataset alone. If you verify that RAM is captured before its same-frame
action, use `--label-offset 0` instead, and rebuild context and model. Pairs never
cross episodes or missing frame numbers. Excluding controller bytes prevents
direct label copying but does not by itself resolve post-action timing leakage.

## Fit and predict

The supplied config starts with a 256-row candidate cap, one estimator and automatic
device selection. This is a small integration run to establish runtime and memory
requirements:

```powershell
uv run tfm4mario prepare
uv run tfm4mario train
```

For a GPU, after `doctor` confirms CUDA, set `runtime.device = "cuda"`, increase
`prepare.max_rows` (for example, 8192), and choose fresh context/model paths:

```powershell
uv run tfm4mario prepare
uv run tfm4mario train
```

The default `fit_with_cache` favors repeated inference at a memory cost. Reduce
context size first if you run out of memory; alternatively use
`--fit-mode fit_preprocessors` or `low_memory`. Increase `--n-estimators` only
after measuring latency. GPU memory needs depend on context size and fit mode;
this project does not assume a particular VRAM capacity or promise 60 FPS.

The model directory contains `policy.tabpfn_fit` and `manifest.json`, recording
features, context provenance, classes, versions, fit timing, first prediction
timing, and an artifact hash. Load only fitted artifacts you trust (TabPFN's
serialization uses Python objects). A partially failed save has no valid manifest.

```powershell
uv run tfm4mario predict
```

Set `predict.ram` to your RAM file, or replace it with `predict.png` for a dataset
frame. `ram.bin` must contain the actual 2048 raw RAM bytes; no text/hex or newline repair
is applied to emulator dumps. Prediction reports dataset action, button names,
model confidence (not a success probability), and latency.

For another emulator, read its RAM and use the same API directly:

```python
from pathlib import Path
from tfm4mario.policy import Policy

policy = Policy(Path("artifacts/policy"), device="cuda")
decision = policy.predict_ram(ram_bytes)  # exactly 2048 bytes at a frame boundary
# Set decision["buttons"], advance ONE emulated frame, then read RAM again.
```

## What to prepare for the game

1. Use the original NES Super Mario Bros. RAM layout; start with the unmodified
   `v0` environment, not a ROM hack or a different Mario game.
2. Install the optional current Gymnasium-based environment, then check it:

   ```powershell
   uv sync --extra game
   uv run tfm4mario doctor
   ```

   Uncomment `doctor.env_id` in the config to enable the emulator check.
   The pinned major versions use Gymnasium. Do not mix old Gym/nes-py 8.x
   installation recipes into this environment. If native emulator installation
   fails on your machine, use a Linux server/WSL and send the build error.
3. Confirm `doctor` reports `ram_bytes: 2048`, `features: 202`, and the intended
   device. Set `play.env_id` to your held-out world and level.
4. Run headless on a server, or set `play.render = true` on a machine with a display:

   ```powershell
   uv run tfm4mario play
   ```

The emulator waits while TabPFN predicts, so slow inference makes gameplay slower
in wall-clock time rather than dropping actions. Default action repeat is one.
`--action-repeat 4` deliberately changes the control policy; it is an experiment,
not a transparent speed optimization. Each episode writes a decision JSONL trace
and summary with completion flag, progress, reward, stop reason and p50/p95
prediction latency. Repeated resets may be deterministic; multiple identical
episodes are not independent evidence of robustness.

If you use your own game setup instead, it needs `reset`, synchronous one-frame
`step`, and access to the 2048-byte NES RAM snapshot. Keep the same feature
extractor and explicit button translation. A remote inference service is not
required: copy this project and the selected context to the GPU machine and run
the emulator and policy together, headless.

## Optional evaluation and checks

You supply the held-out data; the pipeline does not choose it. To prepare a test
table, copy the config to `heldout.toml`, set `paths.selected_data` to the test
directory, `paths.context` to `artifacts/heldout.npz`, and `prepare.outcome` to
`all`. Run preparation with that config, then evaluation with the original config:

```powershell
uv run tfm4mario prepare --config heldout.toml
uv run tfm4mario evaluate
uv run python -m unittest discover -s tests -v
```

Offline action accuracy is not level-completion rate. The evaluation reports
balanced accuracy, per-action metrics and unseen target actions as well. Tests
exercise corruption repair, state features, excluded controller fields, numeric
frame alignment, source preservation, action conversion, and rollout termination
without downloading a model or starting an emulator.

Address sources: [SMB disassembly](https://github.com/nwoeanhinnogaehr/smb-assembler/blob/master/smbdis.asm),
[nes-py button mapping](https://github.com/Kautenja/nes-py/blob/master/nes_py/wrappers/joypad_space.py),
[current Mario environment](https://github.com/Kautenja/gym-super-mario-bros),
[TabPFN API](https://github.com/PriorLabs/TabPFN).
