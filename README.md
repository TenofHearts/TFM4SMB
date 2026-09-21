# TabPFN 3.5 for Super Mario Bros.

A RAM-based imitation policy using successful and failed trajectories cached by
`data_selection.py`. The pipeline prepares a compact context table, fits an
explicit TabPFN **v3.5** classifier, saves it, and predicts controller actions
from live NES RAM. Each sample conditions on two consecutive frames and a desired
per-action success value. The supplied configuration deliberately restricts both
training and live evaluation to level 1-3; it measures same-level control rather
than cross-level generalization.

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
may guide you through authentication instead. Checkpoint-backed fitting and
emulator rollout plumbing have been validated in this workspace.
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

The extraction reads and validates each PNG once, then stores RAM, action, frame
identity, and outcome in one compressed `.npz` shard per episode under
`processed_data/metadata_cache`. It stores no image pixels.
Completed shards are reused if extraction is restarted. Use `--data` and
`--output` to override the defaults, `--outcome win` for a win-only ablation, or
`--workers N` to tune concurrent reads. The cache is self-contained, so the PNGs
are not reopened by `prepare`.

If an episode's embedded `OUTCOME` disagrees with its filename, extraction
discards that entire trajectory and records it under `skipped_trajectories` in
the manifest and final report. Other metadata or RAM corruption remains fatal.

Set `prepare.include_level` to the sole world-level used for training. The
supplied config uses `"1-3"`; filtering happens while the context is built, so
the complete metadata cache can remain intact. `include_level` and
`exclude_level` are mutually exclusive. Point `paths.selected_data` at the cache
directory.
Inputs are read-only and existing context artifacts are never overwritten.

Preparation selects both winning and failed trajectories by default and sorts
frame numbers numerically within each episode. When `max_rows` is binding,
ordinary rows are drawn round-robin across trajectory IDs, with opening rows
preferred. There is no per-action class cap. Non-controllable states are removed
before selection.
`--stride 2` keeps every second teacher decision while reducing adjacent-row
redundancy; it does **not** cause
the live emulator to skip frames. Inspect the returned action counts, especially
for rare jump/climb/pipe combinations, before scaling up.
Any target action that differs from the immediately preceding recorded action
bypasses stride filtering and is mandatory during final selection. Preparation
fails rather than silently removing mandatory changes if they alone exceed
`max_rows`.

The context `.npz` includes `X`, `y`, named feature metadata, episode/level IDs, source and
target paths, and frame numbers. IDs and paths are provenance,
not model inputs. `numpy.load(path, allow_pickle=False)` reads it. No image
decoding, Torch, weights, or emulator is required for preparation.

### RAM features

The base semantic extractor produces **204 state features**:

| Group | Features |
|---|---|
| Mario and timing (19) | Screen-relative x, y, position within a tile, signed raw x/y speed, movement state/direction, size, power-up, swimming/crouching, collision flags, area type, injury/star timers, game timer, and opening screen timer |
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

The model input concatenates the preceding and current state vectors, six bits
for the action applied to the current state (`A`, directions, and `B`), and
`desired_action_value`, producing **415 inputs**. Teacher preparation gets those
bits from action `f` while labeling action `f+1`; live control supplies the last
action actually applied. Each recorded action is labeled
from its observed RAM transition: `1` when it moves forward by at least the
configured `min_progress_delta` in one frame **and** reaches a new trajectory
progress maximum, or when it produces a score or power-up gain; `0` when no
measurable consequence is observed; and `-1` within the configured pre-death
window before a detected death. The supplied 3-pixel threshold selects fast
forward movement rather than rewarding every 1-pixel crawl. Death is
detected from Mario's RAM state rather than inferred from a `fail` filename.
Negative takes precedence over positive when both rules match. Live prediction
deliberately requests `desired_action_value=1`.
The first decision after reset duplicates the current state because no prior
frame exists. Later rollout decisions use genuinely adjacent emulator states,
including when `action_repeat` is greater than one.

The state extractor excludes controller registers, score, frame counter,
level/world IDs, and absolute level progress. Previous action bits are explicit
policy inputs rather than emulator-memory controller registers. The two semantic
timers are retained: without them, the visually identical opening wait states
alias “wait” and “move.” It currently does not add fireball-specific slots or an
exhaustive terrain physics model. Feature schema version and ordered names are
checked when loading a policy or evaluation table.

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

The configured `label_offset = 1` pairs RAM at frame `f` with the action at frame
`f+1`. Across the cache, the action stored in a frame matches SMB's saved-controller
RAM byte on 89.6% of frames after translating the bit layout. This establishes
that the snapshot is post-action state: its same-frame action has already been
applied. A live policy must choose what happens next, hence the next-frame target.
Controller registers remain excluded from model features, and pairs never cross
episodes or gaps in frame numbering. Use offset 0 only to reconstruct the action
that produced an already-recorded state, not for live control.

## Fit and predict

The supplied config uses a 50,000-row candidate cap, one estimator, automatic
device selection, and new artifact paths that preserve the earlier smoke model:

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
features, context provenance, classes, versions, fit timing, and first prediction
timing. Load only fitted artifacts you trust (TabPFN's
serialization uses Python objects). A partially failed save has no valid manifest.

```powershell
uv run tfm4mario predict
```

Set `predict.ram` to your RAM file, or replace it with `predict.png` for a dataset
frame. `ram.bin` must contain the actual 2048 raw RAM bytes; no text/hex or newline repair
is applied to emulator dumps. A dataset PNG supplies its recorded previous action
automatically. For a standalone RAM dump, pass `--previous-action` when the prior
input was not no-op. Prediction reports dataset action, button names, model
confidence (not a success probability), and latency.

For another emulator, read its RAM and use the same API directly:

```python
from pathlib import Path
from tfm4mario.policy import Policy

policy = Policy(Path("artifacts/policy"), device="cuda")
policy.reset_history()  # call at every episode boundary
decision = policy.predict_ram(ram_bytes, action_value=1)  # tracks its last action
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
3. Confirm `doctor` reports `ram_bytes: 2048`, `features: 415`, and the intended
   device. `play.env_id` must name the same level as `prepare.include_level`.
4. Run headless on a server, or set `play.render = true` on a machine with a display.
   Set `play.record_video = true` to write `episode-000.mp4` inside the configured
   rollout directory; this requires FFmpeg on `PATH`. Live display and recording
   are mutually exclusive:

   ```powershell
   uv run tfm4mario play
   ```

The emulator waits while TabPFN predicts, so slow inference makes gameplay slower
in wall-clock time rather than dropping actions. Default action repeat is one.
`--action-repeat 4` deliberately changes the control policy; it is an experiment,
not a transparent speed optimization. Probability-based modes avoid the argmax
failure in which an ambiguous state repeatedly selects the modal wait action and
never changes. Set `play.action_selection = "argmax"` only for a deterministic
mode-policy comparison. Each episode writes a decision JSONL trace
and summary with completion flag, progress, reward, stop reason and p50/p95
prediction latency. A recorded video uses emulator frames at `play.video_fps`, so
slow model inference does not create pauses in the MP4. Repeated resets may be deterministic; multiple identical
episodes are not independent evidence of robustness.

If you use your own game setup instead, it needs `reset`, synchronous one-frame
`step`, and access to the 2048-byte NES RAM snapshot. Keep the same feature
extractor and explicit button translation. A remote inference service is not
required: copy this project and the selected context to the GPU machine and run
the emulator and policy together, headless.

## Between-episode online adaptation

`adapt` is a separate experimental mode. Within each episode the policy stays
frozen. The online cache first holds up to `adapt.online_capacity` adjacent RAM
transitions without action-value labels. When the cache fills, its ending state
assigns one delayed `-1/0/1` value to every action in that cache: death assigns
`-1`; otherwise net new progress, score, or power-up gain assigns `1`; and no
measured result assigns `0`. Progress assigns `+1` only when the batch endpoint
advances by at least `adapt.online_min_progress_delta` pixels from the batch
start **and** establishes a new episode-wide progress maximum. Recovering
previously covered ground is therefore neutral. The labeled batch is appended permanently to the
accumulated online context, written to `paths.online_cache`, and the now-empty
cache accepts another batch. A partial final batch is labeled from the episode's
ending state and flushed the same way.

No refit occurs when a batch is flushed. At episode end, TabPFN is refit once on
the original prepared context plus every accumulated online batch, so those new
rows are first used by the next episode. The saved base policy is never
overwritten. The final episode persists its rows but skips an unnecessary refit;
a later invocation loads and applies that accumulated context before episode one.

```powershell
uv run tfm4mario adapt
```

The supplied configuration runs five consecutive 1-3 episodes. Each episode
writes its own JSONL trace and MP4, while `summary.json` accumulates reward,
progress, completion, online-context size, batch counts, action-value counts, and
refit duration. The
next episode uses the updated in-memory context. On later invocations, the
persistent online context is loaded and applied before episode one. Choose a fresh
`paths.adaptive_rollout` directory for every invocation; retain the same
`paths.online_cache` to continue adapting, or choose a new cache for an
independent experiment.

This is supervised batch-delayed adaptation, not full reinforcement learning:
it has no learned value function or policy-gradient objective. Every action in a
batch receives the same ending-state value, including actions that may not have
caused that result. CPU refits may pause for a substantial time between episodes.
Compare the per-episode metrics rather than treating one stochastic run as proof
of improvement.

The configured `play` and `adapt` modes use unconditional epsilon-sampling. On
every decision they choose a uniformly random known action with probability
`0.30`; otherwise they sample from the model's predicted class probabilities.
There is no confidence threshold and no greedy fallback, so a confident no-op
cannot create a permanent fixed point. Traces and summaries report exploratory
decision counts. Uniform exploration uses only actions already present in the
fitted model's class set.

## Optional evaluation and checks

The configured live evaluation runs the same 1-3 level used for training. This is
not evidence of cross-level generalization. For an honest offline same-level
metric, split complete 1-3 trajectory IDs into disjoint training and evaluation
sets; never randomly split rows from the same trajectory. Point a copied config
at the evaluation cache and a fresh context path, then run:

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
