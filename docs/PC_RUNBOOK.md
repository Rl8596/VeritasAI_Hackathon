# Runbook for the GPU machine (Windows PC with the RTX 3080)

For the Claude session that runs on the PC. The cloud session writes the code and does all analysis.
This machine only trains. Follow the steps in order. Commands are for Git Bash, from the repo folder,
with the virtual environment's Python (`.venv/Scripts/python`).

## Rules (they protect the published models, the live demo and the test data)

1. **Stop and report on any error**, including "unrecognized arguments". Do not repair code, do not change a
   command, a name, a flag or the list of runs. A changed command would break `docs/V2_PROTOCOL.md`.
2. **Do not change tracked files.** The only things this machine commits are the files that
   `scripts/09_gpu_batch.py` adds itself (`results/v2/`, `results/val_v2_*.json`).
3. **Never run** on this machine: `scripts/03_*` to `08_*`, `scripts/hub_models.py`, `scripts/deploy_space.py`,
   the notebook, or `scripts/02_train.py` by hand. No test result is computed here.
4. **Never use the names `kids_only` or `kids_plus_adults`** and never pull or push those models.
5. Work on the branch `pc/v2-runs`. Never push to `main` or to `claude/friendly-wright-kfzg9j`.
6. Answer the team in German (see `CLAUDE.md`). Short reports: what ran, the last lines of output, what is next.

## Step 1: get the code onto your own branch

First time:

```bash
git fetch origin
git checkout -B pc/v2-runs origin/claude/friendly-wright-kfzg9j
```

Later, when the cloud session says there is new code:

```bash
git -c user.name="PneumoShift GPU runs" -c user.email="pneumoshift@users.noreply.github.com" pull --no-rebase origin claude/friendly-wright-kfzg9j
```

(The neutral name matters: judging is blind, so no personal name may appear in a commit.)

## Step 2: preflight

```bash
.venv/Scripts/python scripts/00_preflight.py
```

It must end with `PREFLIGHT OK`. Report the whole output.

- If it says the train / validation / test labels differ: run it once with `--fix-split`, then again without.
- Anything else that fails: stop and report. Do not train.

## Step 3: two-minute test of the whole chain

```bash
.venv/Scripts/python -u scripts/09_gpu_batch.py smoke
```

It must end with `SMOKE TEST OK`. Report the `device:` line and the `speed:` line (images per second, GPU memory).
If the device is not the RTX 3080, stop.

## Step 4: train

Start stage 1 in the background and report when it has finished:

```bash
.venv/Scripts/python -u scripts/09_gpu_batch.py run --stage 1 --commit
```

Then the remaining stages, in one go:

```bash
.venv/Scripts/python -u scripts/09_gpu_batch.py run --stage 2 --stage 3 --stage 4 --commit
```

What to expect:

- 22 runs in total, each a few minutes. The script prints one line per epoch.
- After every run it copies the small result files to `results/v2/<name>/`, commits them and pushes them to
  `origin pc/v2-runs`. Weights stay on this machine in `artifacts/`.
- A finished run is skipped when the script is started again, so an interrupted batch can simply be restarted
  with the same command.
- If a run fails, the script goes on with the next one and lists the failed runs at the end. Report them with the
  last 20 lines of `logs/gpu/<name>.log`. Do not retry with other settings.
- The printed table shows **validation** results. Do not interpret them and do not pick a "best" model:
  the headline model is fixed by name in the protocol.

After each of the two commands, report: the table that the script prints at the end, and the output of

```bash
git log --oneline -3 && git status --short | head
```

## Step 5: upload the weights of the headline model (only when asked)

This needs a person to log in to Hugging Face on this machine once, in their own terminal
(the token must not be typed into a chat):

```bash
.venv/Scripts/hf auth login
```

Then:

```bash
.venv/Scripts/python scripts/09_gpu_batch.py upload v2_kids_plus_all
```

This creates the new model repo `RL8596/pneumoshift-v2-kids-plus-all`. It cannot touch the published models:
the upload script refuses their names.

## If something looks wrong

- No output for several minutes: the first run downloads the pretrained network (45 MB) and Windows needs time to
  start. Check `logs/gpu/<name>.log` before assuming it hangs.
- "CUDA out of memory": report it. Other programs that use the GPU must be closed by the team.
- The PC must not go to sleep while the batch runs.
