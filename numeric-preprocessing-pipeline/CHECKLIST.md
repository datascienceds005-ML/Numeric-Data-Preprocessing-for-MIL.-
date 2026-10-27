# Delivery Checklist — verify this project phase by phase

Work strictly **top to bottom**. Never start the next phase while the current one
is red. After each green phase, tick the checkbox and commit the tick:

```bash
git add -A && git commit -m "checklist: P3 green"
```

Shortcut: `make verify` runs all of P2 + P3 in one command. Run it before every
push. Expected total runtime ≈ 90 s on a 2-core codespace.

---

## P0 — Environment sanity (≈ 10 s)

- [ ] `python3 --version` → **3.11+** (3.12 recommended)
      *In Codespaces the devcontainer guarantees 3.12; skip the worry.*
- [ ] `git status --short --branch` → `## main` and **no other lines** (clean tree)
- [ ] `make help` → prints the target list (proves `make` + Makefile work)

If it fails:
| Symptom | Meaning | Fix |
|---|---|---|
| `make: command not found` | base image lacks build tools | reinstall devcontainer / `sudo apt-get update && sudo apt-get install -y build-essential` |
| dozens of files shown as modified (`M`) | file-mode churn from a non-Linux transfer | `git config core.fileMode false` |
| `python3` < 3.11 | wrong interpreter | use the devcontainer, or install Python 3.12 |

## P1 — Install (≈ 1–3 min, once)

- [ ] `make install-dev` → ends with `Successfully installed numprep …`
- [ ] `python3 -c "import numprep; print('ok')"` → `ok`
- [ ] `numprep --help` → prints usage with subcommands `run / audit / validate / synthesize`

In Codespaces the post-create script already did this; the commands are still
safe to re-run (they are idempotent).

## P2 — Static quality gates (≈ 20 s)

- [ ] `make lint` → `All checks passed!` twice (ruff lint + format check)
- [ ] `make typecheck` → `Success: no issues found in 31 source files`

If it fails: run `make format` first (auto-fixes formatting), re-run `make lint`,
then send me the full output — type errors are usually a real signal, not noise.

## P3 — Test suite + coverage gate (≈ 60 s)

- [ ] `make test` → **`196 passed`** and
      **`Required test coverage of 85.0% reached. Total coverage: 88%+`**

If it fails: paste the whole pytest section (do not truncate the traceback).
Re-run once with `-p no:cacheprovider` (the Makefile already passes it) to rule
out stale cache before investigating.

## P4 — End-to-end smoke, no Docker (≈ 30 s)

- [ ] `make synthesize` → writes ~5k-row seeded messy CSV(s) into `data/raw/`
- [ ] `make run-local` → exit 0; structured JSON logs show every stage
- [ ] `ls outputs/run` → parquet partitions (`train/`, `validation/`, `test/`),
      `preprocessor.joblib`, `manifest.json`, fitted stats, feature catalog,
      `reports/summary.md`
- [ ] `python3 -c "import json;m=json.load(open('outputs/run/manifest.json'));print(m['outputs'])"`
      → shows output file hashes (record them for P5)
- [ ] `numprep validate --input data/raw` → exit 0 (input contract passes)

Expected pipeline stages in the log, in order: `ingest → contract → audit →
clean → split → fit(train) → transform → contract_out → export`.

## P5 — Determinism gate (≈ 40 s)

- [ ] `make determinism` → script exits 0: **byte-identical Parquet hashes**
      across two fresh processes with different `PYTHONHASHSEED` values

This is the project's core reproducibility promise; it is also enforced in CI.

## P6 — Docker (≈ 3–5 min; needs Docker — guaranteed in Codespaces)

- [ ] `make build` → image `numprep:local` built
- [ ] `make run` → containerized run over the read-only `data/` mount, exit 0
- [ ] `make docker-smoke` → smoke test incl. in-container determinism check, exit 0

The scripts self-guard: on a machine without Docker they print a skip notice
instead of failing, so this phase is skippable only when Docker is genuinely
absent (not in Codespaces).

## P7 — CI on GitHub (after first push)

- [ ] Push the repo to GitHub: Actions tab shows the workflow with jobs
      `lint`, `typecheck`, `test`, `docker`, `smoke`, `determinism` — all green
      on Python 3.12.
- [ ] Deliberately break something trivial (add `import os` unused), push,
      watch `lint` go red, revert — proves the gates actually gate.

---

## Progress tracking conventions

1. **Commit per phase.** Green phase ⇒ tick box ⇒ `git commit`. The git log then
   *is* your progress record: `git log --oneline --grep=checklist`.
2. **`make verify` before every push.** It is exactly P2+P3 and costs ~90 s.
3. **Never fix a red phase by editing configs blindly.** Each phase maps to a
   specific layer (environment → install → style → types → behavior → data
   contract → reproducibility → container → CI); a failure in one layer almost
   never has its cause in another.
4. **When asking for help**, include: the phase ID, the exact command, the full
   output (fenced code block), and `echo $?`. README §9 covers the most common
   failure modes; for anything else, bring me the complete log untruncated.
