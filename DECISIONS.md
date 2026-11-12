# Decisions

One line per decision that the original spec left open. Newest at the bottom.

- The folder was already a git repo with history, so it was reused instead of running `git init`.
- The old requirements.txt was a UTF-16 `pip freeze` from PowerShell with torch and earthengine-api pins; replaced with a short UTF-8 list without torch.
- Added `common.py` (shared logging, status, download, raster and boundary helpers) and `summary.py` (stage `summary`) to avoid copying the same code into every script.
- Smoke runs write to `outputs/smoke/` and `results/smoke/` (gitignored) so tiny test numbers can never mix with real results.
- `check_env` and `summary` always rerun (no marker) because they describe the current state.
- A stage that returns NOT RUN (for example missing IDSP file) writes no marker, so it runs again automatically once its input exists.
- Network errors are retried with growing waits, except 401/403/404 style errors which cannot fix themselves.
