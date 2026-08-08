---
description: Publish TV Show Episode Identifier - secret-scan, run tests, commit, push to GitHub. Use when the user asks to publish, ship, or release an update.
---

You are running the publish flow for tv-show-identifier. This is a local
Python CLI tool (no server, no container, no deploy target) — publishing
means getting a clean, secret-free commit onto GitHub. There is no build or
remote-deploy stage.

This is a shared-state operation (pushes to a public GitHub repo). Pause for
explicit user confirmation at the checkpoint marked below — don't push
through it silently. If this is the very first publish (no `.git` directory,
or no GitHub repo yet), read "First-time bootstrap" below before starting;
otherwise skip straight to "Steps."

## Key facts

| Thing | Value |
|---|---|
| GitHub repo | `superdupa11/tv-show-identifier` (public) |
| Secrets file | `tv.secrets` (real credentials — gitignored, never staged) |
| Local run output | `match_logs/` (gitignored — not source, not fixtures) |
| Test suite | `pytest` (see `tests/`, deps in `requirements-dev.txt`) |
| No typecheck/build step | This is a plain script project — `pytest` is the only gate |

## First-time bootstrap (skip if `.git` exists and `origin` is already set)

```bash
git status 2>&1 | head -1   # "fatal: not a git repository" means no .git yet
git remote -v               # empty output means no remote yet
```

If there's no `.git`:

```bash
git init -b main
```

Confirm `.gitignore` covers `tv.secrets` and `match_logs/` before the first
`git add` — check it hasn't drifted:

```bash
cat .gitignore
git check-ignore -v tv.secrets match_logs 2>&1   # each should print a matching rule; if either line is empty, STOP
```

Do the secret preflight (Step 1 below), then commit, then create the GitHub
repo and push in one step — **this is the push checkpoint**, confirm with
the user first since it makes the repo public:

```bash
gh repo create superdupa11/tv-show-identifier --public --source=. --remote=origin --push
```

That single command creates the remote repo, wires up `origin`, and pushes
`main`. After it succeeds, publishing on future runs is just "Steps" below.

## Steps

### 1. Secret preflight — do this before anything is staged

Never trust `.gitignore` alone; verify it actively every run.

```bash
git status --porcelain=v1 -uall | grep -E '^\?\? tv\.secrets$' && echo "STOP: tv.secrets is untracked and about to be missed by review — confirm .gitignore covers it"
git check-ignore -v tv.secrets match_logs 2>&1   # each should print a matching .gitignore rule; if any line is empty, STOP
```

Then scan every file that's about to be staged (not just changed ones — a
first-time file matters as much as a modified one) for secret-shaped values.
This project's real credentials are OpenSubtitles/Anthropic/TMDB keys plus
an OpenSubtitles password (see `.secrets.example` for the full key list):

```bash
git add -A -n   # dry run, lists what WOULD be staged, without staging yet
git status --porcelain=v1 | awk '{print $2}' | xargs -I{} grep -lIE \
  '(OPENSUBTITLES_(API_KEY|PASSWORD)|ANTHROPIC_API_KEY|TMDB_API_KEY|_TOKEN|_SECRET|_KEY)\s*=\s*[A-Za-z0-9_-]{12,}' {} 2>/dev/null
```

Any hit needs a human look — a placeholder like `.secrets.example`'s
`your_api_key_here` is fine and expected; a real key or password is not. If
genuinely unsure whether a match is real, stop and ask rather than deciding
alone.

### 2. Verify it actually works before publishing

```bash
python3 -m pytest
```

If this fails — including "no module named pytest" — stop. Don't publish
a build that doesn't pass its own gate, and don't silently `pip install`
around a missing dependency: this machine's system Python is
externally-managed (Homebrew/PEP 668), so a plain `pip install` will fail,
and the fix isn't obvious from the shell alone (check how `requests`/`numpy`
already resolve, and ask the user how they want `requirements-dev.txt`
installed rather than guessing `--break-system-packages`).

### 3. Commit

```bash
git add -A
git status --short   # eyeball the full list — anything unexpected?
git commit -m "<message describing the actual change being published>"
```

If `git status` shows nothing to commit and `HEAD` already matches what's on
`origin/main`, there's nothing new to publish — confirm with the user
whether they still want to do something else, or stop here.

### 4. Push

**Checkpoint — confirm with the user before this step** (public repo).

```bash
git push origin main
```

## Failure signatures

| Symptom | Likely cause |
|---|---|
| `python3 -m pytest` fails with a real test failure | Real bug — fix it, don't publish around it |
| `python3 -m pytest` fails with "No module named pytest" | Dev deps not installed in this environment — ask the user how to install `requirements-dev.txt` here, don't guess `--break-system-packages` |
| `git push` rejected (non-fast-forward) | `origin/main` has commits this checkout doesn't — `git pull --rebase` and resolve, don't force-push |
| `gh repo create` fails with "name already exists" | Repo was already created on a prior run — just add the remote (`git remote add origin https://github.com/superdupa11/tv-show-identifier.git`) and push instead |
| Secret-shaped grep hit on a real value | Stop. Do not commit. Rotate the credential if it was ever staged/committed, even locally-only — treat as compromised. |

## After publishing

Report: the commit that was pushed and the GitHub repo URL.
