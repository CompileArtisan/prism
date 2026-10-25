"""kaggle_sync.py -- keep a Kaggle run of rmbench_rms.ipynb alive across sessions by syncing the SMALL result files
to a branch of your GitHub repo.  Adds only; it never touches rmbench_rms.py or the notebook logic.

Needs: Internet ON in the Kaggle notebook, and a Kaggle Secret named GITHUB_TOKEN (fine-grained PAT, repo
CompileArtisan/prism, permission "Contents: read and write").  The token is never printed.

    import kaggle_sync as KS
    KS.restore()            # first cell of every session: pulls results_rms/ from the branch (if any)
    KS.start_background()   # every 30 min: push small files (json/csv/jsonl/sha256, models < 50 MB)
    KS.push("note")         # push now (use before FREEZE and after every stage)

Big/regenerable things are NOT synced: results_rms/cache_real (geometry cache), models >= MAX_MB.
"""
import os, shutil, subprocess, threading, time
from pathlib import Path

REPO = "CompileArtisan/prism"
BRANCH = "rms-run"                      # a separate branch, so main keeps its history clean; merge later
WORK = Path("/kaggle/working")
CLONE = Path("/kaggle/temp/prism_sync")  # outside /kaggle/working on purpose
RESULTS = "results_rms"
PATTERNS = ["*.json", "*.csv", "*.jsonl", "*.sha256", "*.md", "*.png"]
MODEL_DIR, MAX_MB = "models", 50.0
EXTRA_FILES = ["rmbench_rms.py", "kaggle_sync.py"]   # copied next to the results


def _token():
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret("GITHUB_TOKEN")


def _git(*args, check=True, cwd=None):
    r = subprocess.run(["git", *args], cwd=cwd or CLONE, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(_token(), "***")
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {out[-400:]}")
    return out


def _ensure_clone():
    url = f"https://x-access-token:{_token()}@github.com/{REPO}.git"
    if not (CLONE / ".git").exists():
        CLONE.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "--depth", "1", url, str(CLONE)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("clone failed: " + (r.stdout + r.stderr).replace(_token(), "***")[-300:])
        _git("config", "user.email", "kaggle@example.invalid"); _git("config", "user.name", "kaggle-run")
    _git("fetch", "--depth", "1", "origin", f"+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", check=False)
    have = _git("ls-remote", "--heads", "origin", BRANCH, check=False).strip()
    if have:
        _git("checkout", "-B", BRANCH, f"origin/{BRANCH}")
    else:
        _git("checkout", "-B", BRANCH)          # new branch from the default branch
    return CLONE


def restore():
    """Copy results_rms/ from the branch into /kaggle/working (does not overwrite newer local files)."""
    try:
        c = _ensure_clone()
    except Exception as e:                       # no internet / no secret: start fresh, say so
        print("restore skipped:", e); return 0
    src = c / RESULTS
    n = 0
    if src.exists():
        for p in src.rglob("*"):
            if p.is_file():
                dst = WORK / RESULTS / p.relative_to(src)
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst); n += 1
    print(f"restored {n} files from branch {BRANCH}")
    return n


def _collect():
    src, dst = WORK / RESULTS, CLONE / RESULTS
    n = skipped = 0
    for pat in PATTERNS:
        for p in src.glob(pat):                  # top level of results_rms only
            d = dst / p.name; d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists() or p.stat().st_mtime > d.stat().st_mtime + 1 or p.stat().st_size != d.stat().st_size:
                shutil.copy2(p, d); n += 1
    for p in (src / MODEL_DIR).glob("*.pt"):
        if p.stat().st_size / 2**20 < MAX_MB:
            d = dst / MODEL_DIR / p.name; d.parent.mkdir(parents=True, exist_ok=True)
            if not d.exists() or p.stat().st_size != d.stat().st_size:
                shutil.copy2(p, d); n += 1
        else:
            skipped += 1
    for f in EXTRA_FILES:
        p = WORK / f
        if p.exists():
            shutil.copy2(p, CLONE / f)
    return n, skipped


def push(msg="kaggle sync"):
    try:
        _ensure_clone()
        n, skipped = _collect()
        _git("add", "-A")
        if not _git("status", "--porcelain", check=False).strip():
            print("sync: nothing new"); return None
        _git("commit", "-m", f"{msg} ({time.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
        try:
            _git("push", "origin", BRANCH)
        except RuntimeError:
            _git("pull", "--rebase", "origin", BRANCH, check=False); _git("push", "origin", BRANCH)
        sha = _git("rev-parse", "HEAD").strip()
        print(f"sync: pushed {n} changed files (models skipped for size: {skipped}) -> {BRANCH} @ {sha[:10]}")
        return sha
    except Exception as e:                       # never let a sync problem kill a long run
        print("sync FAILED (run continues):", str(e)[:300]); return None


_started = {"on": False}


def start_background(every_min=30):
    if _started["on"]:
        return
    _started["on"] = True

    def loop():
        while True:
            time.sleep(every_min * 60)
            push("auto-sync")
    threading.Thread(target=loop, daemon=True).start()
    print(f"background sync every {every_min} min -> branch {BRANCH}")
