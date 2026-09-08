"""Build the Quarto site and stage it on the ``gh-pages`` branch.

The site is *rendered from local artifacts* — it never re-runs the study. That is
deliberate: the numbers on the page must be the numbers in `results/`, produced by
the pinned, seeded, CPU-deterministic pipeline, not a fresh run on some GitHub
runner with a different torch build.

    python -m sarcos_svd.publish              # render + commit onto gh-pages locally
    python -m sarcos_svd.publish --push       # ...and push it
    python -m sarcos_svd.publish --skip-render

Why a branch and not `docs/` on the main branch: `AGENTS.md` says plots and
rendered tables do not get committed next to the code. `gh-pages` keeps the
generated HTML/PNG off the source branch entirely, and GitHub Pages serves it as
the site root.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
WORKTREE = REPO_ROOT / ".gh-pages-worktree"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git hash-object -t tree /dev/null


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def _env() -> dict[str, str]:
    """Environment with this interpreter's own virtualenv activated.

    Quarto resolves `python3` from PATH and refuses to use an unactivated venv, so
    rendering from a script must export VIRTUAL_ENV and put the env's bin first -
    otherwise quarto falls back to a system Python without jupyter/pyyaml.
    """
    env = dict(os.environ)
    prefix = Path(sys.prefix)
    bindir = prefix / "bin"
    if bindir.is_dir():
        env["VIRTUAL_ENV"] = str(prefix)
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    return env


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=_env())
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def branch_exists(branch: str) -> bool:
    return git("rev-parse", "--verify", "--quiet", branch, check=False).returncode == 0


def render(note_source: bool) -> None:
    if shutil.which("quarto") is None:
        raise SystemExit("quarto not on PATH - install it or run with --skip-render")
    if not (REPO_ROOT / "results" / "summary.json").exists():
        raise SystemExit("results/summary.json missing - run `python -m sarcos_svd.report` first")
    if note_source:
        from . import note

        note.main()
    run([sys.executable, "-m", "sarcos_svd.report", "--out", "results/summary.json"])
    run(["quarto", "render"])
    (DOCS / ".nojekyll").write_text("")
    if not (DOCS / "index.html").exists():
        raise SystemExit("render produced no docs/index.html")


SKIP = {".DS_Store", "Thumbs.db"}


def _copy_clean(src: Path, dst: Path) -> None:
    """Copy docs/ into the worktree, dropping macOS clutter — a published site must
    not carry .DS_Store files just because the build machine has them."""
    if src.name in SKIP:
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _copy_clean(child, dst / child.name)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage(branch: str, message: str) -> str:
    """Copy docs/ onto `branch` as the repository root, via a scratch worktree.

    The main working tree is never switched, so this is safe to run with uncommitted
    work on the source branch.
    """
    if WORKTREE.exists():
        raise SystemExit(f"refusing to touch {WORKTREE.name}: remove it first")
    had_branch = branch_exists(branch)
    if had_branch:
        base = branch
    else:
        # `worktree add` needs a commit, and an orphan branch has none: make a
        # throwaway commit whose tree is empty.
        base = git("commit-tree", EMPTY_TREE, "-m", "empty base for gh-pages").stdout.strip()
    git("worktree", "add", "--force", "-B", branch, str(WORKTREE), base)
    try:
        if had_branch:  # clear the previous publication, keeping the branch history
            git("-C", str(WORKTREE), "rm", "-r", "-q", "--cached", "--ignore-unmatch", ".")
        for child in WORKTREE.iterdir():
            if child.name == ".git":
                continue
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
        for item in DOCS.iterdir():
            _copy_clean(item, WORKTREE / item.name)
        git("-C", str(WORKTREE), "add", "-A")
        dirty = git("-C", str(WORKTREE), "diff", "--cached", "--quiet", check=False).returncode != 0
        sha = ""
        if dirty:
            git("-C", str(WORKTREE), "commit", "-q", "-m", message)
            sha = git("-C", str(WORKTREE), "rev-parse", "--short", "HEAD").stdout.strip()
        return sha
    finally:
        git("worktree", "remove", "--force", str(WORKTREE))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--branch", default="gh-pages")
    p.add_argument("--skip-render", action="store_true", help="re-stage whatever is in docs/")
    p.add_argument("--no-note", action="store_true", help="do not regenerate notes/results.qmd")
    p.add_argument("--push", action="store_true", help="git push the branch when done")
    p.add_argument("--message", default=None)
    args = p.parse_args(argv)

    if not args.skip_render:
        render(note_source=not args.no_note)
    if not DOCS.exists():
        raise SystemExit("docs/ missing - run without --skip-render first")

    message = args.message or f"Publish site {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"
    sha = stage(args.branch, message)
    print(f"\ngh-pages branch: {sha or 'unchanged (nothing new to commit)'}")
    print(f"site entry point: {args.branch}:index.html")
    if args.push:
        run(["git", "push", "--force-with-lease", "origin", args.branch])
    else:
        source = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        remote = git("remote", "get-url", "origin", check=False)
        print("\nto publish:")
        if remote.returncode != 0:
            print("  0. no `origin` remote yet - add one:  git remote add origin "
                  "https://github.com/<user>/<repo>.git")
        else:
            print(f"  origin: {remote.stdout.strip()}")
        print(f"  1. git push -u origin {source} {args.branch}")
        print(f"  2. GitHub → Settings → Pages → Source: Deploy from a branch → "
              f"branch `{args.branch}` / `/ (root)`")
        print("  (or re-run this module with --push; --push uses --force-with-lease "
              "on the site branch, never on your source branch)")


if __name__ == "__main__":
    main()
