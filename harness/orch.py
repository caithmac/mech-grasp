"""Orchestrator: task spec -> MI agent script -> sandboxed run (auto-fix) -> chemist -> critic
-> (one revision round) -> scribe -> claim ledger.

  python -m harness.orch run T01              # full pipeline
  python -m harness.orch run T01 --no-review  # script + run only
  python -m harness.orch review T01           # chemist+critic+scribe on latest run
  python -m harness.orch status               # ledger summary
  python -m harness.orch verify C0007 confirmed "note"   # supervisor sign-off
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import agents
from . import llm
from .llm import chat, chat_json

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\Users\ss864\AppData\Local\miniconda3\envs\mech-grasp\python.exe"
LEDGER = ROOT / "ledger.json"
NOTEBOOK = ROOT / "NOTEBOOK.md"
MAX_FIX = 4
MAX_REVISE = 2
MAX_CHEAP = 6
TASK_ID = [""]

BANNED = [r"\bsubprocess\b", r"os\.system", r"shutil\.rmtree", r"os\.remove", r"\.unlink\(",
          r"\brequests\b", r"urllib", r"\bsocket\b", r"open\([^)]*mechlib", r"rmdir",
          r"except[^:\n]*:\s*pass\b"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        raise ValueError("no python code block in MI reply")
    return max(blocks, key=len)


def _strip_comments(code: str) -> str:
    """Drop # comments so a banned word in prose (e.g. 'requests') doesn't block a script."""
    import io
    import tokenize
    try:
        toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type != tokenize.COMMENT]
        return tokenize.untokenize(toks)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code


def guard(code: str) -> list[str]:
    code = _strip_comments(code)
    return [p for p in BANNED if re.search(p, code)]


def lint(code: str) -> str:
    """Fast static check (syntax errors, undefined names) before paying for a full run."""
    from pyflakes.api import check
    from pyflakes.reporter import Reporter
    import io
    out, err = io.StringIO(), io.StringIO()
    check(code, "script.py", Reporter(out, err))
    msgs = [l for l in (out.getvalue() + err.getvalue()).splitlines()
            if "undefined name" in l or "syntax" in l.lower() or "invalid" in l.lower()
            or "unexpected" in l.lower() or "referenced before assignment" in l]
    return "\n".join(msgs)


LOCK = ROOT / "logs" / "gpu.lock"


def _pid_alive(pid: int) -> bool:
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in r.stdout
    except Exception:
        return True


@contextlib.contextmanager
def gpu_lock():
    """One experiment script at a time (LLM calls still overlap). mkdir is atomic; stale locks
    from dead processes are broken automatically."""
    waited = 0
    while True:
        try:
            LOCK.mkdir(parents=False)
            (LOCK / "pid").write_text(str(os.getpid()))
            break
        except FileExistsError:
            try:
                holder = int((LOCK / "pid").read_text())
                if not _pid_alive(holder):
                    shutil.rmtree(LOCK, ignore_errors=True)
                    continue
            except (FileNotFoundError, ValueError):
                pass
            if waited % 300 == 0:
                log(f"  waiting for GPU lock ({waited}s)")
            time.sleep(10)
            waited += 10
    try:
        yield
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)


def execute(code: str, out: Path, timeout: int) -> tuple[bool, str]:
    out.mkdir(parents=True, exist_ok=True)
    script = out / "script.py"
    script.write_text(code, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT), OUT_DIR=str(out.resolve()), HF_HUB_OFFLINE="1",
               TASK_CACHE=str((out.parent / "_cache").resolve()),
               MPLBACKEND="Agg", PYTHONWARNINGS="ignore", PYTHONIOENCODING="utf-8")
    log_path = out / "stdout.txt"  # streamed live so the supervisor can watch long runs
    with gpu_lock():
        llm.unload_all()  # the 35B LLM would otherwise hold most of the 16 GB VRAM
        t0 = time.time()  # timeout clock starts once we own the GPU
        with open(log_path, "w", encoding="utf-8") as f:
            p = subprocess.Popen([PY, "-u", str(script.resolve())], cwd=str(out.resolve()), env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
            try:
                p.wait(timeout=timeout)
                ok = p.returncode == 0
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                ok = False
                f.write(f"\nTIMEOUT after {timeout}s\n")
    output = log_path.read_text(encoding="utf-8", errors="replace")
    log(f"  ran in {time.time() - t0:.0f}s ok={ok}")
    if ok:
        res = out / "results.json"
        if not res.exists():
            return False, output + "\nERROR: results.json was not written to OUT_DIR."
        try:
            r = json.loads(res.read_text(encoding="utf-8"))
        except Exception as e:
            return False, output + f"\nERROR: results.json is not valid JSON: {e}"
        # a 'successful' run whose results are only error bookkeeping is a failure
        if isinstance(r, dict) and not (set(r) - {"figure_error", "error", "errors", "traceback"}):
            return False, output + (f"\nERROR: results.json contains no results, only {sorted(r)}: {str(r)[:300]}. "
                                    "Your plotting code probably replaced the results dict - keep results separate "
                                    "and re-dump it after recording figure_error.")
        # a run with no finite numbers anywhere in results.json computed nothing
        def _nums(o):
            if isinstance(o, bool):
                return 0
            if isinstance(o, (int, float)):
                return int(math.isfinite(o))
            if isinstance(o, dict):
                return sum(_nums(v) for v in o.values())
            if isinstance(o, list):
                return sum(_nums(v) for v in o)
            return 0
        if _nums(r) == 0:
            return False, output + (f"\nERROR: results.json contains no finite numbers (all null/NaN/empty): "
                                    f"{str(r)[:300]}. Nothing was actually computed - check the computation, "
                                    "do not hide errors.")
        # a 'successful' run whose results are mostly recorded errors is a failure
        if isinstance(r, dict) and r:
            errs = [k for k, v in r.items() if isinstance(v, dict) and set(v) <= {"error", "errors", "traceback"}]
            if 2 * len(errs) > len(r):
                first = r[errs[0]]
                return False, output + (f"\nERROR: {len(errs)}/{len(r)} results are errors, e.g. "
                                        f"{errs[0]}: {first.get('error', first)}")
    return ok, output


def tail(s: str, n: int = 4000) -> str:
    lines = [l for l in s.splitlines() if "DEPRECATION WARNING" not in l and "httpx" not in l]
    s = "\n".join(lines)
    return s[-n:]


def compact(obj, nd=3):
    """Round floats so results fit in a small model's context."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, nd)
    if isinstance(obj, dict):
        return {k: compact(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [compact(v, nd) for v in obj]
    return obj


def results_text(run_dir: Path, limit: int = 14000) -> str:
    r = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    s = json.dumps(compact(r), separators=(",", ":"))
    if len(s) > limit:
        s = s[:limit] + " ...[truncated]"
    return s


def next_run_dir(task_id: str) -> Path:
    base = ROOT / "runs" / task_id
    base.mkdir(parents=True, exist_ok=True)
    n = len([d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")]) + 1
    return base / f"v{n}"


def latest_run_dir(task_id: str) -> Path:
    base = ROOT / "runs" / task_id
    dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name[1:].isdigit()], key=lambda d: int(d.name[1:]))
    return dirs[-1]


def hints(task_id: str) -> str:
    p = ROOT / "tasks" / f"{task_id}.hints.md"
    if not p.exists():
        return ""
    return "\n\nSUPERVISOR HINTS (follow these):\n" + p.read_text(encoding="utf-8")


def last_error_line(output: str) -> str:
    lines = [l for l in output.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def mi_loop(task: str, out: Path, timeout: int, prior: str | None = None) -> bool:
    user = (task + hints(TASK_ID[0])) if prior is None else prior
    reply = chat("mi", agents.MI_SYSTEM, user)
    prev_err = None
    attempt, cheap = 0, 0  # cheap = static/guard rejections, budgeted separately (they cost ~1s)
    while attempt <= MAX_FIX and cheap <= MAX_CHEAP:
        try:
            code = extract_code(reply)
        except ValueError as e:
            cheap += 1
            reply = chat("mi", agents.MI_SYSTEM, user + f"\n\n{e}. Reply with one ```python block.")
            continue
        bad = guard(code)
        lint_err = lint(code) if not bad else ""
        if bad or lint_err:
            cheap += 1
            log(f"  static check failed (not executed): {(str(bad) + lint_err)[:120]}")
            why = ("Silently swallowing exceptions (`except ...: pass`) is forbidden because it hides "
                   "failed tests. Either let the error raise, or catch it and RECORD it, e.g. "
                   "`except Exception as e: errors.append(f'{name}: {e}')` and save `errors` in results.json."
                   if any("pass" in b for b in bad) else "")
            output = (f"Script uses forbidden operations: {bad}. Remove them. {why}" if bad
                      else f"Static check failed before running:\n{lint_err}")
            out.mkdir(parents=True, exist_ok=True)
            tag = f"static_{cheap}"
        else:
            attempt += 1
            log(f"  attempt {attempt}: executing")
            ok, output = execute(code, out, timeout)
            if ok:
                return True
            tag = f"attempt_{attempt}"
            if attempt > MAX_FIX:
                (out / f"failed_{tag}.txt").write_text(output, encoding="utf-8")
                break
        (out / f"failed_{tag}.txt").write_text(output, encoding="utf-8")
        err = last_error_line(output)
        repeated = err == prev_err
        prev_err = err
        log(f"  failed ({err[:120]}); asking MI agent to fix" + (" [REPEATED - escalating]" if repeated else ""))
        extra = ("\n\nYou hit the SAME error again: your mental model of the data is wrong. Re-read the "
                 "ARRAY SHAPES table, check every array's shape at the failing line, and rewrite that "
                 "section from scratch rather than patching it." if repeated else "")
        reply = chat("mi", agents.MI_SYSTEM,
                     f"{task}{hints(TASK_ID[0])}\n\nYour previous script:\n```python\n{code}\n```\n\n"
                     + agents.MI_FIX.format(error=tail(output)) + extra, think=repeated)
    return False


def review(task_id: str, task: str, out: Path) -> dict:
    code = (out / "script.py").read_text(encoding="utf-8")
    res = results_text(out)
    printed = tail((out / "stdout.txt").read_text(encoding="utf-8"), 3000)
    log("  chemist reading results")
    chem = chat_json("chem", agents.CHEM_SYSTEM,
                     f"TASK:\n{task}\n\nRESULTS (results.json, rounded):\n{res}\n\nPRINTED OUTPUT:\n{printed}",
                     agents.CHEM_SCHEMA, think=True)
    (out / "chem.json").write_text(json.dumps(chem, indent=1), encoding="utf-8")
    log("  critic reviewing")
    critic = chat_json("critic", agents.CRITIC_SYSTEM,
                       f"TASK:\n{task}\n\nSCRIPT:\n```python\n{code}\n```\n\nRESULTS:\n{res}\n\n"
                       f"PRINTED OUTPUT:\n{printed}\n\nCHEMIST'S READING:\n{json.dumps(chem, indent=1)}",
                       agents.CRITIC_SCHEMA, think=True)
    (out / "critic.json").write_text(json.dumps(critic, indent=1), encoding="utf-8")
    log(f"  critic verdict: {critic['verdict']}")
    return {"chem": chem, "critic": critic}


def scribe(task_id: str, task: str, out: Path, rev: dict):
    entry = chat("scribe", agents.SCRIBE_SYSTEM,
                 f"Experiment {task_id} ({out.name}).\nTASK:\n{task}\n\nRESULTS:\n{results_text(out, 6000)}\n\n"
                 f"CHEMIST:\n{json.dumps(rev['chem'])}\n\nCRITIC:\n{json.dumps(rev['critic'])}")
    (out / "notebook.md").write_text(entry, encoding="utf-8")
    with open(NOTEBOOK, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {task_id} / {out.name}  ({time.strftime('%Y-%m-%d %H:%M')})\n\n{entry}\n")


def load_ledger() -> list:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []


def record_claims(task_id: str, out: Path, rev: dict):
    led = load_ledger()
    status = {c["claim"]: c for c in rev["critic"]["claim_assessments"]}
    for c in rev["chem"]["claims"]:
        crit = status.get(c["claim"])
        if crit is None:  # critic paraphrases claims; fall back to closest text match
            import difflib
            close = difflib.get_close_matches(c["claim"], list(status), n=1, cutoff=0.4)
            crit = status[close[0]] if close else {}
        led.append({"id": f"C{len(led) + 1:04d}", "task": task_id, "run": out.name, **c,
                    "critic_status": crit.get("status", "unassessed"),
                    "critic_reason": crit.get("reason", ""),
                    "critic_verdict": rev["critic"]["verdict"], "supervisor": "pending"})
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")


def cmd_run(task_id: str, no_review: bool, timeout: int):
    TASK_ID[0] = task_id
    task = (ROOT / "tasks" / f"{task_id}.md").read_text(encoding="utf-8")
    out = next_run_dir(task_id)
    log(f"{task_id} -> {out}")
    if not mi_loop(task, out, timeout):
        log("MI agent could not produce a working script; supervisor needed")
        return 1
    if no_review:
        return 0
    rev = review(task_id, task + hints(task_id), out)
    for _ in range(MAX_REVISE):
        if rev["critic"]["verdict"] != "revise":
            break
        log("  critic asked for revision; MI agent re-running")
        code = (out / "script.py").read_text(encoding="utf-8")
        prev, prev_rev = out, rev
        out = next_run_dir(task_id)
        prior = (f"{task}{hints(task_id)}\n\nYour previous script:\n```python\n{code}\n```\n\nThe reviewer asked for changes:\n"
                 f"{rev['critic']['rerun_instructions']}\nIssues: {rev['critic']['bugs_or_flaws']}\n"
                 "Return the COMPLETE revised script.")
        if not mi_loop(task, out, timeout, prior=prior):
            log("revision failed; keeping previous run")
            out, rev = prev, prev_rev
            break
        rev = review(task_id, task + hints(task_id), out)
    scribe(task_id, task, out, rev)
    record_claims(task_id, out, rev)
    log(f"done: {out}")
    return 0


def cmd_status():
    led = load_ledger()
    for c in led:
        print(f"{c['id']} [{c['task']}/{c['run']}] critic={c['critic_status']:<11} sup={c['supervisor']:<10} {c['claim'][:110]}")


def cmd_verify(cid: str, status: str, note: str):
    led = load_ledger()
    for c in led:
        if c["id"] == cid:
            c["supervisor"], c["supervisor_note"] = status, note
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("task"); r.add_argument("--no-review", action="store_true")
    r.add_argument("--timeout", type=int, default=10800)
    v = sub.add_parser("review"); v.add_argument("task")
    sub.add_parser("status")
    f = sub.add_parser("verify"); f.add_argument("id"); f.add_argument("status"); f.add_argument("note")
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(cmd_run(a.task, a.no_review, a.timeout))
    if a.cmd == "review":
        task = (ROOT / "tasks" / f"{a.task}.md").read_text(encoding="utf-8")
        out = latest_run_dir(a.task)
        rev = review(a.task, task, out)
        scribe(a.task, task, out, rev)
        record_claims(a.task, out, rev)
    if a.cmd == "status":
        cmd_status()
    if a.cmd == "verify":
        cmd_verify(a.id, a.status, a.note)


if __name__ == "__main__":
    main()
