"""Ollama client. Every call is logged to logs/llm_calls.jsonl for audit."""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "llm_calls.jsonl"
# Two Ollama servers: the default one (models on C:) and a second one whose model folder is on E:
# (started with OLLAMA_MODELS=E:\ollama-models OLLAMA_HOST=127.0.0.1:11435 ollama serve).
SERVERS = {"c": ("http://localhost:11434", Path.home() / ".ollama/models"),
           "e": ("http://localhost:11435", Path("E:/ollama-models"))}
BIG = "qwen3.5:35b-a3b"
SERVER_OF = {BIG: "e"}  # anything not listed lives on the default server

# Agent -> model. Supervisor sets this after benchmarking (see harness/bench.py).
# Checkpoint 2: everything on the 35B-A3B MoE (the 9B failed on multi-step tasks T04b/T05b/T05c).
MODELS = {
    "mi": BIG,
    "chem": BIG,
    "critic": BIG,
    "scribe": BIG,
}


def _base(model: str) -> str:
    return SERVERS[SERVER_OF.get(model, "c")][0]


def unload_all():
    """Free VRAM before a GRASP script runs: ask both servers to unload whatever they hold."""
    for base, _ in SERVERS.values():
        try:
            for m in requests.get(base + "/api/ps", timeout=10).json().get("models", []):
                requests.post(base + "/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
        except Exception as e:
            print(f"[llm] unload on {base} failed: {e}", flush=True)


def _unwedge(model: str):
    """If Ollama reports the model runner as 'Stopping...', kill that llama-server so a fresh one spawns."""
    import subprocess
    try:
        import os
        srv = SERVER_OF.get(model, "c")
        env = dict(os.environ, OLLAMA_HOST=SERVERS[srv][0].replace("http://", ""))
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30, env=env).stdout
        if not any(model in l and "Stopping" in l for l in ps.splitlines()):
            return
        name, tag = model.split(":")
        man = SERVERS[srv][1] / "manifests/registry.ollama.ai/library" / name / tag
        digest = next(l["digest"] for l in json.loads(man.read_text())["layers"]
                      if "model" in l["mediaType"]).replace(":", "-")
        ps_cmd = ("Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | "
                  f"Where-Object {{ $_.CommandLine -match '{digest}' }} | "
                  "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], timeout=60)
        print(f"[llm] killed wedged runner for {model}", flush=True)
    except Exception as e:
        print(f"[llm] unwedge failed: {e}", flush=True)


def chat(agent: str, system: str, user: str, schema: dict | None = None,
         think: bool = False, temperature: float = 0.2, num_ctx: int = 32768,
         timeout: int = 900, model: str | None = None) -> str:
    model = model or MODELS[agent]
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_ctx": num_ctx,
                    "num_predict": 1200 if agent == "scribe" else 8000, "repeat_penalty": 1.1},
    }
    body["keep_alive"] = "60m"  # avoid unload/reload churn (a stuck 'Stopping...' runner was seen)
    if schema:
        body["format"] = schema
    t0 = time.time()
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(_base(model) + "/api/chat", json=body, timeout=timeout)
            r.raise_for_status()
            msg = r.json()["message"]
            out = msg.get("content", "")
            break
        except Exception as e:  # network/ollama hiccup: retry
            last_err = e
            if isinstance(e, requests.Timeout):
                _unwedge(model)
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"LLM call failed for {agent}/{model}: {last_err}")
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "agent": agent,
                            "model": model, "secs": round(time.time() - t0, 1),
                            "system": system[:2000], "user": user, "output": out,
                            "thinking": msg.get("thinking", "")[:20000]}) + "\n")
    return out


def chat_json(agent: str, system: str, user: str, schema: dict, **kw) -> dict:
    """Structured output; Ollama enforces the JSON schema. One repair retry on parse failure."""
    out = chat(agent, system, user, schema=schema, **kw)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        pass
    # Thinking mode sometimes spends the whole budget reasoning and returns empty content:
    # retry once as-is, then with thinking off (forces a direct structured answer).
    for think in (kw.get("think", False), False):
        kw2 = dict(kw, think=think)
        out = chat(agent, system, user + "\n\nReturn ONLY valid JSON matching the schema. Be concise.",
                   schema=schema, **kw2)
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"{agent}: no valid JSON after 3 attempts; last output: {out[:300]!r}")
