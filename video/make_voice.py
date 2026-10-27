"""Generate one WAV per narration line with Kokoro (offline). Writes voice/index.json {key: {file, dur}}."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, "E:/grasp_video")
from narration import SAY
from kokoro import KPipeline

VOICE = sys.argv[1] if len(sys.argv) > 1 else "af_heart"
OUT = Path("E:/grasp_video/voice"); OUT.mkdir(exist_ok=True)
pipe = KPipeline(lang_code="a")
idx = {}
for key, text in SAY.items():
    h = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:12]
    f = OUT / f"{h}.wav"
    if not f.exists():
        audio = np.concatenate([np.asarray(a) for _, _, a in pipe(text, voice=VOICE, speed=0.95)])
        sf.write(f, audio, 24000)
    idx[key] = {"file": str(f).replace("\\", "/"), "dur": round(sf.info(f).duration, 3)}
    print(f"{idx[key]['dur']:6.2f}s  {text[:70]}")
json.dump(idx, open(OUT / "index.json", "w"), indent=1)
print("total", round(sum(v["dur"] for v in idx.values()), 1), "s")
