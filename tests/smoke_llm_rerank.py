import sys, os
sys.path.insert(0, os.path.abspath("."))

# 1) config loads with new llm section
from config import config
llm = config.get_section("llm")
print("LLM config section:", llm)
assert llm["enabled"] is False
assert llm["provider"] == "gemini"
assert "GEMINI_API_KEY" in llm["key_env"].values()

# 2) reranker imports and no-ops without keys
from ai.llm_reranker import LLMReranker, list_models, DEFAULT_MODELS, FALLBACK_MODEL_CHOICES
from core.models import ScoredSegment, TranscriptSegment, ViralScore

class _Log:
    def info(self, *a, **k): print("INFO", a, k)
    def warning(self, *a, **k): print("WARN", a, k)
    def error(self, *a, **k): print("ERR", a, k)

# Ensure no keys are visible for this test
for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(k, None)

segs = []
for i in range(3):
    ts = TranscriptSegment(id=i, start=i*10.0, end=i*10.0+20, text=f"sentence number {i}")
    vs = ViralScore(); vs.total = 50 + i
    segs.append(ScoredSegment(segment=ts, viral_score=vs))

# disabled -> unchanged
r = LLMReranker({"enabled": False, "provider": "gemini"})
out = r.rerank(list(segs), _Log())
print("disabled is_active:", r.is_active())
assert len(out) == 3

# enabled but no key -> falls back, no crash, scores untouched
r2 = LLMReranker({"enabled": True, "provider": "gemini"})
print("enabled-no-key is_active:", r2.is_active())
out2 = r2.rerank(list(segs), _Log())
assert all(s.llm_score is None for s in out2), "no key should mean no llm scores"

# 3) parse_verdicts robustness (markdown fence + extra prose)
raw = '```json\n[{"id":0,"score":80,"reason":"strong hook"},{"id":1,"score":20,"reason":"rambling"}]\n```'
v = LLMReranker._parse_verdicts(raw)
print("parsed verdicts:", v)
assert v[0]["score"] == 80 and v[1]["score"] == 20

# 4) list_models falls back to static list without a key (never raises)
for p in ("gemini", "openai", "anthropic"):
    m = list_models(p, None)
    print(f"models[{p}] fallback:", m)
    assert m == FALLBACK_MODEL_CHOICES[p]

# 5) blending math with a fake provider verdict (monkeypatch _judge)
r3 = LLMReranker({"enabled": True, "provider": "gemini", "blend_weight": 0.5})
os.environ["GEMINI_API_KEY"] = "fake-key-for-test"
r3._judge = lambda payload: {0: {"score": 100, "reason": "x"}, 1: {"score": 0, "reason": "y"}, 2: {"score": 100, "reason": "z"}}
out3 = r3.rerank(list(segs), _Log())
# seg0 heuristic 50 -> blend 0.5*50+0.5*100 = 75 ; seg1 51 ->25.5 ; seg2 52 -> 76
by_id = {s.segment.id: s for s in out3}
print("blended:", {i: (by_id[i].heuristic_score, by_id[i].llm_score, round(by_id[i].viral_score.total,2)) for i in by_id})
assert abs(by_id[0].viral_score.total - 75.0) < 1e-6
assert abs(by_id[1].viral_score.total - 25.5) < 1e-6
os.environ.pop("GEMINI_API_KEY", None)

print("\nALL SMOKE TESTS PASSED")
