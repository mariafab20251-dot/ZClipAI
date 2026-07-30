"""Optional multi-provider LLM re-ranking (Phase 2).

The free heuristic scorer (`ai/scorer.py`) always runs first and produces a
complete, shippable result. This layer is *additive*: when enabled and an API
key is available, it asks an LLM to re-judge the top heuristic candidates for
"would this actually go viral as a standalone short", then blends the LLM's
verdict with the heuristic score.

Design rules:
  * NEVER hard-fail. If the layer is disabled, no key is set, the SDK isn't
    installed, or the API errors out, we log a line and return the heuristic
    ranking unchanged. The tool always works without keys.
  * API keys are read from the GUI key store (api_keys.json, gitignored) at
    call time. They are never stored in yaml/config and never logged.
  * Three providers are supported behind one interface: Google Gemini, OpenAI,
    and Anthropic. Each SDK is imported lazily so a missing package only
    disables that one provider.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from core.models import ScoredSegment
from utils.logging import get_logger, JobLogger

logger = get_logger("llm_reranker")

# ── API key store ───────────────────────────────────────────────────────────
# Keys typed in the GUI are persisted here instead of .env. This file is
# gitignored. The key store is a simple JSON dict of {provider_name: key}.
KEY_STORE_PATH = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"


def _load_key_store() -> Dict[str, str]:
    """Load saved API keys from the JSON key store."""
    if KEY_STORE_PATH.exists():
        try:
            return json.loads(KEY_STORE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_key_store(store: Dict[str, str]):
    """Persist API keys to the JSON key store (gitignored)."""
    KEY_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def get_saved_key(provider: str) -> Optional[str]:
    """Read a saved API key from the JSON key store."""
    store = _load_key_store()
    return store.get(provider, "").strip() or None


def save_api_key(provider: str, key: str) -> str:
    """Save an API key for the given provider. Returns the filename written."""
    store = _load_key_store()
    store[provider.strip().lower()] = key.strip()
    _save_key_store(store)
    # Also set the env var so the current process picks it up immediately.
    env_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "")
    if env_name:
        os.environ[env_name] = key.strip()
    return str(KEY_STORE_PATH)

# Sensible cheap defaults per provider. Used only when the user leaves the model
# blank. The GUI can fetch the live list via `list_models()` so this never has
# to be authoritative.
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}

# Shown in the UI dropdown before/while a live list is fetched. Live fetching
# (list_models) overrides these whenever a valid key is present.
FALLBACK_MODEL_CHOICES = {
    "gemini": ["gemini-2.0-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-4-6"],
}

_SYSTEM_PROMPT = (
    "You are a veteran short-form video editor who has produced thousands of "
    "viral TikToks, Reels and YouTube Shorts. You judge whether a transcript "
    "excerpt, taken as a standalone clip, would hook a scrolling viewer in the "
    "first 2 seconds and hold them to the end. Reward: strong hooks, emotional "
    "or surprising moments, controversy, humor, a clear payoff, and quotable "
    "lines. Punish: rambling, mid-thought starts, no payoff, generic filler.\n\n"
    "You MUST return ONLY a valid JSON array. No markdown fences, no prose, no "
    "explanations before or after. Every object MUST have an integer 'id', an "
    "integer 'score' (0-100), and a string 'reason' (max 12 words)."
)


class LLMReranker:
    def __init__(self, llm_config: Dict[str, Any]):
        cfg = llm_config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.provider = (cfg.get("provider") or "gemini").lower().strip()
        self.model = (cfg.get("model") or "").strip() or DEFAULT_MODELS.get(self.provider, "")
        self.blend_weight = float(cfg.get("blend_weight", 0.6))
        self.blend_weight = min(1.0, max(0.0, self.blend_weight))
        self.max_candidates = int(cfg.get("max_candidates", 25))
        self.transcript_chars = int(cfg.get("transcript_chars", 600))
        self.timeout_seconds = float(cfg.get("timeout_seconds", 60.0))
        self.key_env = cfg.get("key_env", {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        })

    # ------------------------------------------------------------------ public
    def is_active(self) -> Tuple[bool, str]:
        """Return (active, reason). Active means: enabled AND a key is present."""
        if not self.enabled:
            return False, "LLM reranking disabled in config"
        if self.provider not in DEFAULT_MODELS:
            return False, f"unknown provider '{self.provider}'"
        if not self._api_key():
            env = self.key_env.get(self.provider, "?")
            return False, f"no API key in ${env}"
        return True, "ok"

    def rerank(self, scored_segments: List[ScoredSegment], job_logger: JobLogger) -> List[ScoredSegment]:
        """Blend LLM judgment into the heuristic ranking. Always returns a valid
        list — falls back to the input order on any problem."""
        active, reason = self.is_active()
        if not active:
            if job_logger is not None:
                job_logger.info("LLM rerank skipped; using free heuristic", reason=reason)
            return scored_segments
        if not scored_segments:
            return scored_segments

        # Cost cap: only send the top-N heuristic candidates.
        top = scored_segments[: self.max_candidates]
        payload = [
            {"id": i, "transcript": (s.segment.text or "")[: self.transcript_chars]}
            for i, s in enumerate(top)
        ]

        try:
            verdicts = self._judge(payload)
        except Exception as e:  # noqa: BLE001 - any failure must fall back, not crash
            if job_logger is not None:
                job_logger.warning(
                    "LLM rerank failed; keeping heuristic ranking",
                    provider=self.provider,
                    model=self.model,
                    error=str(e),
                )
            return scored_segments

        if not verdicts:
            if job_logger is not None:
                job_logger.warning("LLM returned no usable scores; keeping heuristic ranking")
            return scored_segments

        w = self.blend_weight
        judged = 0
        for i, seg in enumerate(top):
            v = verdicts.get(i)
            if not v:
                continue
            llm_score = float(v.get("score", 0.0))
            llm_score = min(100.0, max(0.0, llm_score))
            seg.heuristic_score = seg.viral_score.total
            seg.llm_score = llm_score
            seg.llm_reason = (v.get("reason") or "").strip()[:200]
            seg.viral_score.total = (1.0 - w) * seg.heuristic_score + w * llm_score
            judged += 1

        # Re-sort the whole list by the (possibly) blended totals.
        scored_segments.sort(key=lambda s: s.viral_score.total, reverse=True)
        if job_logger is not None:
            job_logger.info(
                "LLM rerank complete",
                provider=self.provider,
                model=self.model,
                judged=judged,
                blend_weight=w,
            )
        return scored_segments

    # ------------------------------------------------------------- key + prompt
    def _api_key(self) -> Optional[str]:
        env_name = self.key_env.get(self.provider)
        if not env_name:
            return None
        key = os.getenv(env_name)
        if key and key.strip():
            return key.strip()
        # Fall back to the GUI key store and promote to env for next time.
        saved = get_saved_key(self.provider)
        if saved:
            os.environ[env_name] = saved
            return saved
        return None

    def _build_prompt(self, payload: List[Dict[str, Any]]) -> str:
        lines = [
            "Rate each candidate clip 0-100 for standalone viral potential.",
            "Return ONLY a JSON array, one object per clip, no prose:",
            '[{"id": <int>, "score": <0-100>, "reason": "<<=12 words>"}]',
            "",
            "Candidates:",
        ]
        for item in payload:
            text = item["transcript"].replace("\n", " ").strip()
            lines.append(f'#{item["id"]}: "{text}"')
        return "\n".join(lines)

    def _judge(self, payload: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        prompt = self._build_prompt(payload)
        if self.provider == "gemini":
            raw = self._call_gemini(prompt)
        elif self.provider == "openai":
            raw = self._call_openai(prompt)
        elif self.provider == "anthropic":
            raw = self._call_anthropic(prompt)
        else:
            raise ValueError(f"unknown provider '{self.provider}'")
        return self._parse_verdicts(raw)

    @staticmethod
    def _parse_verdicts(raw: str) -> Dict[int, Dict[str, Any]]:
        if not raw:
            return {}
        # Strip markdown fences and isolate the JSON array.
        text = raw.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        out: Dict[int, Dict[str, Any]] = {}
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict) and "id" in entry:
                    try:
                        out[int(entry["id"])] = entry
                    except (TypeError, ValueError):
                        continue
        return out

    # --------------------------------------------------------------- providers
    def _call_gemini(self, prompt: str) -> str:
        # Prefer the new unified google-genai SDK; fall back to the legacy one.
        key = self._api_key()
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"system_instruction": _SYSTEM_PROMPT},
            )
            return getattr(resp, "text", "") or ""
        except ImportError:
            pass
        # Legacy google-generativeai
        import google.generativeai as genai_legacy  # type: ignore
        genai_legacy.configure(api_key=key)
        model = genai_legacy.GenerativeModel(self.model, system_instruction=_SYSTEM_PROMPT)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or ""

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=self._api_key(), timeout=self.timeout_seconds)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, prompt: str) -> str:
        # Official Anthropic SDK. Haiku is plenty for a cheap ranking task, so we
        # skip extended thinking. Keep max_tokens generous for many candidates.
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=self._api_key(), timeout=self.timeout_seconds)
        msg = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts)


# ---------------------------------------------------------------- model listing
def test_api_key(provider: str, api_key: str) -> Tuple[bool, str]:
    """Lightweight API key validation.

    Makes a tiny API call to verify the key works. Returns (success, message).
    Never raises.
    """
    provider = (provider or "").lower().strip()
    if not api_key:
        return False, "No API key provided"
    try:
        if provider == "gemini":
            # List models — cheap, fast, validates key + permissions.
            from google import genai  # type: ignore
            client = genai.Client(api_key=api_key)
            found = False
            for m in client.models.list():
                actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
                if actions and "generateContent" not in actions:
                    continue
                name = getattr(m, "name", "") or getattr(m, "display_name", "")
                if "gemini" in name.lower():
                    found = True
                    break
            if found:
                return True, "✓ Key works — Gemini API responded"
            return True, "✓ Key valid (but no Gemini models found — using defaults)"
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.models.list()
            return True, "✓ OpenAI key is valid"
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            client.models.list()
            return True, "✓ Anthropic key is valid"
        else:
            return False, f"Unknown provider: {provider}"
    except Exception as e:  # noqa: BLE001
        err = str(e).strip()
        # Trim verbose error messages to just the useful part
        if not err or len(err) > 200:
            err = err[:200]
        return False, f"✗ Key test failed: {err}"


def list_models(provider: str, api_key: Optional[str] = None) -> List[str]:
    """Best-effort live model list for the GUI dropdown. Returns the static
    fallback list if no key is given or the lookup fails — never raises."""
    provider = (provider or "").lower().strip()
    fallback = FALLBACK_MODEL_CHOICES.get(provider, [])
    if not api_key:
        return fallback
    try:
        if provider == "gemini":
            return _list_gemini_models(api_key) or fallback
        if provider == "openai":
            return _list_openai_models(api_key) or fallback
        if provider == "anthropic":
            return _list_anthropic_models(api_key) or fallback
    except Exception as e:  # noqa: BLE001
        logger.warning("Live model listing failed; using fallback", provider=provider, error=str(e))
    return fallback


def _list_gemini_models(api_key: str) -> List[str]:
    try:
        from google import genai  # type: ignore
        client = genai.Client(api_key=api_key)
        names = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", [])
            if actions and "generateContent" not in actions:
                continue
            name = getattr(m, "name", "")
            names.append(name.split("/")[-1] if name else "")
    except ImportError:
        import google.generativeai as genai_legacy  # type: ignore
        genai_legacy.configure(api_key=api_key)
        names = [
            m.name.split("/")[-1]
            for m in genai_legacy.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
    return sorted({n for n in names if n})


def _list_openai_models(api_key: str) -> List[str]:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=api_key)
    ids = [m.id for m in client.models.list().data]
    # Keep only chat-capable gpt models; drop embeddings/audio/etc.
    chat = [i for i in ids if i.startswith("gpt") and "instruct" not in i]
    return sorted(chat) or sorted(ids)


def _list_anthropic_models(api_key: str) -> List[str]:
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=api_key)
    return [m.id for m in client.models.list().data]
