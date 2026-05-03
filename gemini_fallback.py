"""
gemini_fallback.py
==================
Drop-in fallback: if local TinyLlama takes > 30 seconds,
automatically retries the query through Google Gemini's free API.

HOW TO USE
----------
1. Get a free Gemini API key:
   https://aistudio.google.com/app/apikey  (completely free, no credit card)

2. Set it as an environment variable OR paste it in GEMINI_API_KEY below:
   export GEMINI_API_KEY="AIza..."

3. In inference_api.py, add near the top (after CONFIG):
   from gemini_fallback import GeminiFallback
   gemini_fallback = GeminiFallback(api_key=os.getenv("GEMINI_API_KEY", ""))

4. Replace the _generate() call in MultiCharacterChatbot.generate_answer()
   with the patched version at the bottom of this file.

WHAT IT DOES
------------
- Wraps every local model call in a 30-second timeout
- On timeout (or any generation error), calls Gemini Flash (free tier)
- Gemini is prompted with the same character persona + RAG context
- Returns the answer in the same dict format the rest of the system expects
- Falls back gracefully if Gemini also fails
"""

import os
import time
import threading
import json
import urllib.request
import urllib.error
from typing import Optional, Dict

# ── Gemini free model — gemini-1.5-flash is free up to 15 req/min ─────────────
GEMINI_MODEL   = "gemini-1.5-flash"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
GEMINI_TIMEOUT_SECONDS = 20   # Gemini API call timeout
LOCAL_TIMEOUT_SECONDS  = 30   # If local model exceeds this → use Gemini


# =============================================================================
# CHARACTER SYSTEM PROMPTS  (mirrors CHARACTERS dict in inference_api.py)
# =============================================================================

CHARACTER_PROMPTS = {
    "king": (
        "You are King Sri Wikrama Rajasinha, the last sovereign of the Kingdom of Kandy "
        "(1780–1832). You are royal, wise, dignified, and deeply protective of Buddhist heritage. "
        "You speak formally and with authority, using 'I', 'we', 'our kingdom'. "
        "Answer only from the historical knowledge provided. Stay fully in character."
    ),
    "nilame": (
        "You are the Head Thero of Sri Dalada Maligawa — the Temple of the Sacred Tooth Relic. "
        "You are devoted, ceremonial, and knowledgeable about Buddhist rituals and the Esala Perahera. "
        "You speak with reverence, using words like 'sacred' and 'blessed'. "
        "Answer only from the historical knowledge provided. Stay fully in character."
    ),
    "dutch": (
        "You are Captain Willem van der Berg of the Dutch East India Company (VOC), "
        "stationed at Galle Fort from 1740–1755. You are military, strategic, and pragmatic. "
        "You speak with precision, occasionally using Dutch terms. "
        "Answer only from the historical knowledge provided. Stay fully in character."
    ),
    "citizen": (
        "You are Rathnayake Mudalige Sunil, a modern Sri Lankan historian and heritage guide. "
        "You are friendly, educational, and proud of Sri Lankan heritage. "
        "You speak warmly and use modern examples for comparison. "
        "Answer only from the historical knowledge provided. Stay fully in character."
    ),
}

EXPERTISE_SUFFIXES = {
    "child": (
        "Speak very simply, like explaining to a curious 8-year-old. "
        "Use short sentences and fun comparisons. Add enthusiasm!"
    ),
    "student": (
        "Speak clearly and educationally. Connect events to causes and effects. "
        "Be engaging and informative."
    ),
    "researcher": (
        "Speak with academic precision. Use correct historical terminology. "
        "Cite specific dates and names where possible. Be thorough."
    ),
    "tourist": (
        "Be warm, enthusiastic, and descriptive. Paint a vivid picture. "
        "Include interesting anecdotes. Be a great tour guide!"
    ),
}


# =============================================================================
# GEMINI FALLBACK CLASS
# =============================================================================

class GeminiFallback:
    """
    Wraps local model generation with a timeout.
    Falls back to Gemini 1.5 Flash (free) when local model is too slow.
    """

    def __init__(self, api_key: str = "", timeout: int = LOCAL_TIMEOUT_SECONDS):
        self.api_key  = api_key.strip()
        self.timeout  = timeout
        self.enabled  = bool(self.api_key)
        self.call_count   = 0
        self.fallback_count = 0
        self._lock    = threading.Lock()

        if self.enabled:
            print(f"[GeminiFallback] ✅ Ready — model: {GEMINI_MODEL} | timeout: {timeout}s")
        else:
            print("[GeminiFallback] ⚠️  No API key — fallback disabled. "
                  "Set GEMINI_API_KEY env var or pass api_key= to enable.")

    # ── public entry point ────────────────────────────────────────────────────

    def generate_with_fallback(
        self,
        local_fn,           # callable: () → str   (the local _generate call)
        query: str,
        char_id: str,
        context: str = "",
        expertise_level: str = "tourist",
    ) -> Dict:
        """
        Call local_fn() with a timeout.
        If it times out or fails, call Gemini instead.

        Returns:
            {
                "answer":   str,
                "source":   "local" | "gemini" | "fallback_error",
                "latency":  float,
                "model":    str,
            }
        """
        with self._lock:
            self.call_count += 1

        t0 = time.time()

        # ── Try local model with timeout ──────────────────────────────────────
        result_holder = {"value": None, "error": None}

        def _run_local():
            try:
                result_holder["value"] = local_fn()
            except Exception as e:
                result_holder["error"] = str(e)

        thread = threading.Thread(target=_run_local, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        elapsed = time.time() - t0

        if thread.is_alive():
            # Local model timed out
            print(f"[GeminiFallback] ⏱  Local model exceeded {self.timeout}s "
                  f"— switching to Gemini for: '{query[:60]}...'")
        elif result_holder["error"]:
            print(f"[GeminiFallback] ❌ Local model error: {result_holder['error'][:80]} "
                  f"— switching to Gemini")
        elif result_holder["value"] and len(result_holder["value"].strip()) >= 20:
            # Local model succeeded in time
            return {
                "answer":  result_holder["value"],
                "source":  "local",
                "latency": round(elapsed, 2),
                "model":   "TinyLlama-1.1B",
            }
        else:
            print(f"[GeminiFallback] ⚠️  Local model returned empty/short answer "
                  f"— switching to Gemini")

        # ── Fallback to Gemini ────────────────────────────────────────────────
        with self._lock:
            self.fallback_count += 1

        if not self.enabled:
            return {
                "answer":  (
                    f"I apologize — the response is taking longer than expected. "
                    f"Please try again or ask a simpler question."
                ),
                "source":  "fallback_error",
                "latency": round(time.time() - t0, 2),
                "model":   "none",
            }

        gemini_answer = self._call_gemini(query, char_id, context, expertise_level)
        return {
            "answer":  gemini_answer,
            "source":  "gemini",
            "latency": round(time.time() - t0, 2),
            "model":   GEMINI_MODEL,
        }

    # ── Gemini API call ───────────────────────────────────────────────────────

    def _build_prompt(self, query: str, char_id: str,
                      context: str, expertise_level: str) -> str:
        char_prompt = CHARACTER_PROMPTS.get(char_id, CHARACTER_PROMPTS["citizen"])
        expertise   = EXPERTISE_SUFFIXES.get(expertise_level, EXPERTISE_SUFFIXES["tourist"])

        parts = [char_prompt, expertise]
        if context.strip():
            parts.append(f"\nHistorical Knowledge to draw from:\n{context.strip()[:3000]}")
        parts.append(f"\nUser question: {query}")
        parts.append("\nAnswer in character (2–4 sentences, historically accurate):")
        return "\n\n".join(parts)

    def _call_gemini(self, query: str, char_id: str,
                     context: str, expertise_level: str) -> str:
        prompt = self._build_prompt(query, char_id, context, expertise_level)
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature":     0.7,
                "maxOutputTokens": 300,
                "topP":            0.9,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }).encode("utf-8")

        url = f"{GEMINI_API_URL}?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Extract text from Gemini response structure
            candidates = data.get("candidates", [])
            if not candidates:
                return self._error_answer(char_id, "No candidates in Gemini response")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return self._error_answer(char_id, "Empty parts in Gemini response")

            text = parts[0].get("text", "").strip()
            if not text:
                return self._error_answer(char_id, "Empty text in Gemini response")

            print(f"[GeminiFallback] ✅ Gemini answered ({len(text)} chars)")
            return text

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                err_msg = json.loads(body).get("error", {}).get("message", body)
            except Exception:
                err_msg = body[:200]

            if e.code == 429:
                print("[GeminiFallback] ⚠️  Rate limit hit (15 req/min free tier)")
                return self._error_answer(char_id, "rate_limit")
            elif e.code == 400:
                print(f"[GeminiFallback] ❌ Bad request: {err_msg[:100]}")
                return self._error_answer(char_id, err_msg)
            else:
                print(f"[GeminiFallback] ❌ HTTP {e.code}: {err_msg[:100]}")
                return self._error_answer(char_id, f"HTTP {e.code}")

        except Exception as e:
            print(f"[GeminiFallback] ❌ Unexpected error: {str(e)[:100]}")
            return self._error_answer(char_id, str(e))

    def _error_answer(self, char_id: str, reason: str) -> str:
        char_names = {
            "king":    "King Sri Wikrama Rajasinha",
            "nilame":  "the Head Thero",
            "dutch":   "Captain van der Berg",
            "citizen": "Sunil",
        }
        name = char_names.get(char_id, "I")
        if "rate_limit" in reason:
            return (
                f"I, {name}, apologize — our communication channel is momentarily "
                f"overwhelmed. Please ask again in a moment."
            )
        return (
            f"I, {name}, must beg your pardon — I am unable to provide a response "
            f"at this moment. Please try rephrasing your question."
        )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "enabled":        self.enabled,
            "model":          GEMINI_MODEL if self.enabled else "disabled",
            "total_calls":    self.call_count,
            "fallback_calls": self.fallback_count,
            "fallback_rate":  (
                round(self.fallback_count / self.call_count * 100, 1)
                if self.call_count else 0
            ),
            "local_timeout_s":  self.timeout,
            "gemini_timeout_s": GEMINI_TIMEOUT_SECONDS,
        }


# =============================================================================
# PATCH FUNCTION — call this once after creating MultiCharacterChatbot
# =============================================================================

def patch_chatbot_with_gemini(chatbot, api_key: str = ""):
    """
    Monkey-patches chatbot._generate() to use GeminiFallback.
    Call this once after creating the chatbot instance:

        from gemini_fallback import patch_chatbot_with_gemini
        patch_chatbot_with_gemini(chatbot, api_key=os.getenv("GEMINI_API_KEY",""))

    After patching, every generate_answer() call automatically uses Gemini
    if the local model takes more than 30 seconds.
    """
    import torch

    fallback = GeminiFallback(
        api_key=api_key or os.getenv("GEMINI_API_KEY", ""),
        timeout=LOCAL_TIMEOUT_SECONDS,
    )
    chatbot._gemini_fallback = fallback

    # Save the original _generate method
    original_generate = chatbot._generate

    # Store context for Gemini (set by generate_answer before calling _generate)
    chatbot._current_context     = ""
    chatbot._current_query       = ""
    chatbot._current_char_id     = "citizen"
    chatbot._current_expertise   = "tourist"

    def patched_generate(prompt: str) -> str:
        """
        Replacement for chatbot._generate(prompt).
        Runs local model with timeout; falls back to Gemini on slow/failure.
        """
        def local_fn():
            return original_generate(prompt)

        result = fallback.generate_with_fallback(
            local_fn         = local_fn,
            query            = chatbot._current_query,
            char_id          = chatbot._current_char_id,
            context          = chatbot._current_context,
            expertise_level  = chatbot._current_expertise,
        )

        # Log the source for transparency
        source = result.get("source", "local")
        if source == "gemini":
            print(f"[GeminiFallback] Answer delivered via Gemini "
                  f"(latency: {result['latency']}s)")
        elif source == "fallback_error":
            print("[GeminiFallback] Both local and Gemini failed")

        return result["answer"]

    # Override the method on the instance
    import types
    chatbot._generate = types.MethodType(
        lambda self, prompt: patched_generate(prompt), chatbot
    )

    # Also patch generate_answer to set context before calling _generate
    original_generate_answer = chatbot.generate_answer

    def patched_generate_answer(query, char_id, session_id="default",
                                expertise_level=None):
        # Set context variables so patched_generate can pass them to Gemini
        chatbot._current_query     = query
        chatbot._current_char_id   = char_id
        chatbot._current_expertise = (
            expertise_level
            or chatbot.expertise_adapter.get_user_level(session_id)
            or "tourist"
        )
        # Context will be populated by RAG inside generate_answer;
        # we approximate it here for Gemini by using the query itself
        chatbot._current_context = query

        return original_generate_answer(query, char_id, session_id, expertise_level)

    chatbot.generate_answer = patched_generate_answer

    print(f"[GeminiFallback] ✅ Chatbot patched — "
          f"Gemini fallback {'ACTIVE' if fallback.enabled else 'DISABLED (no key)'}")
    return fallback


# =============================================================================
# FLASK ROUTE — add to your existing Flask app
# =============================================================================

def register_gemini_routes(app, chatbot):
    """
    Registers /gemini/stats endpoint.
    Call after create_flask_api():
        from gemini_fallback import register_gemini_routes
        register_gemini_routes(flask_app, chatbot)
    """
    from flask import jsonify

    @app.route("/gemini/stats", methods=["GET"])
    def gemini_stats():
        fb = getattr(chatbot, "_gemini_fallback", None)
        if not fb:
            return jsonify({"error": "Gemini fallback not initialized"}), 404
        return jsonify({"success": True, "stats": fb.get_stats()})

    @app.route("/gemini/test", methods=["GET", "POST"])
    def gemini_test():
        """Quick test endpoint — sends a fixed query directly to Gemini."""
        from flask import request as freq
        fb = getattr(chatbot, "_gemini_fallback", None)
        if not fb or not fb.enabled:
            return jsonify({"error": "Gemini fallback not enabled — set GEMINI_API_KEY"}), 503

        data    = freq.get_json(force=True, silent=True) or {}
        query   = data.get("query", "Tell me about the Sacred Tooth Relic")
        char_id = data.get("character_id", "king")

        answer = fb._call_gemini(query, char_id, context="", expertise_level="tourist")
        return jsonify({
            "success":      True,
            "character_id": char_id,
            "query":        query,
            "answer":       answer,
            "model":        GEMINI_MODEL,
            "stats":        fb.get_stats(),
        })

    print("[GeminiFallback] Routes registered: /gemini/stats, /gemini/test")