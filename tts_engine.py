"""
tts_engine.py — Free Text-to-Speech for all 4 Historical Characters
====================================================================
Uses gTTS (Google TTS, free, no API key needed) for the Flask /tts endpoint.
Uses Web Speech API (browser-native, zero cost) for the Gradio UI.

Character voice profiles:
  king    → slow, deep — en (UK accent, slow rate)
  nilame  → calm, spiritual — en-IN (Indian English)
  dutch   → brisk, authoritative — en (US, normal rate)
  citizen → friendly, conversational — en-AU (Australian)

Install:  pip install gtts
"""

import io
import base64
import threading
from pathlib import Path
from typing import Optional

# ── gTTS import (graceful fallback if not installed) ──────────────────────────
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("WARNING: gTTS not installed. Run: pip install gtts")
    print("         Flask /tts endpoint will return 503 until installed.")

# ── Character voice profiles ──────────────────────────────────────────────────
# Each profile drives BOTH the gTTS server-side call AND the
# Web Speech API client-side config returned in the API response.

CHARACTER_VOICE_PROFILES = {
    "king": {
        "display_name":   "King Sri Wikrama Rajasinha",
        # gTTS settings
        "gtts_lang":      "en",
        "gtts_tld":       "co.uk",   # British accent
        "gtts_slow":      True,      # majestic, slow delivery
        # Web Speech API hints (client uses these)
        "wsa_lang":       "en-GB",
        "wsa_rate":       0.80,      # 0.1–2.0 (1.0 = normal)
        "wsa_pitch":      0.85,      # 0–2 (1.0 = normal)
        "wsa_volume":     1.0,
        "wsa_voice_name": "Google UK English Male",  # preferred (may not exist on all devices)
        "description":    "Slow, deep, regal — British English accent",
    },
    "nilame": {
        "display_name":   "Thero (Chief Custodian)",
        "gtts_lang":      "en",
        "gtts_tld":       "co.in",   # Indian English accent
        "gtts_slow":      True,
        "wsa_lang":       "en-IN",
        "wsa_rate":       0.82,
        "wsa_pitch":      1.05,
        "wsa_volume":     1.0,
        "wsa_voice_name": "Google हिन्दी",
        "description":    "Calm, ceremonial — Indian English accent",
    },
    "dutch": {
        "display_name":   "Captain Willem van der Berg",
        "gtts_lang":      "en",
        "gtts_tld":       "com",     # American English
        "gtts_slow":      False,     # brisk military delivery
        "wsa_lang":       "en-US",
        "wsa_rate":       1.05,
        "wsa_pitch":      0.90,
        "wsa_volume":     1.0,
        "wsa_voice_name": "Google US English",
        "description":    "Brisk, authoritative — American English accent",
    },
    "citizen": {
        "display_name":   "Rathnayake Mudalige Sunil",
        "gtts_lang":      "en",
        "gtts_tld":       "com.au",  # Australian English (friendly lilt)
        "gtts_slow":      False,
        "wsa_lang":       "en-AU",
        "wsa_rate":       0.95,
        "wsa_pitch":      1.10,
        "wsa_volume":     1.0,
        "wsa_voice_name": "Google Australian English",
        "description":    "Warm, conversational — Australian English accent",
    },
}

# ── Simple in-memory audio cache (avoids re-generating identical text) ─────────
_tts_cache: dict = {}
_cache_lock = threading.Lock()
_MAX_CACHE = 100   # keep last 100 entries


def _cache_key(text: str, char_id: str) -> str:
    import hashlib
    return hashlib.md5(f"{char_id}:{text[:200]}".encode()).hexdigest()


def synthesize_to_mp3_bytes(text: str, char_id: str) -> Optional[bytes]:
    """
    Convert text to MP3 bytes using gTTS with the character's voice profile.
    Returns None if gTTS is unavailable or synthesis fails.
    Caches results in memory.
    """
    if not GTTS_AVAILABLE:
        return None

    key = _cache_key(text, char_id)
    with _cache_lock:
        if key in _tts_cache:
            return _tts_cache[key]

    profile = CHARACTER_VOICE_PROFILES.get(char_id, CHARACTER_VOICE_PROFILES["citizen"])

    try:
        tts = gTTS(
            text=text,
            lang=profile["gtts_lang"],
            tld=profile["gtts_tld"],
            slow=profile["gtts_slow"],
        )
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        mp3_bytes = buf.getvalue()

        with _cache_lock:
            if len(_tts_cache) >= _MAX_CACHE:
                # Evict oldest entry
                oldest = next(iter(_tts_cache))
                del _tts_cache[oldest]
            _tts_cache[key] = mp3_bytes

        return mp3_bytes

    except Exception as e:
        print(f"[TTS] gTTS synthesis error for char={char_id}: {e}")
        return None


def synthesize_to_base64(text: str, char_id: str) -> Optional[str]:
    """Returns base64-encoded MP3 string (for embedding in JSON responses)."""
    mp3 = synthesize_to_mp3_bytes(text, char_id)
    if mp3 is None:
        return None
    return base64.b64encode(mp3).decode("utf-8")


def get_voice_profile(char_id: str) -> dict:
    """Return the full voice profile for a character (safe copy)."""
    return dict(CHARACTER_VOICE_PROFILES.get(char_id, CHARACTER_VOICE_PROFILES["citizen"]))


def register_tts_routes(app, chatbot=None):
    """
    Register all /tts/* Flask routes onto the given Flask app.
    Call this from create_flask_api() after the app is created.

    Usage in inference_api.py:
        from tts_engine import register_tts_routes
        register_tts_routes(flask_app, chatbot)
    """
    from flask import request, jsonify, send_file, Response

    # ── GET /tts/voices — list all character voices ───────────────────────────
    @app.route("/tts/voices", methods=["GET"])
    def tts_voices():
        """
        Returns all character voice profiles including Web Speech API hints.
        The VR app uses these to configure client-side speech synthesis.
        """
        profiles = {}
        for cid, p in CHARACTER_VOICE_PROFILES.items():
            profiles[cid] = {
                "character_id":   cid,
                "display_name":   p["display_name"],
                "description":    p["description"],
                "gtts_available": GTTS_AVAILABLE,
                "web_speech_api": {
                    "lang":       p["wsa_lang"],
                    "rate":       p["wsa_rate"],
                    "pitch":      p["wsa_pitch"],
                    "volume":     p["wsa_volume"],
                    "voice_name": p["wsa_voice_name"],
                },
            }
        return jsonify({
            "success":            True,
            "gtts_available":     GTTS_AVAILABLE,
            "total_characters":   len(profiles),
            "voices":             profiles,
            "usage_note": (
                "For browser playback: use web_speech_api config with window.speechSynthesis. "
                "For server-side MP3: POST to /tts/speak or GET /tts/stream/<char_id>"
            ),
        })

    # ── POST /tts/speak — return base64 MP3 in JSON ───────────────────────────
    @app.route("/tts/speak", methods=["POST"])
    def tts_speak():
        """
        POST body: {"text": "...", "character_id": "king"}
        Returns: JSON with base64-encoded MP3 + voice profile for client use.

        The client can:
          1. Decode the base64 and play it as an <audio> element, OR
          2. Use the web_speech_api config to synthesize locally (no server round-trip)
        """
        if not GTTS_AVAILABLE:
            return jsonify({
                "success": False,
                "error":   "gTTS not installed. Run: pip install gtts",
                "fallback": "Use web_speech_api config from GET /tts/voices instead",
            }), 503

        data    = request.get_json(force=True, silent=True) or {}
        text    = (data.get("text") or "").strip()
        char_id = data.get("character_id", "citizen")

        if not text:
            return jsonify({"success": False, "error": "'text' field is required"}), 400
        if len(text) > 3000:
            return jsonify({"success": False, "error": "Text too long (max 3000 chars)"}), 400
        if char_id not in CHARACTER_VOICE_PROFILES:
            return jsonify({
                "success": False,
                "error":   f"Unknown character_id '{char_id}'",
                "valid":   list(CHARACTER_VOICE_PROFILES.keys()),
            }), 400

        b64 = synthesize_to_base64(text, char_id)
        if b64 is None:
            return jsonify({"success": False, "error": "TTS synthesis failed"}), 500

        profile = get_voice_profile(char_id)
        return jsonify({
            "success":        True,
            "character_id":   char_id,
            "display_name":   profile["display_name"],
            "text_length":    len(text),
            "audio_format":   "mp3",
            "audio_base64":   b64,
            "web_speech_api": {
                "lang":       profile["wsa_lang"],
                "rate":       profile["wsa_rate"],
                "pitch":      profile["wsa_pitch"],
                "volume":     profile["wsa_volume"],
                "voice_name": profile["wsa_voice_name"],
            },
            "play_hint": "Decode audio_base64 → Blob → URL.createObjectURL → <audio>.play()",
        })

    # ── GET /tts/stream/<char_id> — stream raw MP3 file directly ─────────────
    @app.route("/tts/stream/<char_id>", methods=["GET", "POST"])
    def tts_stream(char_id):
        """
        Stream an MP3 file directly — use as <audio src="/tts/stream/king?text=Hello">
        GET  ?text=Hello+World&character_id=king
        POST {"text": "Hello", "character_id": "king"}
        """
        if not GTTS_AVAILABLE:
            return Response(
                "gTTS not installed. Run: pip install gtts",
                status=503, mimetype="text/plain"
            )

        if request.method == "POST":
            data = request.get_json(force=True, silent=True) or {}
            text = (data.get("text") or "").strip()
        else:
            text = (request.args.get("text") or "").strip()

        if not text:
            return Response("'text' parameter is required", status=400, mimetype="text/plain")
        if len(text) > 3000:
            return Response("Text too long (max 3000 chars)", status=400, mimetype="text/plain")
        if char_id not in CHARACTER_VOICE_PROFILES:
            return Response(
                f"Unknown char_id. Valid: {list(CHARACTER_VOICE_PROFILES.keys())}",
                status=404, mimetype="text/plain"
            )

        mp3_bytes = synthesize_to_mp3_bytes(text, char_id)
        if mp3_bytes is None:
            return Response("TTS synthesis failed", status=500, mimetype="text/plain")

        return send_file(
            io.BytesIO(mp3_bytes),
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name=f"{char_id}_tts.mp3",
        )

    # ── POST /tts/chat-and-speak — ask a character AND get audio in one call ──
    @app.route("/tts/chat-and-speak", methods=["POST"])
    def tts_chat_and_speak():
        """
        Combined endpoint: ask a question → get text answer + MP3 audio.
        Body: {"query": "...", "character_id": "king", "session_id": "user_1"}
        Returns: answer text + base64 MP3 + web_speech_api config
        """
        if chatbot is None:
            return jsonify({"success": False, "error": "Chatbot not attached"}), 503

        data       = request.get_json(force=True, silent=True) or {}
        query      = (data.get("query") or data.get("question") or "").strip()
        char_id    = data.get("character_id", "king")
        session_id = data.get("session_id", "default")

        if not query:
            return jsonify({"success": False, "error": "'query' field is required"}), 400
        if char_id not in CHARACTER_VOICE_PROFILES:
            return jsonify({
                "success": False,
                "error":   f"Unknown character_id '{char_id}'",
                "valid":   list(CHARACTER_VOICE_PROFILES.keys()),
            }), 400

        # Get text answer from chatbot
        try:
            resp   = chatbot.generate_answer(query, char_id, session_id)
            answer = resp.get("answer", "")
        except Exception as e:
            return jsonify({"success": False, "error": f"Chatbot error: {e}"}), 500

        # Synthesize answer to audio
        audio_b64 = synthesize_to_base64(answer, char_id) if GTTS_AVAILABLE else None
        profile   = get_voice_profile(char_id)

        return jsonify({
            "success":        True,
            "character_id":   char_id,
            "display_name":   profile["display_name"],
            "question":       query,
            "answer":         answer,
            "confidence":     resp.get("confidence"),
            "intent":         resp.get("intent"),
            "topic":          resp.get("topic"),
            "session_id":     session_id,
            "audio_available": audio_b64 is not None,
            "audio_format":   "mp3" if audio_b64 else None,
            "audio_base64":   audio_b64,
            "web_speech_api": {
                "lang":       profile["wsa_lang"],
                "rate":       profile["wsa_rate"],
                "pitch":      profile["wsa_pitch"],
                "volume":     profile["wsa_volume"],
                "voice_name": profile["wsa_voice_name"],
            },
        })

    print("[TTS] Routes registered: /tts/voices, /tts/speak, /tts/stream/<char_id>, /tts/chat-and-speak")
    return app