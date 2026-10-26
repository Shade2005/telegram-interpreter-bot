#!/usr/bin/env python3
"""
Hindi/Urdu <-> Gulf Arabic Two-Way WhatsApp & Telegram Voice Interpreter Bot.

Workflow:
1. User receives a voice message on WhatsApp (e.g., from an Arabic speaker).
2. User shares/forwards the voice note directly to this bot on Telegram.
3. Bot detects the language (Arabic), transcribes it, and translates it into Urdu/Hindi.
4. Bot asks the user: "What would you like to reply?"
5. User replies in Urdu/Hindi (via voice note OR text).
6. Bot translates user's reply into spoken Gulf Arabic (Khaleeji dialect) and synthesizes
   an authentic Arabic voice note using edge-tts (ar-AE-HamdanNeural).
7. User can tap & hold the voice note to share it directly back to their WhatsApp contact!
8. Also supports standalone Urdu/Hindi-to-Arabic voice translations anytime.
"""

import json
import logging
import os
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
import edge_tts
from google import genai
from google.genai import types
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Environment & Configuration Setup
# ---------------------------------------------------------------------------
# Ensure stdout/stderr handle UTF-8 on Windows consoles without charmap errors
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables from .env file (prioritizing .env file)
load_dotenv(override=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest").strip()
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "ar-AE-HamdanNeural").strip()

# Automatically upgrade deprecated model strings to active models
if GEMINI_MODEL in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash-lite"]:
    GEMINI_MODEL = "gemini-flash-lite-latest"

# Keep GOOGLE_API_KEY in sync with GEMINI_API_KEY
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Set up logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("VoiceInterpreterBot")

# System instruction for two-way translation
SYSTEM_PROMPT = (
    "You are an expert bilingual interpreter between Hindi/Urdu and Gulf Arabic (Khaleeji dialect). "
    "Analyze the input (audio or text) and determine whether it is spoken/written in Arabic or Hindi/Urdu:\n\n"
    "1. If the input is in Arabic (Gulf/Khaleeji, Saudi, Emirati, or general Arabic):\n"
    "   - Transcribe what was spoken/written in Arabic accurately.\n"
    "   - Translate it into clear, natural, spoken Urdu/Hindi so an Urdu/Hindi speaker easily understands.\n"
    "   - Set 'detected_language': 'arabic'\n"
    "   - Set 'transcription': <original Arabic text>\n"
    "   - Set 'translation': <Urdu/Hindi translation>\n"
    "   - Set 'arabic_text': <original Arabic text>\n\n"
    "2. If the input is in Hindi or Urdu:\n"
    "   - Transcribe what was spoken/written in Urdu/Hindi.\n"
    "   - Translate it into natural, authentic spoken Gulf Arabic (Khaleeji dialect) as used in UAE, Saudi Arabia, Kuwait, Qatar.\n"
    "   - Set 'detected_language': 'urdu_hindi'\n"
    "   - Set 'transcription': <original Urdu/Hindi text>\n"
    "   - Set 'translation': <Gulf Arabic translation>\n"
    "   - Set 'arabic_text': <Gulf Arabic translation>\n\n"
    "Output strictly in JSON format with these exact four keys: "
    "'detected_language', 'transcription', 'translation', 'arabic_text'."
)

# Initialize Gemini Client if API key is provided
gemini_client: Optional[genai.Client] = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"Initialized Gemini client successfully with model: {GEMINI_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
else:
    logger.warning("GEMINI_API_KEY is not set in environment or .env file.")

# In-memory session tracking for WhatsApp conversation flows:
# chat_id -> {"awaiting_reply": bool, "last_arabic_message": str}
user_sessions: Dict[int, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Core Translation & Speech Processing Functions
# ---------------------------------------------------------------------------
def parse_gemini_json_response(raw_text: str) -> Dict[str, str]:
    """
    Parses and sanitizes the JSON response returned by Gemini.
    Removes Markdown code blocks if present (```json ... ```).
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            transcription = str(data.get("transcription", "")).strip()
            translation = str(data.get("translation", "")).strip()
            arabic_text = str(data.get("arabic_text", "")).strip()
            detected_lang = str(data.get("detected_language", "")).strip().lower()

            # Backwards compatibility fallback with 'arabic_translation'
            if not translation and "arabic_translation" in data:
                translation = str(data.get("arabic_translation", "")).strip()
            if not arabic_text:
                arabic_text = str(data.get("arabic_translation", "")).strip() or translation

            return {
                "detected_language": detected_lang or ("arabic" if "arabic" in detected_lang else "urdu_hindi"),
                "transcription": transcription,
                "translation": translation,
                "arabic_text": arabic_text,
                "arabic_translation": arabic_text,
            }
    except json.JSONDecodeError as err:
        logger.error(f"Failed to decode JSON from Gemini output: {cleaned} - Error: {err}")

    return {
        "detected_language": "urdu_hindi",
        "transcription": "",
        "translation": "",
        "arabic_text": "",
        "arabic_translation": "",
    }


async def interpret_audio_with_gemini(
    audio_path: str,
    client: genai.Client,
    model_name: str = GEMINI_MODEL,
) -> Dict[str, str]:
    """
    Feeds the raw audio file directly to Gemini to detect language,
    transcribe, and translate between Urdu/Hindi and Gulf Arabic.
    """
    logger.info(f"Reading audio file for Gemini: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Determine audio MIME type based on file extension
    ext = os.path.splitext(audio_path)[1].lower()
    mime_type_map = {
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".opus": "audio/ogg",
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".m4a": "audio/m4a",
        ".aac": "audio/aac",
    }
    mime_type = mime_type_map.get(ext, "audio/ogg")

    # Pass audio bytes directly as an audio Part
    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

    models_to_try = [model_name]
    for fallback in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    response = None
    last_err = None
    for current_model in models_to_try:
        try:
            logger.info(f"Sending audio to Gemini ({current_model})...")
            response = client.models.generate_content(
                model=current_model,
                contents=[
                    "Please interpret this voice message according to system instructions.",
                    audio_part,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            if response and response.text:
                break
        except Exception as err:
            logger.warning(f"Model {current_model} call failed: {err}")
            last_err = err
            continue

    if not response or not response.text:
        raise ValueError(f"Gemini generation failed across models: {last_err}")

    logger.debug(f"Raw Gemini response: {response.text}")
    parsed = parse_gemini_json_response(response.text)

    if not parsed.get("transcription") and not parsed.get("translation"):
        raise ValueError(f"Could not extract transcription from response: {response.text}")

    return parsed


async def translate_text_with_gemini(
    text: str,
    client: genai.Client,
    model_name: str = GEMINI_MODEL,
) -> Dict[str, str]:
    """
    Translates user text input (Urdu/Hindi to Gulf Arabic or vice-versa).
    """
    models_to_try = [model_name]
    for fallback in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"]:
        if fallback not in models_to_try:
            models_to_try.append(fallback)

    response = None
    last_err = None
    for current_model in models_to_try:
        try:
            logger.info(f"Translating text with Gemini ({current_model}): {text[:50]}...")
            response = client.models.generate_content(
                model=current_model,
                contents=[
                    "Translate this text according to system instructions:",
                    text,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            if response and response.text:
                break
        except Exception as err:
            logger.warning(f"Model {current_model} text translation failed: {err}")
            last_err = err
            continue

    if not response or not response.text:
        raise ValueError(f"Gemini text translation failed: {last_err}")

    return parse_gemini_json_response(response.text)


async def text_to_speech_edge(
    text: str,
    output_path: str,
    voice: str = EDGE_TTS_VOICE,
) -> str:
    """
    Converts Arabic text to speech using edge-tts with a Gulf Arabic voice.
    """
    logger.info(f"Generating Arabic speech with voice '{voice}' for: {text}")
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)
    logger.info(f"Speech generated successfully at: {output_path}")
    return output_path


def cleanup_files(*file_paths: Optional[str]) -> None:
    """
    Safely removes temporary files from disk.
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")


# ---------------------------------------------------------------------------
# Telegram Bot Command Handlers
# ---------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles /start command.
    """
    user_name = update.effective_user.first_name if update.effective_user else "there"
    welcome_message = (
        f"👋 Hello {user_name}!\n\n"
        "🎙️ **WhatsApp & Telegram Voice Interpreter**\n"
        "*(Urdu / Hindi 🔁 Gulf Arabic / Khaleeji)*\n\n"
        "📲 **How to use with WhatsApp:**\n"
        "1. In WhatsApp, tap & hold any voice note you received.\n"
        "2. Tap **Share ➔ Telegram ➔ This Bot**.\n"
        "3. I will translate it into Urdu/Hindi and ask what you want to reply.\n"
        "4. Reply to me with a voice note or text in Urdu/Hindi.\n"
        "5. I will create a **Gulf Arabic voice note** for you to share back to WhatsApp!\n\n"
        "💡 You can also record your own voice note in Urdu/Hindi anytime to get an Arabic voice note."
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles /help command.
    """
    help_message = (
        "💡 **Help & Instructions:**\n\n"
        "• **WhatsApp Voice Notes**: Share or forward any voice note from WhatsApp directly to this bot.\n"
        "• **Arabic ➔ Urdu**: When someone speaks Arabic, I transcribe and translate it to Urdu/Hindi.\n"
        "• **Urdu ➔ Arabic Voice Reply**: When you reply, I generate a natural Gulf Arabic voice message.\n"
        "• **Share Back**: Tap & hold the generated voice note ➔ Share ➔ WhatsApp.\n"
        "• **Reset**: Send `/cancel` or `/reset` to start fresh.\n\n"
        f"⚙️ **Voice:** `{EDGE_TTS_VOICE}` (Gulf Arabic)\n"
        f"🤖 **Model:** `{GEMINI_MODEL}`"
    )
    await update.message.reply_text(help_message, parse_mode="Markdown")


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Resets the conversation state.
    """
    chat_id = update.effective_chat.id
    if chat_id in user_sessions:
        del user_sessions[chat_id]
    await update.message.reply_text(
        "🔄 Session reset. You can share an incoming WhatsApp voice note or send an Urdu/Hindi message anytime!"
    )


# ---------------------------------------------------------------------------
# Message Handlers (Audio & Text)
# ---------------------------------------------------------------------------
async def extract_audio_file(update: Update) -> Optional[Tuple[Any, str]]:
    """
    Extracts the audio file object and filename whether sent as a voice note,
    audio file, or document (common when sharing from WhatsApp).
    """
    msg = update.message
    if not msg:
        return None

    if msg.voice:
        voice_file = await msg.voice.get_file()
        return voice_file, "voice.ogg"

    if msg.audio:
        audio_file = await msg.audio.get_file()
        filename = msg.audio.file_name or "audio.mp3"
        return audio_file, filename

    if msg.document:
        mime = (msg.document.mime_type or "").lower()
        name = (msg.document.file_name or "").lower()
        audio_extensions = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".aac", ".oga", ".flac")
        if mime.startswith("audio/") or any(name.endswith(ext) for ext in audio_extensions):
            doc_file = await msg.document.get_file()
            return doc_file, (msg.document.file_name or "audio.ogg")

    return None


async def handle_incoming_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles incoming voice notes, audio files, or WhatsApp shared media.
    """
    if not update.message:
        return

    audio_info = await extract_audio_file(update)
    if not audio_info:
        return

    file_obj, original_filename = audio_info

    if not gemini_client:
        await update.message.reply_text(
            "⚠️ Bot configuration error: `GEMINI_API_KEY` is not set.\n"
            "Please check the `.env` settings and restart.",
            parse_mode="Markdown",
        )
        return

    chat_id = update.effective_chat.id
    session_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(original_filename)[1] or ".ogg"
    temp_dir = tempfile.gettempdir()
    input_audio_path = os.path.join(temp_dir, f"in_{session_id}{ext}")
    output_audio_path = os.path.join(temp_dir, f"out_{session_id}.mp3")

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
    status_msg = await update.message.reply_text("⏳ Processing voice message...")

    try:
        # 1. Download audio file
        logger.info(f"[{session_id}] Downloading incoming audio ({original_filename})...")
        await file_obj.download_to_drive(custom_path=input_audio_path)

        # 2. Transcribe and interpret with Gemini
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        result = await interpret_audio_with_gemini(
            audio_path=input_audio_path,
            client=gemini_client,
            model_name=GEMINI_MODEL,
        )

        detected_lang = result.get("detected_language", "urdu_hindi")
        transcription = result.get("transcription", "")
        translation = result.get("translation", "")
        arabic_text = result.get("arabic_text", "")

        logger.info(f"[{session_id}] Language detected: {detected_lang}")

        # Check if user was already waiting to reply to a previous Arabic message
        is_replying = user_sessions.get(chat_id, {}).get("awaiting_reply", False)

        # -------------------------------------------------------------------
        # CASE A: Incoming message is in ARABIC (e.g. from WhatsApp contact)
        # -------------------------------------------------------------------
        if detected_lang == "arabic" and not is_replying:
            user_sessions[chat_id] = {
                "awaiting_reply": True,
                "last_arabic_message": transcription,
            }

            text_response = (
                "📩 **Received Message (Arabic):**\n"
                f"🇸🇦/🇦🇪 `{transcription}`\n\n"
                "🇵🇰/🇮🇳 **Translation (Urdu/Hindi):**\n"
                f"{translation}\n\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "👉 **What would you like to reply?**\n"
                "Send me a **voice note** or **type a message** in Urdu/Hindi. "
                "I will translate it into Gulf Arabic and generate a voice note for you to share back on WhatsApp!"
            )
            try:
                await update.message.reply_text(text_response, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(text_response)

        # -------------------------------------------------------------------
        # CASE B: Message is in URDU/HINDI (user replying or initiating)
        # -------------------------------------------------------------------
        else:
            # Generate Gulf Arabic voice note
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
            await text_to_speech_edge(
                text=arabic_text,
                output_path=output_audio_path,
                voice=EDGE_TTS_VOICE,
            )

            # Send voice note
            try:
                with open(output_audio_path, "rb") as voice_fh:
                    await update.message.reply_voice(
                        voice=voice_fh,
                        caption="🎙️ Gulf Arabic Voice Note (Ready to share on WhatsApp)",
                    )
            except Exception:
                with open(output_audio_path, "rb") as audio_fh:
                    await update.message.reply_audio(
                        audio=audio_fh,
                        caption="🎙️ Gulf Arabic Voice Note (Ready to share on WhatsApp)",
                    )

            # Send transcript & WhatsApp sharing tip
            text_response = (
                f"🗣️ **Your Urdu/Hindi:**\n{transcription}\n\n"
                f"🇦🇪 **Gulf Arabic Translation:**\n{arabic_text}\n\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "📲 **How to share to WhatsApp:**\n"
                "Tap & hold the voice message above ➔ Tap **Share / Forward** ➔ Select your WhatsApp chat."
            )
            try:
                await update.message.reply_text(text_response, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(text_response)

            # Clear awaiting reply state
            if chat_id in user_sessions:
                del user_sessions[chat_id]

        # Clean up "Processing..." message
        try:
            await status_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.exception(f"[{session_id}] Error processing media: {e}")
        try:
            await status_msg.edit_text(f"❌ Error processing audio:\n`{str(e)}`", parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(f"❌ Error processing audio: {str(e)}")

    finally:
        cleanup_files(input_audio_path, output_audio_path)


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles text messages. If user types their reply in Urdu/Hindi,
    translates to Gulf Arabic and sends a voice note to share to WhatsApp!
    """
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    chat_id = update.effective_chat.id

    if not gemini_client:
        await update.message.reply_text("⚠️ GEMINI_API_KEY is not configured.")
        return

    session_id = uuid.uuid4().hex[:8]
    temp_dir = tempfile.gettempdir()
    output_audio_path = os.path.join(temp_dir, f"out_text_{session_id}.mp3")

    status_msg = await update.message.reply_text("⏳ Translating your reply to Gulf Arabic...")

    try:
        # Translate text with Gemini
        result = await translate_text_with_gemini(
            text=user_text,
            client=gemini_client,
            model_name=GEMINI_MODEL,
        )

        arabic_text = result.get("arabic_text", "") or result.get("translation", "")
        urdu_text = result.get("transcription", user_text)

        if not arabic_text:
            raise ValueError("Could not obtain Arabic translation.")

        # Generate Gulf Arabic Voice Note with edge-tts
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.RECORD_VOICE)
        await text_to_speech_edge(
            text=arabic_text,
            output_path=output_audio_path,
            voice=EDGE_TTS_VOICE,
        )

        # Send voice note
        try:
            with open(output_audio_path, "rb") as voice_fh:
                await update.message.reply_voice(
                    voice=voice_fh,
                    caption="🎙️ Gulf Arabic Voice Note (Ready to share on WhatsApp)",
                )
        except Exception:
            with open(output_audio_path, "rb") as audio_fh:
                await update.message.reply_audio(
                    audio=audio_fh,
                    caption="🎙️ Gulf Arabic Voice Note (Ready to share on WhatsApp)",
                )

        # Send transcript & sharing instructions
        text_response = (
            f"🗣️ **Your Urdu/Hindi:**\n{urdu_text}\n\n"
            f"🇦🇪 **Gulf Arabic Translation:**\n{arabic_text}\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "📲 **How to share to WhatsApp:**\n"
            "Tap & hold the voice message above ➔ Tap **Share / Forward** ➔ Select your WhatsApp contact."
        )
        try:
            await update.message.reply_text(text_response, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(text_response)

        # Clear awaiting reply state
        if chat_id in user_sessions:
            del user_sessions[chat_id]

        try:
            await status_msg.delete()
        except Exception:
            pass

    except Exception as e:
        logger.exception(f"[{session_id}] Error translating text: {e}")
        try:
            await status_msg.edit_text(f"❌ Error: {str(e)}")
        except Exception:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    finally:
        cleanup_files(output_audio_path)


# ---------------------------------------------------------------------------
# Lightweight HTTP Health Check Server (For Render / Cloud Hosting)
# ---------------------------------------------------------------------------
class HealthCheckHandler(BaseHTTPRequestHandler):
    """Responds to GET and HEAD requests so Render/UptimeRobot know the service is alive."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response_body = json.dumps({"status": "ok", "bot": "running"}).encode("utf-8")
        self.wfile.write(response_body)

    def do_HEAD(self) -> None:
        # UptimeRobot sends HEAD requests by default to verify uptime
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy healthcheck logs
        pass


def start_health_check_server(port: int = 10000) -> None:
    """
    Runs a lightweight HTTP server in a background daemon thread.
    Required for Render.com Web Services to detect an open port and stay healthy.
    """
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Health check HTTP server listening on 0.0.0.0:{port}")
    except Exception as e:
        logger.warning(f"Could not start health check server on port {port}: {e}")


# ---------------------------------------------------------------------------
# Bot Application Runner
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Initializes and starts the Telegram bot using long-polling.
    """
    # Always start health check server on Render PORT (default 10000)
    try:
        web_port = int(os.getenv("PORT", "10000"))
    except ValueError:
        web_port = 10000
    start_health_check_server(web_port)

    if not TELEGRAM_BOT_TOKEN:
        logger.error(
            "TELEGRAM_BOT_TOKEN is missing! Please provide it in the .env file or environment variables."
        )
        sys.exit(1)

    logger.info("Initializing Telegram Bot with python-telegram-bot (v20+)...")
    application: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CommandHandler("reset", cancel_command))

    # Handle incoming audio (voice notes, WhatsApp forwarded audio, audio documents)
    media_filter = filters.VOICE | filters.AUDIO | filters.Document.ALL
    application.add_handler(MessageHandler(media_filter, handle_incoming_media))

    # Handle text messages (typed Urdu/Hindi replies)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    logger.info("Starting bot using standard long-polling (all audio + text active)...")
    logger.info("Bot is ready. Press Ctrl+C to stop.")

    # Run bot with long-polling
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
