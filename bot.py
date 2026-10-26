#!/usr/bin/env python3
"""
Hindi/Urdu to Gulf Arabic Voice-to-Voice Telegram Interpreter Bot.

Flow:
1. Listens for incoming Telegram voice messages (filters.VOICE).
2. Downloads the incoming voice note (.ogg) locally.
3. Transcribes & translates using Google Gemini (gemini-2.5-flash via google-genai SDK).
4. Converts the Gulf Arabic translation into speech using edge-tts (ar-AE-HamdanNeural / ar-SA-HamedNeural).
5. Replies back with:
   - Voice note (Gulf Arabic spoken translation)
   - Text message (Urdu/Hindi transcription + Gulf Arabic translation)
6. Cleans up all temporary audio files after each message is sent.
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
from typing import Dict, Optional, Tuple

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

# Load environment variables from .env file (with override=True to prioritize .env)
load_dotenv(override=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "ar-AE-HamdanNeural").strip()

# Keep GOOGLE_API_KEY in sync with GEMINI_API_KEY if explicitly specified
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

# System instruction as requested
SYSTEM_PROMPT = (
    "You are an expert bilingual interpreter between Hindi/Urdu and Gulf Arabic (Khaleeji dialect). "
    "Transcribe what is spoken in Urdu/Hindi, and provide the natural spoken Gulf Arabic translation. "
    "Output strictly in JSON format with two keys: 'transcription' and 'arabic_translation'."
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
            return {
                "transcription": str(data.get("transcription", "")).strip(),
                "arabic_translation": str(data.get("arabic_translation", "")).strip(),
            }
    except json.JSONDecodeError as err:
        logger.error(f"Failed to decode JSON from Gemini output: {cleaned} - Error: {err}")

    return {"transcription": "", "arabic_translation": ""}


async def interpret_audio_with_gemini(
    audio_path: str,
    client: genai.Client,
    model_name: str = GEMINI_MODEL,
) -> Tuple[str, str]:
    """
    Feeds the raw audio file directly to Gemini to obtain:
    1. Hindi/Urdu transcription
    2. Gulf Arabic spoken translation
    """
    logger.info(f"Reading audio file for Gemini: {audio_path}")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Determine audio MIME type based on file extension (Telegram voice is .ogg)
    ext = os.path.splitext(audio_path)[1].lower()
    mime_type_map = {
        ".ogg": "audio/ogg",
        ".oga": "audio/ogg",
        ".mp3": "audio/mp3",
        ".wav": "audio/wav",
        ".m4a": "audio/m4a",
    }
    mime_type = mime_type_map.get(ext, "audio/ogg")

    # Pass audio bytes directly as an audio Part
    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

    # List of models to try (primary model followed by robust fallbacks)
    models_to_try = [model_name]
    for fallback in ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]:
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
                    "Transcribe this voice message and provide the Gulf Arabic translation.",
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
        raise ValueError(f"Gemini generation failed: {last_err}")

    logger.debug(f"Raw Gemini response: {response.text}")
    parsed = parse_gemini_json_response(response.text)
    transcription = parsed.get("transcription", "")
    arabic_translation = parsed.get("arabic_translation", "")

    if not transcription and not arabic_translation:
        raise ValueError(f"Could not extract transcription or translation from response: {response.text}")

    return transcription, arabic_translation


async def text_to_speech_edge(
    text: str,
    output_path: str,
    voice: str = EDGE_TTS_VOICE,
) -> str:
    """
    Converts Arabic text to speech using edge-tts with a Gulf Arabic voice.
    Saves the output to output_path.
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
# Telegram Bot Command & Message Handlers
# ---------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles /start command.
    """
    user_name = update.effective_user.first_name if update.effective_user else "there"
    welcome_message = (
        f"👋 Hello {user_name}!\n\n"
        "🎙️ **Urdu/Hindi ➔ Gulf Arabic Voice Interpreter Bot**\n\n"
        "Simply send me a **voice message** in Hindi or Urdu.\n\n"
        "I will:\n"
        "1. 📝 Transcribe your Hindi/Urdu speech\n"
        "2. 🇦🇪 Translate it into natural spoken Gulf Arabic (Khaleeji dialect)\n"
        "3. 🗣️ Reply with a Gulf Arabic voice note & the text translation\n\n"
        "Press and hold the microphone icon to send your first voice note!"
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles /help command.
    """
    help_message = (
        "💡 **How to use this bot:**\n\n"
        "• Record and send a voice note in Hindi or Urdu.\n"
        "• The bot uses Google Gemini to interpret and translate into spoken Khaleeji Arabic.\n"
        "• The bot replies with an authentic Gulf Arabic audio note and transcript.\n\n"
        f"⚙️ **Active Voice:** `{EDGE_TTS_VOICE}`\n"
        f"🤖 **Model:** `{GEMINI_MODEL}`"
    )
    await update.message.reply_text(help_message, parse_mode="Markdown")


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles incoming voice notes (filters.VOICE).
    Downloads audio, passes to Gemini, generates TTS, and replies back.
    Ensures all temporary files are cleaned up.
    """
    if not update.message or not update.message.voice:
        return

    # Check if Gemini client is ready
    if not gemini_client:
        await update.message.reply_text(
            "⚠️ Bot configuration error: `GEMINI_API_KEY` is not set.\n"
            "Please configure the `.env` file with a valid Gemini API key and restart.",
            parse_mode="Markdown",
        )
        return

    session_id = uuid.uuid4().hex[:8]
    temp_dir = tempfile.gettempdir()
    input_audio_path = os.path.join(temp_dir, f"in_{session_id}.ogg")
    output_audio_path = os.path.join(temp_dir, f"out_{session_id}.mp3")

    # Send typing/record_voice status action
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.RECORD_VOICE,
    )
    status_msg = await update.message.reply_text("⏳ Processing your voice note...")

    try:
        # 1. Download incoming Telegram voice note (.ogg format) locally
        logger.info(f"[{session_id}] Downloading incoming voice note...")
        voice = update.message.voice
        voice_file = await voice.get_file()
        await voice_file.download_to_drive(custom_path=input_audio_path)
        logger.info(f"[{session_id}] Voice note downloaded to: {input_audio_path}")

        # 2. Transcribe & translate using Gemini multimodal
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )
        transcription, arabic_translation = await interpret_audio_with_gemini(
            audio_path=input_audio_path,
            client=gemini_client,
            model_name=GEMINI_MODEL,
        )
        logger.info(f"[{session_id}] Transcription: {transcription}")
        logger.info(f"[{session_id}] Arabic Translation: {arabic_translation}")

        # 3. Convert Arabic translation text to speech using edge-tts
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.RECORD_VOICE,
        )
        await text_to_speech_edge(
            text=arabic_translation,
            output_path=output_audio_path,
            voice=EDGE_TTS_VOICE,
        )

        # 4. Reply with the generated Arabic voice note
        logger.info(f"[{session_id}] Sending Arabic voice note response...")
        try:
            with open(output_audio_path, "rb") as voice_fh:
                await update.message.reply_voice(
                    voice=voice_fh,
                    caption="🎙️ Gulf Arabic Voice Note",
                )
        except Exception as voice_err:
            logger.warning(f"[{session_id}] reply_voice failed ({voice_err}), trying reply_audio fallback...")
            with open(output_audio_path, "rb") as audio_fh:
                await update.message.reply_audio(
                    audio=audio_fh,
                    caption="🎙️ Gulf Arabic Audio",
                )

        # 5. Reply with text containing both Urdu transcription & Arabic translation
        text_reply = (
            f"🗣️ **Original (Urdu/Hindi):**\n"
            f"{transcription}\n\n"
            f"🇦🇪 **Translation (Gulf Arabic):**\n"
            f"{arabic_translation}"
        )
        try:
            await update.message.reply_text(text_reply, parse_mode="Markdown")
        except Exception:
            # Fallback to plain text if Markdown parsing fails due to special chars
            await update.message.reply_text(text_reply)

        # Remove the temporary "Processing..." message
        try:
            await status_msg.delete()
        except Exception:
            pass

        logger.info(f"[{session_id}] Successfully processed and replied to voice message.")

    except Exception as e:
        logger.exception(f"[{session_id}] Error processing voice message: {e}")
        try:
            await status_msg.edit_text(
                f"❌ Sorry, an error occurred while processing your voice note:\n`{str(e)}`",
                parse_mode="Markdown",
            )
        except Exception:
            await update.message.reply_text(
                f"❌ Sorry, an error occurred while processing your voice note:\n{str(e)}"
            )

    finally:
        # 6. Clean up temporary audio files after each message is sent
        cleanup_files(input_audio_path, output_audio_path)
        logger.info(f"[{session_id}] Cleaned up temporary files.")


# ---------------------------------------------------------------------------
# Lightweight HTTP Health Check Server (For Render / Cloud Hosting)
# ---------------------------------------------------------------------------
class HealthCheckHandler(BaseHTTPRequestHandler):
    """Responds to GET requests so Render/UptimeRobot know the service is alive."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response_body = json.dumps({"status": "ok", "bot": "running"}).encode("utf-8")
        self.wfile.write(response_body)

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy healthcheck logs from Render/UptimeRobot
        pass


def start_health_check_server(port: int = 8080) -> None:
    """
    Runs a lightweight HTTP server in a background daemon thread.
    Required for Render.com Web Services to detect an open port,
    and enables UptimeRobot to keep the free service awake 24/7.
    """
    try:
        server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Health check HTTP server listening on port {port}")
    except Exception as e:
        logger.warning(f"Could not start health check server on port {port}: {e}")


# ---------------------------------------------------------------------------
# Bot Application Runner
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Initializes and starts the Telegram bot using long-polling.
    """
    # Start health check server if PORT is set (Render, Heroku, Cloud Run, etc.)
    port_str = os.getenv("PORT", "").strip()
    if port_str:
        try:
            start_health_check_server(int(port_str))
        except ValueError:
            start_health_check_server(8080)

    if not TELEGRAM_BOT_TOKEN:
        logger.error(
            "TELEGRAM_BOT_TOKEN is missing! Please provide it in the .env file or environment variables."
        )
        sys.exit(1)

    logger.info("Initializing Telegram Bot with python-telegram-bot (v20+)...")
    application: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    logger.info("Starting bot using standard long-polling (filters.VOICE active)...")
    logger.info("Bot is ready. Press Ctrl+C to stop.")

    # Run bot with long-polling
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
