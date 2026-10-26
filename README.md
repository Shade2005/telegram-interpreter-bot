# 🎙️ Hindi/Urdu to Gulf Arabic Voice-to-Voice Telegram Interpreter Bot

An intelligent voice-to-voice Telegram interpreter bot built with **python-telegram-bot** (v20+ asyncio), **Google Gemini multimodal** (`google-genai`), and **Microsoft Edge TTS** (`edge-tts`).

It listens to spoken voice notes in **Urdu / Hindi**, transcribes the speech, translates it into natural **Gulf Arabic** (*Khaleeji dialect*), and replies with both an authentic Gulf Arabic audio voice note and the text translation.

---

## 🌟 Key Features

- **Long-Polling Telegram Bot**: Built on `python-telegram-bot` v20+ with native `asyncio`. No webhooks, SSL domains, or paid servers required.
- **Direct Multimodal Audio Processing**: Directly feeds incoming `.ogg` Telegram voice notes into Google Gemini without needing external transcription software.
- **Specialized Interpreter Persona**: System prompt tuned for natural, colloquial Gulf Arabic (Khaleeji dialect) translation suitable for UAE, Saudi Arabia, Kuwait, Qatar, Oman, and Bahrain.
- **Structured JSON Output**: Gemini outputs strict JSON (`transcription` and `arabic_translation`) for clean parsing and display.
- **Neural Gulf Arabic Text-to-Speech**: Uses open-source `edge-tts` with high-fidelity Gulf voices (`ar-AE-HamdanNeural` / `ar-SA-HamedNeural`).
- **Zero Disk Clutter**: Automatically deletes all temporary input `.ogg` and output `.mp3` audio files after each turn via `finally` blocks.

---

## 📁 Project Structure

```text
Tranlstation Bot/
│
├── bot.py               # Main Telegram bot application & handlers
├── test_pipeline.py     # Automated test suite (JSON parsing, TTS, Mock Gemini)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── .env                 # Your local secrets (API keys)
└── README.md            # Setup guide and documentation
```

---

## 🚀 Setup Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.10 to 3.14)
- A Telegram account
- A Google Gemini API Key

---

### 2. Obtain Your API Keys

#### A. Telegram Bot Token
1. Open Telegram and search for [`@BotFather`](https://t.me/BotFather).
2. Send `/newbot` and follow the instructions to pick a name and username.
3. Copy the HTTP API token provided by BotFather (e.g. `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).

#### B. Google Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **Get API key** and create/copy your API key.

---

### 3. Environment Configuration

Copy the example environment file:
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

Edit `.env` and fill in your keys:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: customize model or voice
GEMINI_MODEL=gemini-2.5-flash
EDGE_TTS_VOICE=ar-AE-HamdanNeural
```

#### Available Gulf Voices:
- `ar-AE-HamdanNeural` (UAE - Male, Default)
- `ar-AE-FatimaNeural` (UAE - Female)
- `ar-SA-HamedNeural` (Saudi Arabia - Male)
- `ar-SA-ZariyahNeural` (Saudi Arabia - Female)

---

### 4. Install Dependencies

Install the required packages using pip:
```bash
pip install -r requirements.txt
```

---

### 5. Run the Bot

Start the bot with standard long-polling:
```bash
python bot.py
```

You should see logs indicating the bot is active:
```text
2026-10-25 18:55:00 [INFO] VoiceInterpreterBot: Initialized Gemini client successfully with model: gemini-2.5-flash
2026-10-25 18:55:00 [INFO] VoiceInterpreterBot: Initializing Telegram Bot with python-telegram-bot (v20+)...
2026-10-25 18:55:01 [INFO] VoiceInterpreterBot: Starting bot using standard long-polling (filters.VOICE active)...
2026-10-25 18:55:01 [INFO] VoiceInterpreterBot: Bot is ready. Press Ctrl+C to stop.
```

---

## 📱 How to Use in Telegram

1. Open Telegram and search for your bot username.
2. Tap **Start** or send `/start`.
3. Press and hold the **Microphone** icon to record a voice note in **Hindi** or **Urdu** (e.g., *"آپ کیسے ہیں؟ میں دبئی میں نیا ہوں۔"* or *"नमस्ते, मुझे एयरपोर्ट जाना है"*).
4. Send the voice note.
5. The bot will:
   - Display a status update while processing.
   - Reply with an **audio voice note** speaking the fluent Gulf Arabic translation.
   - Reply with a **text message** showing:
     - 🗣️ Original Urdu/Hindi transcription
     - 🇦🇪 Spoken Gulf Arabic translation

---

## 🧪 Testing

Run the test suite to verify the pipeline components without consuming your Telegram or Gemini quota:

```bash
python -m unittest test_pipeline.py -v
```

The test suite covers:
- ✅ JSON parsing with and without Markdown code blocks.
- ✅ Resilient fallback on malformed responses.
- ✅ Automatic cleanup of temporary audio files.
- ✅ Synthesis of authentic Arabic speech via `edge-tts`.
- ✅ Gemini multimodal payload verification with mocked responses.

---

## 🛡️ Architecture & Security Details

- **No Public IP or Webhook Needed**: Uses Telegram long-polling (`getUpdates`), which runs behind firewalls and NATs without any port forwarding.
- **Privacy & Storage**: Audio files are saved with ephemeral UUIDs in the system temp directory and immediately deleted in a `finally` block once sent.
- **Direct Streaming to Gemini**: Raw `.ogg` Opus audio from Telegram is ingested directly by Gemini multimodal API, avoiding lossy intermediate conversions.
