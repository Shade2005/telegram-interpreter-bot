#!/usr/bin/env python3
"""
Live End-to-End Pipeline Verification:
1. Generates a realistic Urdu voice sample using edge-tts (ur-PK-AsadNeural).
2. Sends the audio to Gemini (gemini-2.5-flash) via google-genai SDK.
3. Receives Urdu transcription & Gulf Arabic (Khaleeji) translation.
4. Synthesizes the Arabic translation using edge-tts (ar-AE-HamdanNeural).
5. Cleans up all generated audio files.
"""

import asyncio
import os
import sys
import tempfile

from dotenv import load_dotenv
import edge_tts
from google import genai

from bot import (
    GEMINI_MODEL,
    cleanup_files,
    interpret_audio_with_gemini,
    text_to_speech_edge,
)

# Reconfigure stdout for UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)


async def run_live_test() -> None:
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY found. Skipping live Gemini test.")
        return

    print("🔑 Initializing Gemini client...")
    client = genai.Client(api_key=api_key)

    temp_dir = tempfile.gettempdir()
    sample_urdu_audio = os.path.join(temp_dir, "test_input_urdu.mp3")
    sample_arabic_audio = os.path.join(temp_dir, "test_output_arabic.mp3")

    urdu_phrase = "السلام علیکم! کیا آپ مجھے دبئی ایئرپورٹ کا راستہ بتا سکتے ہیں؟"

    try:
        # Step 1: Generate sample Urdu audio
        print(f"\n1️⃣ Synthesizing test Urdu voice note: '{urdu_phrase}'...")
        comm = edge_tts.Communicate(text=urdu_phrase, voice="ur-PK-AsadNeural")
        await comm.save(sample_urdu_audio)
        print(f"   Saved to {sample_urdu_audio} ({os.path.getsize(sample_urdu_audio)} bytes)")

        # Step 2: Feed directly to Gemini multimodal
        print(f"\n2️⃣ Ingesting audio into Gemini ({GEMINI_MODEL}) for transcription & Khaleeji translation...")
        try:
            transcription, arabic_translation = await interpret_audio_with_gemini(
                audio_path=sample_urdu_audio,
                client=client,
                model_name=GEMINI_MODEL,
            )
        except Exception as api_err:
            print(f"\n⚠️ Gemini API call failed: {api_err}")
            print("Please ensure your GEMINI_API_KEY in .env is valid and has active quota.")
            return

        print("\n🎉 Gemini Result:")
        print(f"   📝 Urdu/Hindi Transcription: {transcription}")
        print(f"   🇦🇪 Gulf Arabic Translation : {arabic_translation}")

        # Step 3: Synthesize Gulf Arabic voice note
        print(f"\n3️⃣ Generating Gulf Arabic voice note with voice: ar-AE-HamdanNeural...")
        await text_to_speech_edge(
            text=arabic_translation,
            output_path=sample_arabic_audio,
            voice="ar-AE-HamdanNeural",
        )
        print(f"   Saved to {sample_arabic_audio} ({os.path.getsize(sample_arabic_audio)} bytes)")

        print("\n✅ Full Voice-to-Voice Pipeline Verification Succeeded!")

    finally:
        # Step 4: Cleanup
        print("\n4️⃣ Cleaning up temporary test audio files...")
        cleanup_files(sample_urdu_audio, sample_arabic_audio)
        print("   Done. No files left behind.")


if __name__ == "__main__":
    asyncio.run(run_live_test())
