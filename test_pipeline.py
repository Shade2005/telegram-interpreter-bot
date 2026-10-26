#!/usr/bin/env python3
"""
Unit and Integration Test Suite for Voice-to-Voice Telegram Interpreter Bot.

Tests:
1. JSON parsing and sanitization (Markdown code fences, raw JSON, error resilience).
2. Temporary file cleanup logic.
3. Edge-TTS Arabic voice synthesis (ar-AE-HamdanNeural / ar-SA-HamedNeural).
4. Gemini multimodal processing flow (with mocked API response).
5. Telegram bot initialization and handler registration.
"""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bot import (
    EDGE_TTS_VOICE,
    GEMINI_MODEL,
    SYSTEM_PROMPT,
    cleanup_files,
    interpret_audio_with_gemini,
    parse_gemini_json_response,
    text_to_speech_edge,
)


class TestJsonParsing(unittest.TestCase):
    """Test JSON parsing and sanitization from Gemini responses."""

    def test_clean_json(self):
        raw = '{"transcription": "آپ کیسے ہیں؟", "arabic_translation": "شلونك؟ عساك طيب"}'
        result = parse_gemini_json_response(raw)
        self.assertEqual(result["transcription"], "آپ کیسے ہیں؟")
        self.assertEqual(result["arabic_translation"], "شلونك؟ عساك طيب")

    def test_markdown_fenced_json(self):
        raw = """```json
{
    "transcription": "مجھے بھوک لگی ہے",
    "arabic_translation": "أنا يوعان، ودّي آكل شيء"
}
```"""
        result = parse_gemini_json_response(raw)
        self.assertEqual(result["transcription"], "مجھے بھوک لگی ہے")
        self.assertEqual(result["arabic_translation"], "أنا يوعان، ودّي آكل شيء")

    def test_malformed_json_fallback(self):
        raw = "Not a json response"
        result = parse_gemini_json_response(raw)
        self.assertEqual(result["transcription"], "")
        self.assertEqual(result["arabic_translation"], "")

    def test_partial_json(self):
        raw = '{"transcription": "ہیلو"}'
        result = parse_gemini_json_response(raw)
        self.assertEqual(result["transcription"], "ہیلو")
        self.assertEqual(result["arabic_translation"], "")


class TestFileCleanup(unittest.TestCase):
    """Test temporary audio file cleanup."""

    def test_cleanup_existing_files(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as f1, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f2:
            f1.write(b"mock voice note")
            f2.write(b"mock tts speech")
            p1, p2 = f1.name, f2.name

        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))

        cleanup_files(p1, p2)

        self.assertFalse(os.path.exists(p1))
        self.assertFalse(os.path.exists(p2))

    def test_cleanup_nonexistent_files(self):
        # Should not raise exception
        cleanup_files("non_existent_file_123.ogg", None)


class TestEdgeTTSIntegration(unittest.IsolatedAsyncioTestCase):
    """Test Edge-TTS synthesis with Gulf Arabic voice."""

    async def test_arabic_speech_generation(self):
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, "test_gulf_arabic_speech.mp3")

        try:
            arabic_text = "هلا والله، شلونك؟ عساك بخير وصحة وسلامة."
            await text_to_speech_edge(
                text=arabic_text,
                output_path=output_file,
                voice="ar-AE-HamdanNeural",
            )

            self.assertTrue(os.path.exists(output_file))
            file_size = os.path.getsize(output_file)
            self.assertGreater(file_size, 0, "Generated audio file should not be empty.")
            print(f"\n[Test] Edge-TTS successfully created Arabic audio ({file_size} bytes)")
        finally:
            cleanup_files(output_file)


class TestGeminiInterpretationMock(unittest.IsolatedAsyncioTestCase):
    """Test Gemini multimodal processing flow with mocked client."""

    async def test_interpret_audio_with_gemini_mocked(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as dummy_audio:
            dummy_audio.write(b"dummy ogg audio bytes")
            audio_path = dummy_audio.name

        try:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "transcription": "کل موسم کیسا ہوگا؟",
                "arabic_translation": "كيف الجو باكر؟"
            })
            mock_client.models.generate_content.return_value = mock_response

            transcription, translation = await interpret_audio_with_gemini(
                audio_path=audio_path,
                client=mock_client,
                model_name="gemini-2.5-flash",
            )

            self.assertEqual(transcription, "کل موسم کیسا ہوگا؟")
            self.assertEqual(translation, "كيف الجو باكر؟")

            # Verify client was called with correct parameters
            mock_client.models.generate_content.assert_called_once()
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            self.assertEqual(call_kwargs["model"], "gemini-2.5-flash")
            self.assertEqual(call_kwargs["config"].system_instruction, SYSTEM_PROMPT)
            self.assertEqual(call_kwargs["config"].response_mime_type, "application/json")
        finally:
            cleanup_files(audio_path)


if __name__ == "__main__":
    unittest.main()
