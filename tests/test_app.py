"""Exercise the deployed model and API, including startup outside the repo directory."""
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_directory = Path.cwd()
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.addClassCleanup(os.chdir, cls.previous_directory)
        os.chdir(cls.directory.name)
        import main

        cls.main = main
        cls.client = cls.enterClassContext(TestClient(main.app))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])

    def test_web_assets(self):
        for path, content_type in (
            ("/", "text/html"),
            ("/static/script.js", "javascript"),
            ("/static/style.css", "text/css"),
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response.headers["content-type"])
                self.assertTrue(response.content)

    def test_saved_model_regressions(self):
        for text, expected in (
            ("I love you so much", "love"),
            ("I am so grateful to have such caring people around me", "love"),
            ("I feel so alone and hopeless today.", "sadness"),
            ("I am furious that they cancelled the trip at the last minute.", "anger"),
            ("I feel terrified when walking down dark alleyways alone.", "fear"),
        ):
            with self.subTest(text=text):
                response = self.client.post("/predict", json={"text": text})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["predicted_emotion"], expected)
                self.assertEqual(result["text"], text)
                probabilities = result["all_probabilities"]
                self.assertEqual(set(probabilities), set(self.main.emotion_labels))
                self.assertEqual(probabilities, result["all_probabilites"])
                self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()))
                self.assertAlmostEqual(sum(probabilities.values()), 1, places=5)
                self.assertEqual(result["confidence"], max(probabilities.values()))

    def test_unusable_text(self):
        for text in ("     ", "!!! ???", "asdkjh qwlekj zxcmnb", "मैं बहुत खुश हूँ"):
            with self.subTest(text=text):
                response = self.client.post("/predict", json={"text": text})
                self.assertEqual(response.status_code, 422)
                self.assertIsInstance(response.json()["detail"], str)
                self.assertNotIn("predicted_emotion", response.json())

    def test_request_validation(self):
        for payload in ({}, {"text": ""}, {"text": None}, {"text": 123}, {"text": "a" * 2001}):
            with self.subTest(payload=str(payload)[:80]):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_known_word_beyond_truncation_does_not_validate_gibberish(self):
        text = "zxqv " * self.main.max_sequence_length + "love"
        response = self.client.post("/predict", json={"text": text})
        self.assertEqual(response.status_code, 422)

    def test_unavailable_model(self):
        tokenizer = self.main.dl_model.pop("Tokenizer")
        try:
            self.assertEqual(self.client.get("/health").status_code, 503)
            self.assertEqual(
                self.client.post("/predict", json={"text": "I love you so much"}).status_code,
                503,
            )
        finally:
            self.main.dl_model["Tokenizer"] = tokenizer


if __name__ == "__main__":
    unittest.main()
