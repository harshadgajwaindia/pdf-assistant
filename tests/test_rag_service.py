import unittest

from backend.rag_service import _extract_text_content, _normalize_pdf_text


class ExtractTextContentTests(unittest.TestCase):
    def test_extracts_answer_text_from_gemini_content_blocks(self) -> None:
        content = [
            {
                "type": "text",
                "text": "The harvest begins in September.",
                "extras": {"signature": "provider metadata"},
            }
        ]

        self.assertEqual(
            _extract_text_content(content),
            "The harvest begins in September.",
        )


class NormalizePdfTextTests(unittest.TestCase):
    def test_joins_extracted_words_split_across_lines(self) -> None:
        extracted = "entity.\n\nIt\n\nis\n\nan\n\ninstance\n\nof"

        self.assertEqual(
            _normalize_pdf_text(extracted),
            "entity. It is an instance of",
        )


if __name__ == "__main__":
    unittest.main()
