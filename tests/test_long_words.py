import unittest
import numpy as np

from ai.inference.hw_model import GenerationError
from document.calibration import PageTemplate
from document.layout import generate_document


class RecordingModel:
    alphabet = list('\0 abcdefghijklmnopqrstuvwxyz-')

    def __init__(self, reject_above=None):
        self.calls = []
        self.accepted = []
        self.reject_above = reject_above

    def sample(self, text, **options):
        self.calls.append(text)
        if self.reject_above is not None and len(text) > self.reject_above:
            raise GenerationError('Rejected draw')
        self.accepted.append(text)
        return np.array([[0, 0, 0], [len(text) * 7, 1, 0], [0, -1, 1]], np.float32)


class LongWordTests(unittest.TestCase):
    def test_reported_hyphenated_word_is_generated_without_losing_characters(self):
        model = RecordingModel()
        doc = generate_document('farmingcattle-herding', model, PageTemplate(), chunk_chars=12)
        self.assertEqual(''.join(model.accepted), 'farmingcattle-herding')
        self.assertTrue(all(len(text) <= 12 for text in model.calls))
        self.assertEqual(len(doc.pages), 1)

    def test_long_unhyphenated_word_has_a_smaller_join_than_a_word_gap(self):
        long = RecordingModel()
        words = RecordingModel()
        long_doc = generate_document('abcdefghijklmnopqrstuvwx', long, PageTemplate(), chunk_chars=12)
        words_doc = generate_document('abcdefghijkl mnopqrstuvwx', words, PageTemplate(), chunk_chars=12)
        def gap(doc):
            strokes = doc.pages[0].lines[0].strokes
            return strokes[1].points_mm[0, 0] - strokes[0].points_mm[-1, 0]
        self.assertEqual(''.join(long.accepted), 'abcdefghijklmnopqrstuvwx')
        self.assertLess(gap(long_doc), gap(words_doc) / 2)

    def test_failed_single_word_subdivides_only_that_word(self):
        model = RecordingModel(reject_above=6)
        generate_document('hello farmingcattle-herding', model, PageTemplate(), chunk_chars=20)
        self.assertEqual(''.join(model.accepted), 'hellofarmingcattle-herding')
        self.assertEqual(model.accepted.count('hello'), 1)
        self.assertTrue(all(len(text) <= 6 for text in model.accepted))

    def test_failed_character_still_raises_with_a_bounded_retry_count(self):
        model = RecordingModel(reject_above=0)
        with self.assertRaises(GenerationError):
            generate_document('abcdefghijklmnopqrstuvwx', model, PageTemplate(), chunk_chars=12)
        self.assertLess(len(model.calls), 10)


if __name__ == '__main__':
    unittest.main()
