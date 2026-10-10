import unittest
from unittest.mock import patch
import numpy as np
from ai.inference.hw_model import HandwritingModel, _split_missed_pen_lifts
from document.calibration import PageTemplate, PrinterProfile
from document.layout import generate_document, wrap_text
from document.gcode import generate_gcode, GCodeValidationError

class GenerationTests(unittest.TestCase):
    def test_lined_text_fits_between_rulings(self):
        class Model:
            alphabet = ['\0', 'a']
            def sample(self, text, **options):
                return np.array([[0, 0, 0], [10, 50, 0], [10, -50, 1]], np.float32)
        page = PageTemplate(height_mm=58, margin_top_mm=10, margin_bottom_mm=10, line_spacing_mm=9)
        document = generate_document('\n'.join(['a'] * 5), Model(), page)
        self.assertEqual([len(p.lines) for p in document.pages], [4, 1])
        for p in document.pages:
            for index, line in enumerate(p.lines):
                ys = np.concatenate([s.points_mm[:, 1] for s in line.strokes])
                top = page.margin_top_mm + index * page.line_spacing_mm
                self.assertGreater(ys.min(), top)
                self.assertLess(ys.max(), top + page.line_spacing_mm)
                self.assertAlmostEqual((ys.min() + ys.max()) / 2, top + page.line_spacing_mm / 2)

    def test_physical_lines_use_full_page_width_without_a_character_cap(self):
        text = 'word ' * 100
        normal = wrap_text(text, PageTemplate())
        wide = wrap_text(text, PageTemplate(width_mm=300))
        self.assertGreater(len(normal[0]['text']), 72)
        self.assertGreater(len(wide[0]['text']), len(normal[0]['text']))
        self.assertLess(len(wrap_text(text, PageTemplate(), letter_spacing=1.5)[0]['text']), len(normal[0]['text']))

    def test_rejected_generation_never_returns_scribbles(self):
        model = HandwritingModel.__new__(HandwritingModel)
        model.alphabet = ['\0', 'a']
        model.char_to_idx = {'\0': 0, 'a': 1}
        with patch.object(model, '_sample_once', return_value=(np.array([[1, 0, 0]], np.float32), False)) as draw:
            with self.assertRaisesRegex(ValueError, 'Could not finish'):
                model.sample('a', attempts=2)
            self.assertEqual(draw.call_count, 2)

    def test_pen_lift_precedes_large_jump(self):
        points = np.array([[1,0,0], [1,0,0], [20,0,0], [1,0,1]], np.float32)
        split = _split_missed_pen_lifts(points)
        self.assertEqual(split[1,2], 1)
        self.assertEqual(split[2,2], 0)

    def test_long_documents_use_bounded_chunks_and_progress(self):
        class Model:
            alphabet = list('\0 abcdefghijklmnopqrstuvwxyz')
            texts = []
            def sample(self, text, **options):
                self.texts.append(text)
                return np.array([[0,0,0],[len(text)*8,1,0],[1,-1,1]], np.float32)
        model = Model()
        page = PageTemplate(height_mm=65, margin_top_mm=10, margin_bottom_mm=10, line_spacing_mm=9)
        progress = []
        text = ('hello world this is a longer phrase for bounded generation\n') * 9
        doc = generate_document(text, model, page, chunk_chars=20, progress_cb=lambda a,b:progress.append((a,b)))
        self.assertGreater(len(doc.pages), 1)
        self.assertTrue(all(len(chunk) <= 20 for chunk in model.texts))
        self.assertEqual(progress[-1][0], progress[-1][1])
        for page in doc.pages:
            for line in page.lines:
                for stroke in line.strokes:
                    self.assertTrue(np.isfinite(stroke.points_mm).all())
                    self.assertGreaterEqual(stroke.points_mm[:,0].min(), 20)
                    self.assertLessEqual(stroke.points_mm[:,0].max(), 195.001)

    def test_failed_chunk_splits_into_words_without_dropping_text(self):
        from ai.inference.hw_model import GenerationError
        class Model:
            alphabet = list('\0 hello world')
            calls = []
            def sample(self, text, **options):
                self.calls.append(text)
                if ' ' in text:
                    raise GenerationError('Could not finish')
                return np.array([[0,0,0],[len(text)*8,1,0],[1,-1,1]], np.float32)
        model = Model()
        document = generate_document('hello world', model, PageTemplate())
        self.assertEqual(model.calls, ['hello world', 'hello', 'world'])
        self.assertEqual(len(document.pages[0].lines[0].strokes), 2)

    def test_off_printer_geometry_blocks_gcode(self):
        from document.layout import Document, Page, LaidOutLine, Stroke
        doc=Document([Page(210,297,[LaidOutLine([Stroke(np.array([[20,25],[30,280]]))],25)])])
        with self.assertRaises(GCodeValidationError):
            generate_gcode(doc, PageTemplate(), PrinterProfile())

    def test_single_point_dots_survive_layout_and_gcode(self):
        class Model:
            alphabet = ['\0', 'i']
            def sample(self, text, **options):
                return np.array([[0,0,0],[3,0,1],[0,3,1],[2,-3,0],[2,0,1]], np.float32)
        page = PageTemplate()
        doc = generate_document('i', Model(), page)
        strokes = doc.pages[0].lines[0].strokes
        self.assertEqual(len(strokes), 3)
        np.testing.assert_array_equal(strokes[1].points_mm[0], strokes[1].points_mm[1])
        self.assertEqual(generate_gcode(doc, page, PrinterProfile()).count('; pen down'), 3)

if __name__ == '__main__': unittest.main()
