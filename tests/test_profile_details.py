import json
import os
import tempfile
import unittest

from ai import profiles


class ProfileDetailsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = self.temp.name
        for name in ['zeta', 'alpha', 'default', 'beta']:
            with open(os.path.join(self.directory, name + '.npz'), 'wb') as stream:
                stream.write(b'model')
            with open(os.path.join(self.directory, name + '.alphabet.json'), 'w') as stream:
                json.dump(['\0', ' ', 'a', 'Z', '&'], stream)

    def test_default_is_first_even_when_a_saved_order_tries_to_move_it(self):
        self.assertEqual(profiles.ordered_profiles(self.directory), ['default', 'alpha', 'beta', 'zeta'])
        profiles.save_order(self.directory, ['zeta', 'beta', 'default', 'alpha'])
        self.assertEqual(profiles.ordered_profiles(self.directory), ['default', 'zeta', 'beta', 'alpha'])

    def test_reordering_persists_and_new_profiles_remain_visible(self):
        profiles.move_profile(self.directory, 'zeta', -1)
        self.assertEqual(profiles.ordered_profiles(self.directory), ['default', 'alpha', 'zeta', 'beta'])
        with open(os.path.join(self.directory, 'new.npz'), 'wb') as stream:
            stream.write(b'model')
        self.assertEqual(profiles.ordered_profiles(self.directory), ['default', 'alpha', 'zeta', 'beta', 'new'])
        with self.assertRaises(ValueError):
            profiles.move_profile(self.directory, 'default', 1)

    def test_details_expose_character_alphabet_without_null_or_word_vocabulary(self):
        detail = profiles.describe_profile(self.directory, 'zeta')
        self.assertEqual(detail['alphabet'], [' ', 'a', 'Z', '&'])
        self.assertEqual(detail['alphabet_size'], 4)
        self.assertNotIn('Z', detail['missing_uppercase'])
        self.assertIn('Q', detail['missing_uppercase'])
        self.assertEqual(detail['style_samples'], 0)

    def test_invalid_alphabet_is_reported_on_its_profile(self):
        with open(os.path.join(self.directory, 'alpha.alphabet.json'), 'w') as stream:
            json.dump(['a', 'a'], stream)
        detail = profiles.describe_profile(self.directory, 'alpha')
        self.assertTrue(detail['error'])
        self.assertEqual(profiles.ordered_profiles(self.directory)[0], 'default')

    def test_corrupt_order_recovers_and_rejects_path_traversal(self):
        with open(os.path.join(self.directory, '.order.json'), 'w') as stream:
            stream.write('{broken')
        self.assertEqual(profiles.ordered_profiles(self.directory), ['default', 'alpha', 'beta', 'zeta'])
        with self.assertRaises(ValueError):
            profiles.move_profile(self.directory, '../escape', -1)


if __name__ == '__main__':
    unittest.main()
