import unittest
from unittest.mock import MagicMock
import numpy as np

from lcdc.preprocessing import Compose

class TestCompose(unittest.TestCase):

    def test_call(self):
        track = MagicMock()
        preprocessor1 = MagicMock()
        preprocessor2 = MagicMock()
        preprocessor1.return_value = [track]
        preprocessor2.return_value = [track]
        compose = Compose(preprocessor1, preprocessor2)
        result = compose(track, MagicMock())
        self.assertEqual(result, [track])

if __name__ == '__main__':
    unittest.main()