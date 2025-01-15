import unittest
from unittest.mock import MagicMock
import numpy as np

from lcdc.preprocessing import Compose

class TestCompose(unittest.TestCase):

    def test_call(self):
        record = MagicMock()
        preprocessor1 = MagicMock()
        preprocessor2 = MagicMock()
        preprocessor1.return_value = [record]
        preprocessor2.return_value = [record]
        compose = Compose(preprocessor1, preprocessor2)
        result = compose(record)
        self.assertEqual(result, [record])

if __name__ == '__main__':
    unittest.main()
