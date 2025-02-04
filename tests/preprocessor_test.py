import unittest
# ...existing code...
from ..src.lcdc.preprocessing.preprocessor import CustomProcessor, Compose

class TestCustomProcessor(unittest.TestCase):
    def test_custom_processor(self):
        def custom_fun(record):
            record['new_field'] = 42
            return record

        processor = CustomProcessor(custom_fun)
        record = {'field': 'value'}
        result = processor(record)
        self.assertEqual(result[0]['new_field'], 42)

class TestCompose(unittest.TestCase):
    def test_compose(self):
        def fun1(record):
            record['field1'] = 'value1'
            return record

        def fun2(record):
            record['field2'] = 'value2'
            return record

        processor = Compose(CustomProcessor(fun1), CustomProcessor(fun2))
        record = {'field': 'value'}
        result = processor(record)
        self.assertEqual(result[0]['field1'], 'value1')
        self.assertEqual(result[0]['field2'], 'value2')

if __name__ == '__main__':
    unittest.main()
