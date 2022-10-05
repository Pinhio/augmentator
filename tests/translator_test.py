import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.translator import Translator

class TestTranslator(unittest.TestCase):

    def test_default_values(self):
        t = Translator()
        self.assertEqual(t.direction, 'up')
        self.assertEqual(t.value, 0.2)
        

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setter(self, mock_stdout):
        t = Translator('down', 0.3)
        self.assertEqual(t.direction, 'down')
        self.assertEqual(t.value, 0.3)

        t.set_direction('???')
        t.set_value(3.0)
        self.assertEqual(t.direction, 'up')
        self.assertEqual(t.value, 0.2)


    def test_image_dims(self):
        t = Translator()

        img = np.array(Image.open('images/Abra/0.jpg'))
        img_orig_shape = img.shape
        img = t.augment(img)

        self.assertEqual(img_orig_shape, img.shape)



### RUN
unittest.main()