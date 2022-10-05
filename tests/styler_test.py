from asyncio.windows_events import NULL
import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image
from tensorflow import Tensor

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.styler import Styler


class TestStyler(unittest.TestCase):

    def test_default_values(self):
        s = Styler('styles', 'wave')
        self.assertNotEqual(s.model, NULL)
        self.assertEqual(s.style_folder, 'styles')
        self.assertIsInstance(s.style, Tensor)


    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setter(self, mock_stdout):
        try:
            s = Styler('???', 'wave')
        except:
            s = NULL
        self.assertNotIsInstance(s, Styler)

        try:
            s = Styler('styles', '???')
        except:
            s = NULL
        self.assertNotIsInstance(s, Styler)


    def test_image_dims(self):
        s = Styler('styles', 'wave')

        img = np.array(Image.open('tests/images/butterfly.jpg'))
        img_orig_shape = img.shape
        img = s.augment(img)

        self.assertNotEqual(img_orig_shape, img.shape)



### RUN
unittest.main()