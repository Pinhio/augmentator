import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.cropper import Cropper


class TestCropper(unittest.TestCase):

    def test_default_values(self):
        c = Cropper()
        self.assertEqual(c.method, 'center')
        self.assertEqual(c.value, 0.3)


    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setter(self, mock_stdout):
        c = Cropper('random', 0.25)
        self.assertEqual(c.method, 'random')
        self.assertEqual(c.value, 0.25)

        c.set_method('???')
        c.set_value(3.0)
        self.assertEqual(c.method, 'center')
        self.assertEqual(c.value, 0.3)


    def test_image_dims(self):
        c = Cropper()

        img = np.array(Image.open('images/Abra/0.jpg'))
        img_orig_shape = img.shape
        img = c.augment(img)

        self.assertNotEqual(img_orig_shape, img.shape)



### RUN
unittest.main()