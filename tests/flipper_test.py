import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.flipper import Flipper

class TestFlipper(unittest.TestCase):

    def test_default_values(self):
        f = Flipper()
        self.assertEqual(f.axis, 'y')
        

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setter(self, mock_stdout):
        f = Flipper('x')
        self.assertEqual(f.axis, 'x')

        f.set_axis('???')
        self.assertEqual(f.axis, 'y')


    def test_image_dims(self):
        f = Flipper()

        img = np.array(Image.open('images/Abra/0.jpg'))
        img_orig_shape = img.shape
        img = f.augment(img)

        self.assertEqual(img_orig_shape, img.shape)



### RUN
unittest.main()