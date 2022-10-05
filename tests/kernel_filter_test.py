import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.kernel_filter import KernelFilter

class TestKernelFilter(unittest.TestCase):

    def test_default_values(self):
        ftf = KernelFilter()
        self.assertEqual(ftf.method, 'gaussian_high_pass')
        self.assertEqual(ftf.value, 0.5)


    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setter(self, mock_stdout):
        ftf = KernelFilter('gaussian_low_pass', 0.2)
        self.assertEqual(ftf.method, 'gaussian_low_pass')
        self.assertEqual(ftf.value, 0.2)

        ftf.set_method('???')
        ftf.set_value(3.0)
        self.assertEqual(ftf.method, 'gaussian_high_pass')
        self.assertEqual(ftf.value, 0.5)


    def test_image_dims(self):
        ftf = KernelFilter()

        img = np.array(Image.open('tests/images/butterfly.jpg'))
        img_orig_shape = img.shape
        img = ftf.augment(img)

        self.assertGreaterEqual(img_orig_shape[0], img.shape[0])
        self.assertGreaterEqual(img_orig_shape[1], img.shape[1])
        self.assertEqual(img_orig_shape[2], img.shape[2])



### RUN
unittest.main()