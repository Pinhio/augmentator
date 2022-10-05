import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.mixer import Mixer

class TestMixer(unittest.TestCase):

    TEST_IMG_DIR = 'tests/assets/mixer_test/'

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_default_without_mix_img(self, mock_stdout):
        # must raise error
        with self.assertRaises(ValueError):
            c = Mixer()

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_default_values(self, mock_stdout, test_img_dir=TEST_IMG_DIR):
        # define test image and create instance
        img = np.array(Image.open(f'{test_img_dir}0.jpg'))
        c = Mixer(mix_img=img)

        self.assertEquals(c.method, 'avg')
        self.assertEquals(c.value, 0.5)
        self.assertIsInstance(c.mix_img, np.ndarray)
        self.assertEquals(c.mix_img.shape, img.shape)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setters(self, mock_stdout, test_img_dir=TEST_IMG_DIR):
        # define test images and create instance
        img_1 = np.array(Image.open(f'{test_img_dir}0.jpg'))
        img_2 = np.array(Image.open(f'{test_img_dir}1.jpg'))
        c = Mixer(method='rel', value=0.1, mix_img=img_1)

        # instantiation
        self.assertEquals(c.method, 'rel')
        self.assertEquals(c.value, 0.1)
        self.assertIsInstance(c.mix_img, np.ndarray)
        self.assertEquals(c.mix_img.shape, img_1.shape)

        # method
        c.set_method('avg')
        self.assertEquals(c.method, 'avg')

        # value
        c.set_value(0.2)
        self.assertEquals(c.value, 0.5)

        # mix_img
        c.set_mix_img(img_2)
        self.assertIsInstance(c.mix_img, np.ndarray)
        self.assertEquals(c.mix_img.shape, img_2.shape)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setters_invalid_inputs(self, mock_stdout, test_img_dir=TEST_IMG_DIR):
        img = np.array(Image.open(f'{test_img_dir}0.jpg'))
        c = Mixer(method='xxx', value=3, mix_img=img)

        # instantiation
        self.assertEquals(c.method, 'avg')
        self.assertEquals(c.value, 0.5)

        # method
        c.set_method('qwertz')
        self.assertEquals(c.method, 'avg')

        # value
        c.set_value(-1)
        self.assertEquals(c.value, 0.5)

        # mix_img
        with self.assertRaises(ValueError):
            c.set_mix_img('invalid type')

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_image_dims_no_reshape(self, mock_stdout, test_img_dir=TEST_IMG_DIR):
        img_1 = np.array(Image.open(f'{test_img_dir}0.jpg'))
        img_2 = np.array(Image.open(f'{test_img_dir}0.jpg'))
        c = Mixer(method='avg', value=0.5, mix_img=img_1)

        should_shape = img_2.shape
        img_result = c.augment(img_2)
        self.assertEqual(should_shape, img_result.shape)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_image_dims_with_reshape(self, mock_stdout, test_img_dir=TEST_IMG_DIR):
        img_1 = np.array(Image.open(f'{test_img_dir}0.jpg'))
        img_2 = np.array(Image.open(f'{test_img_dir}1.jpg'))
        c = Mixer(method='avg', value=0.5, mix_img=img_1)

        should_shape = (max(img_1.shape[0], img_2.shape[0]),
                        max(img_1.shape[1], img_2.shape[1]),
                        3)
        img_result = c.augment(img_2)
        self.assertEqual(should_shape, img_result.shape)

### RUN
unittest.main()