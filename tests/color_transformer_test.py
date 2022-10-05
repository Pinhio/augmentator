import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.color_transformer import Color_Transformer


class TestColorTransformer(unittest.TestCase):

    def test_default_values(self):
        c = Color_Transformer()
        self.assertEqual(c.method, 'keep')
        self.assertEqual(c.value, 10)
        self.assertEqual(c.channel, 'rgb')
        self.assertEqual(c.is_value_percentage, False)


    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setters(self, mock_stdout):
        c = Color_Transformer(method='kill', value=0.25, channel='g', is_value_percentage=True)
        # during instantiation
        self.assertEqual(c.method, 'kill')
        self.assertEqual(c.value, 0.25)
        self.assertEqual(c.channel, 'g')
        self.assertTrue(c.is_value_percentage)

        # error cases that lead to default values
        c.set_method('???')
        self.assertEqual(c.method, 'keep')
        
        c.set_value(-3)
        self.assertEqual(c.value, 10)

        # channel chars
        c.set_channel('xyz')
        self.assertEqual(c.channel, 'rgb')

        # multiple chars
        c.set_channel('rr')
        self.assertEqual(c.channel, 'rgb')

        # channel length
        c.set_channel('rgbr')
        self.assertEqual(c.channel, 'rgb')


    def test_get_channel_encodings(self):
        c = Color_Transformer()

        self.assertEqual(c._get_channel_encoding(), [0,1,2])
        c.set_channel('r')
        self.assertEqual(c._get_channel_encoding(), [0])
        c.set_channel('gb')
        self.assertEqual(c._get_channel_encoding(), [1,2])


    def test_image_dims(self):
        c = Color_Transformer(method='kill', channel='gb')

        img = np.array(Image.open('images/Abra/0.jpg'))
        should_shape = img.shape
        img = c.augment(img)

        self.assertEqual(should_shape, img.shape)


    def test_rgb_vals_kill(self):
        c = Color_Transformer(method='kill', channel='gb')

        img = np.array(Image.open('images/Abra/0.jpg'))
        img = c.augment(img)

        self.assertEqual(img[0][0][0], 255)
        self.assertEqual(img[0][0][1], 0)
        self.assertEqual(img[0][0][2], 0)


    def test_rgb_vals_keep(self):
        c = Color_Transformer(method='keep', channel='gb')

        img = np.array(Image.open('images/Abra/0.jpg'))
        img = c.augment(img)
        
        self.assertEqual(img[0][0][0], 0)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)


    def test_rgb_vals_min_max(self):
        c = Color_Transformer(method='max', channel='r', value=200)

        # max
        img = np.array(Image.open('images/Abra/0.jpg'))
        img = c.augment(img)
        
        self.assertEqual(img[0][0][0], 200)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)

        # min
        c.set_method('min')
        c.set_value(240)
        img = c.augment(img)

        self.assertEqual(img[0][0][0], 240)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)


    def test_rgb_vals_dec_inc_abs(self):
        c = Color_Transformer(method='dec', value=15, channel='rgb')

        # decrease
        img = np.array(Image.open('images/Abra/0.jpg'))
        img = c.augment(img)
        
        self.assertEqual(img[0][0][0], 240)
        self.assertEqual(img[0][0][1], 240)
        self.assertEqual(img[0][0][2], 240)

        # increase again
        c.set_method('inc')
        img = c.augment(img)

        self.assertEqual(img[0][0][0], 255)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)


    def test_rgb_vals_dec_percentage(self):
        c = Color_Transformer(method='dec', value=155, channel='r')

        # create baseline with r-value at 100
        img = np.array(Image.open('images/Abra/0.jpg'))
        img = c.augment(img)
        
        self.assertEqual(img[0][0][0], 100)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)

        # decrease 10%
        c.set_value(10)
        c.set_is_value_percentage(True)
        img = c.augment(img)

        self.assertEqual(img[0][0][0], 90)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)


    def test_rgb_vals_inc_percentage(self):
        c = Color_Transformer(method='dec', value=155, channel='r')

        # create baseline with r-value at 100
        img = np.array(Image.open('images/Abra/0.jpg'))
        img = c.augment(img)
        
        self.assertEqual(img[0][0][0], 100)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)

        # decrease 10%
        c.set_method('inc')
        c.set_value(10)
        c.set_is_value_percentage(True)
        img = c.augment(img)

        self.assertEqual(img[0][0][0], 110)
        self.assertEqual(img[0][0][1], 255)
        self.assertEqual(img[0][0][2], 255)



### RUN
unittest.main()