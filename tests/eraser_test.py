import unittest
import unittest.mock
import io
import numpy as np
from PIL import Image

# add folder to path to make relative imports work
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from augmentator.eraser import Eraser

class TestEraser(unittest.TestCase):

    def test_default_values(self):
        c = Eraser()

        self.assertEqual(c.size, 0.01)
        self.assertEqual(c.count, 1)
        self.assertEqual(c.fill, 'color')
        self.assertEqual(c.color, (87,87,87))

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setters(self, mock_stdout):
        c = Eraser(size=0.01, count=5, fill='gauss', color='#FFFFFF')

        # instanciation
        self.assertEquals(c.size, 0.01)
        self.assertEquals(c.count, 5)
        self.assertEquals(c.fill, 'gauss')
        self.assertEquals(c.color, (255,255,255))

        # size
        c.set_size(0.1)
        self.assertEquals(c.size, 0.1)
        
        # count
        c.set_count(3)
        self.assertEquals(c.count, 3)
        
        # fill
        c.set_fill('color')
        self.assertEquals(c.fill, 'color')
        
        # color
        c.set_color('#000000')
        self.assertEquals(c.color, (0,0,0))
    
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_setters_invalid_inputs(self, mock_stdout):
        c = Eraser()

        # size
        c.set_size(2)
        self.assertEquals(c.size, 0.01)
        c.set_size(0)
        self.assertEquals(c.size, 0.01)
        
        # count
        c.set_count(-1)
        self.assertEquals(c.count, 40)
        c.set_count(100)
        self.assertEquals(c.count, 40)
        
        # fill
        c.set_fill('xyz')
        self.assertEquals(c.fill, 'color')
        
        # color
        c.set_color('xyz')
        self.assertEquals(c.color, (87,87,87))

    def test_image_dims(self):
        c = Eraser()

        img = np.array(Image.open('images/Abra/0.jpg'))
        should_shape = img.shape
        img_result = c.augment(img)
        self.assertEqual(should_shape, img_result.shape)

### RUN
unittest.main()