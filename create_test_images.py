from PIL import Image
import numpy as np

img = Image.fromarray(np.full((300, 500, 3), [100, 150, 200], dtype='uint8'))
img.save('person.jpg')

img2 = Image.fromarray(np.full((400, 400, 3), [200, 100, 50], dtype='uint8'))
img2.save('garment.jpg')

print("person.jpg and garment.jpg created successfully")
