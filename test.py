from PIL import Image

im1 = Image.open(r"./datasets/playroom_split/color/input/DSC05573.jpg")
im2 = Image.open(r"./datasets/playroom_split/structure/input/DSC05573.jpg")
im3 = Image.blend(im1, im2, 0.3)
im3.show()
im3.save('test.jpg')
