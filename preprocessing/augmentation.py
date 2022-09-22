import cv2
import os

green_arr = list()
for i in range(125):
    img = cv2.imread('dataset/train/mask/{}.png'.format(i))
    green = img[:,:,1]
    if(green[green==255].sum()!=0):
        green_arr.append(i)

red_arr = set(range(125)) - set(green_arr)

image_types = ('mask', 'ndvi', 'nir', 'red')

for image_type in image_types:
    name = 0
    for green in green_arr:
        for red in red_arr:
            foreground = cv2.imread("dataset/train/{}/{}.png".format(image_type, green))
            background = cv2.imread("dataset/train/{}/{}.png".format(image_type, red))
            alpha = cv2.imread("dataset/train/augmentation/{}.png".format(green))

            foreground = cv2.resize(foreground, (1469, 1008))
            background = cv2.resize(background, (1469, 1008))
            alpha = cv2.resize(alpha, (1469, 1008))

            foreground = foreground.astype(float)
            background = background.astype(float)

            alpha = alpha.astype(float)/255

            foreground = cv2.multiply(alpha, foreground)

            background = cv2.multiply(1.0 - alpha, background)

            outImage = cv2.add(foreground, background)

            if not os.path.exists('dataset_augmentation'):
                os.makedirs('dataset_augmentation')
                os.makedirs('dataset_augmentation/red')
                os.makedirs('dataset_augmentation/ndvi')
                os.makedirs('dataset_augmentation/nir')
                os.makedirs('dataset_augmentation/mask')
            cv2.imwrite('dataset_augmentation/{}/{}.png'.format(image_type, name), outImage)
            name += 1

            # outImage = cv2.resize(outImage, (1600, 1000))
            # cv2.imshow("outImg", outImage/255)
            # cv2.waitKey(0)
