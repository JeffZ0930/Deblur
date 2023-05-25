#Python OpenCV script to test out JPEG compression rates

import cv2
import numpy as np

def save(path, image, jpg_quality=None): #, png_compression=None
  '''
  persist :image: object to disk. if path is given, load() first.
  jpg_quality: for jpeg only. 0 - 100 (higher means better). Default is 95.
  png_compression: For png only. 0 - 9 (higher means a smaller size and longer compression time).
                  Default is 3.
  '''

  if jpg_quality:

    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
#   elif png_compression:
#     cv2.imwrite(path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
  else:
    cv2.imwrite(path, image)
    

def resize(path, image, dim):
    # resize imagek
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(path, resized)

def main():
    train_img = cv2.imread("my_test/original_clear.jpg")
    test_img = cv2.imread("my_test/original_blur.jpeg")
    train_dim = (train_img.shape[1], train_img.shape[0])
    test_dim = (test_img.shape[1], test_img.shape[0])

    outpath_jpg = "my_test/converted_blur.jpg"

    resize(outpath_jpg, train_img, test_dim)

    img = cv2.imread(outpath_jpg)
    #display original image
    cv2.imshow('My JPG Conversion', img)
    cv2.waitKey(0)

    # Gaussian = cv2.GaussianBlur(image, (13, 13), 6)
    # cv2.imshow('Gaussian Blurring', Gaussian)

    save(outpath_jpg, img, jpg_quality = 1)


if __name__ == "__main__":
    main()