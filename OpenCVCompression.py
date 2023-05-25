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
    
    imgpath = "Example3/my-convert-001.jpg"
    img = cv2.imread(imgpath)

    #display the image
    cv2.imshow('My JPG Conversion', img)

    # save the image in JPEG format with 85% quality
    outpath_jpeg = "Example3/OpenCVCompression.jpg"

    save(outpath_jpeg,img,jpg_quality=1)

    compImg_path = "Example3/OpenCVCompression.jpg"
    compImg = cv2.imread(compImg_path)

    compOutPath_jpeg = "Example3/OpenCVResized.jpg"

    width = 640
    height = 829
    dim = (width, height)
 
    resize(compOutPath_jpeg, compImg, dim)

    # outpath_png = "Hanif_Save_PNG.png"

    # # save the image in PNG format with 4 Compression
    # save(outpath_png, img,png_compression=4)

    cv2.waitKey(0)
    #destroy a certain window
    cv2.destroyWindow('My JPG Conversion')

if __name__ == "__main__":
    main()