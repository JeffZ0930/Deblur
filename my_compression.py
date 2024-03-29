#Python OpenCV script to test out JPEG compression rates

import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def jpeg_compression(image, quality=50):
    # Convert PyTorch tensor to NumPy array
    image_numpy = (image * 255).numpy().astype(np.uint8)
    print(image_numpy.shape)
    _, compressed_img = cv2.imencode('.jpg', image_numpy, [cv2.IMWRITE_JPEG_QUALITY, quality])

    result = cv2.imdecode(compressed_img, cv2.IMREAD_GRAYSCALE)

    # Normalize the image to the range [0, 1]
    result = result / 255.0

    result = torch.from_numpy(result).float()

    return result

def resize(path, image, dim):
    # resize imagek
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(path, resized)

def main(): 
    # batch_size = 64
    # train_dataset = ImageFolder(root="./mnist_clear", transform=transforms.Compose([transforms.ToTensor()]))

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # for batch in train_loader:
    #     images, _ = batch
    #     for img in images:
    #         compressed_img = jpeg_compression(img, 1)
    #         # save the compressed image
    #         cv2.imwrite("./mnist_blur", compressed_img)
    mnist_input_path = "./mnist_clear"
    mnist_output_path = "./mnist_blur"
    label_folder = os.listdir(mnist_input_path)

    for label in label_folder:
        print("Bluring label " + label)
        label_input_path = os.path.join(mnist_input_path, label)
        label_output__path = os.path.join(mnist_output_path, label)
        imgs = os.listdir(label_input_path)

        for img_file in imgs:
            img_input_path = os.path.join(label_input_path, img_file)
            img = cv2.imread(img_input_path)
        
            output_path = os.path.join(label_output__path, img_file)

            # save the image in JPEG format with 1% quality
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 1])


            



if __name__ == "__main__":
    main()