
import torchvision
import torch
from torchvision import transforms
def test_sample_prepare(sample_path):
    #sample preparation
    sample_image=torchvision.io.read_image(sample_path).type(torch.float32)

    #number (0-1)
    # print(sample_image.shape)
    sample_image=sample_image/255.0
    # print(sample_image)
    transformator=transforms.Compose([
        transforms.Resize(size=(224,224))
    ])

    img=transformator(sample_image)
    print(img.shape)
    return img