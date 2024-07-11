import PIL
from PIL import Image
import torch
import glob
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
def test_on_custom_img(model:torch.nn.Module,
                       class_names,
                       img_path,
                       image_transforms,
                       device:torch.device="cpu"):

    """
    your file only should contain .jpg file
    model : model
    class_names : list of classes of out puts
    img_path : path where images are
    image_transforms:transformation of image before getting to model input
    device:device at which you want to train model 

    
    """
    img_path=Path(img_path)
    image_list=img_path.glob("*.jpg")

    for img in image_list:

        img_opened=Image.open(fp=img)

        img_transformed=image_transforms(img_opened).unsqueeze(dim=0)

        model.eval()
        with torch.inference_mode():

            image_logit=model(img_transformed.to(device))

            image_pred=torch.softmax(image_logit,dim=1)

            prediction=image_pred.max().item()

            image_pred_label=torch.argmax(image_pred,dim=1)

        plt.figure()

        plt.imshow(img_opened)
        plt.title(f" predicted:- {class_names[image_pred_label]} prediction prob :- {prediction: .4f}")
        plt.axis(False)
        plt.show()






