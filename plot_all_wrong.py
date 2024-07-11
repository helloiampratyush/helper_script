import torch
import matplotlib.pyplot as plt
from torchvision import transforms

def plot_all_wrong_predicted(dis_prediction_tensor:list,
                             class_names,
                             predicted_label_tensor:list,
                             true_label_tensor:list,
                             image_list_tensor:list
                             ):
  

    """
    doctring!
    input : dis_pridiction_tensor : distributed predicted tensor on all test data : list
            class_names : list of all out put classes:list
            predicted_labe_tensor : predicted label tensor list of all test data:list
            true_label_tensor : true label of all result of test data
            imge_list_tensor : a list of all test images

    """
    
    prediction=[]
    #store prediction in good manner
    for i in range(len(dis_prediction_tensor)):

        prediction.append(dis_prediction_tensor[i].max().item())

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )


    for i in range(len(image_list_tensor)):

        #if label will not match then we will plot image

        if predicted_label_tensor[i].item() != true_label_tensor[i].item():

            img=inv_normalize(image_list_tensor[i])

            plt.figure()

            plt.imshow(img.permute(1,2,0))

            plt.title(f"predicted label: {class_names[predicted_label_tensor[i]]} | true label : {class_names[true_label_tensor[i]]} | prediction probability : {prediction[i]:.4f}")

            plt.axis(False)

    plt.show()

    


