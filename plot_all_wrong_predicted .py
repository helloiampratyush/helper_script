import torch
import matplotlib.pyplot as plt
from torchvision import transforms

def plot_all_wrong_predicted(dis_prediction_tensor:list,
                             class_names,
                             predicted_label_tensor:list,
                             true_label_tensor:list,
                             image_list_tensor:list
                             ):
    
    prediction=[]
    #store prediction in good manner
    for i in range(len(dis_prediction_tensor)):

        prediction[i].append(dis_prediction_tensor[i].max().item())

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )


    for i in range(len(image_list_tensor)):

        #if label will not match then we will plot image

        if predicted_label_tensor[i].item() != true_label_tensor[i].item():

            img=inv_normalize(image_list_tensor[i])

            plt.figure()

            plt.imshow(img)

            plt.title(f"predicted label: {class_names[predicted_label_tensor[i]]} | true label : {class_names[predicted_label_tensor[i]]} | prediction probability : {prediction[i]:.4f}")

            plt.axis(False)

    plt.show()

    


