import torch
from torch import nn

def prediction_on_entire_test_data(model:torch.nn.Module,
                                   test_dataloader:torch.utils.data.DataLoader,
                                   device:torch.device="cpu"
                                   ):
    
    """
    step 1. This is for analysis of your model to see where it is wrong so first train your model.
    
    step 2. input requirement: model : your model, test_dataloader: put you entire test_loader data,loss_fn: loss_fn

    device: On which device you want to test data

    returning statement:- dis_prediction_tensor:it is distributed probability among all item
    ,image_list_tensor: it will be all your test image
    ,true_label_tensor:it is true label of image
    ,predicted_label_tensor:it is predicted label of image

                                                     
    """
    #variable will keep your record
    dis_prediction=[]
    true_label=[]
    predicted_label=[]
    image_list=[]

    #start
    model.to(device)

    #setting torch.inference mode
    model.eval()
    with torch.inference_mode():

        #looping

        for x_test,y_test in test_dataloader:

            x_test,y_test=x_test.to(device),y_test.to(device)
            image_list.append(x_test)

            y_test_logit=model(x_test)

            y_test_pred=torch.softmax(y_test_logit,dim=1)

            dis_prediction.append(y_test_pred)

            y_test_pred_label=torch.argmax(y_test_pred,dim=1)

            true_label.append(y_test)

            predicted_label.append(y_test_pred_label)

        dis_prediction_tensor=torch.cat(dis_prediction).cpu()

        image_list_tensor=torch.cat(image_list).cpu()

        true_label_tensor=torch.cat(true_label).cpu()

        predicted_label_tensor=torch.cat(predicted_label).cpu()


        return dis_prediction_tensor,image_list_tensor,true_label_tensor,predicted_label_tensor







