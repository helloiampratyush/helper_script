# we will create train,test and training step

import torch
from torch import nn
from tqdm.auto import tqdm
def train_step(model:torch.nn.Module,
               train_data:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device:torch.device):
    
    model.train()

    #initiate loss and accuracy

    train_loss,train_acc=0,0

    for batch,(X,y) in enumerate(train_data):

        #change to device

        X,y=X.to(device),y.to(device)

        #forward pass

        y_train_logit=model(X)

        y_train_prob=torch.softmax(y_train_logit,dim=1)

        y_train_label=torch.argmax(y_train_prob,dim=1)


        #loss
        loss=loss_fn(y_train_logit,y)

        train_loss += loss.item()

        train_acc += (y_train_label==y).sum().item()/len(y_train_logit)

        #optimizer zero grad

        optimizer.zero_grad()

        #loss backward

        loss.backward()

        #optimizer step

        optimizer.step()

    #loss distribution
    train_loss /= len(train_data)

    train_acc /= len(train_data)

    return train_loss,train_acc


def test_step(model:torch.nn.Module,
              test_data:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device):
    
    
    model.eval()
    with torch.inference_mode():
        #initiate test_loss,test_acc

        test_loss,test_acc=0,0
        #initiate loop
        for batch,(x_test,y_test) in enumerate(test_data):

            #change device 
            x_test,y_test= x_test.to(device),y_test.to(device)

            #forward pass

            y_pred=model(x_test)

            y_pred_prob=torch.softmax(y_pred,dim=1)

            y_pred_label=torch.argmax(y_pred_prob,dim=1)

            #loss

            loss=loss_fn(y_pred,y_test)

            #loss and accuracy accumulation

            test_loss += loss.item()

            test_acc += (y_pred_label==y_test).sum().item()/len(y_pred)

        # reaverage the loss and accuracy

        test_loss /= len(test_data)
        test_acc /=len(test_data)

    return test_loss,test_acc

# creting training loop step

def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          learning_index:int,
          epoches:int,
          device:torch.device):
    
    #create loss fn and optimizer
    loss_fn=nn.CrossEntropyLoss()

    optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_index)

    #creating dictionary to storing data

    results= {
        "train_loss" :[],
        "train_acc" : [],
        "test_loss" :[],
        "test_acc" :[]
                }
    
    for epoch in tqdm(range(epoches)):

        train_loss,train_acc=train_step(model=model,
                                        train_data=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)
        
        test_loss,test_acc=test_step(model=model,
                                     test_data=test_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)
        
        #print then store

        print(f"train loss: {train_loss : .3f} |  train acc: {train_acc : .3f} | test loss : {test_loss: .3f}  | test acc : {test_acc :.3f}")

        #storing data to results

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    #returning results
    return results
    


