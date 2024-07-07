import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import os

num_workers=os.cpu_count()
# print(num_workers)
def dataloader_creator(train_dir,test_dir,batch_size:int,
                       train_transforms,
                      test_transforms):
    
    #creating test_data and train data

    train_data=datasets.ImageFolder(root=train_dir,
                                    transform=train_transforms,
                                    target_transform=None)
    
    test_data=datasets.ImageFolder(root=test_dir,
                                   transform=test_transforms,
                                   target_transform=None)
    #class_names

    class_names=train_data.classes
    
    #creating data loader
    train_dataloader=DataLoader(dataset=train_data,batch_size=batch_size,
                                shuffle=True,num_workers=num_workers,pin_memory=True)
    
    test_dataloader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,
                               num_workers=num_workers,pin_memory=True)
    
    return class_names,train_dataloader,test_dataloader
