import torchvision
import torch
from torch import nn
from torchinfo import summary
def alexnet(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,get_transforms
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.AlexNet_Weights.DEFAULT

    #model
    model=torchvision.models.alexnet(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=9216,out_features=4096,bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=4096,out_features=4096,bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=4096,out_features=out_features,bias=True)
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
    

def densenet161(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.DenseNet161_Weights.DEFAULT

    #model
    model=torchvision.models.densenet161(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Linear(in_features=2208,
                       out_features=out_features,
                       bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

    
def densenet169(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.DenseNet169_Weights.DEFAULT

    #model
    model=torchvision.models.densenet169(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Linear(in_features=1664,
                       out_features=out_features,
                       bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def densenet201(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.DenseNet201_Weights.DEFAULT

    #model
    model=torchvision.models.densenet201(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classification layer

    model.classifier=nn.Linear(in_features=1920,
                       out_features=out_features,
                       bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def efficientnet_b0(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b0(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classification layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model

def efficientnet_b1(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b1(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classification layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
def efficientnet_b2(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b2(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classification layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.3,inplace=True),
        nn.Linear(in_features=1408,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
    
def efficientnet_b3(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b3(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classification layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.3,inplace=True),
        nn.Linear(in_features=1536,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def efficientnet_b4(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b4(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classification layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.4,inplace=True),
        nn.Linear(in_features=1792,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
    
def efficientnet_b5(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B5_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b5(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.4,inplace=True),
        nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

   
    
def efficientnet_b6(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B6_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b6(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.5,inplace=True),
        nn.Linear(in_features=2304,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def efficientnet_b7(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_b7(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.5,inplace=True),
        nn.Linear(in_features=2560,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model


    
def efficientnet_v2_l(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_V2_L_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_v2_l(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.4,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
    
def efficientnet_v2_m(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_V2_M_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_v2_m(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.3,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def efficientnet_v2_s(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT

    #model
    model=torchvision.models.efficientnet_v2_s(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    )
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
def googlenet(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.GoogLeNet_Weights.DEFAULT

    #model
    model=torchvision.models.googlenet(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing fc layer
    model.dropout=nn.Dropout(p=0.2,inplace=False).to(device)

    model.fc=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def inception_v3(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Inception_V3_Weights.DEFAULT

    #model
    model=torchvision.models.inception_v3(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing fc layer
    model.dropout=nn.Dropout(p=0.5,inplace=False).to(device)

    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def maxvit_t(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,get_transforms
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MaxVit_T_Weights.DEFAULT

    #model
    model=torchvision.models.maxvit_t(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(start_dim=1,end_dim=-1),
        nn.LayerNorm((512,),eps=0.00001,elementwise_affine=True),
        nn.Linear(in_features=512,out_features=512,bias=True),
        nn.Tanh(),
        nn.Linear(in_features=512,out_features=out_features,bias=False)

    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def mnasnet0_5(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MNASNet0_5_Weights.DEFAULT

    #model
    model=torchvision.models.mnasnet0_5(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
    
def mnasnet0_75(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MNASNet0_75_Weights.DEFAULT

    #model
    model=torchvision.models.mnasnet0_75(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def mnasnet1_0(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MNASNet1_0_Weights.DEFAULT

    #model
    model=torchvision.models.mnasnet1_0(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
def mnasnet1_3(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MNASNet1_3_Weights.DEFAULT

    #model
    model=torchvision.models.mnasnet1_3(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def mobilenet_v2(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MobileNet_V2_Weights.DEFAULT

    #model
    model=torchvision.models.mobilenet_v2(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Dropout(p=0.2,inplace=False),
        nn.Linear(in_features=1280,out_features=out_features,bias=True).to(device)
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

    

def mobilenet_v3_large(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT

    #model
    model=torchvision.models.mobilenet_v3_large(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Linear(in_features=960,out_features=1280,bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1280,out_features=out_features,bias=True)
       
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model


def mobilenet_v3_small(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT

    #model
    model=torchvision.models.mobilenet_v3_small(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer

    model.classifier=nn.Sequential(
        nn.Linear(in_features=576,out_features=1024,bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2,inplace=True),
        nn.Linear(in_features=1024,out_features=out_features,bias=True)
       
    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_x_16gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_X_16GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_x_16gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model


def regnet_x_1_6gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_X_1_6GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_x_1_6gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=912,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_x_32gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_X_32GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_x_32gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2520,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_x_3_2gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_X_3_2GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_x_3_2gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=1008,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_x_400mf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_x_400mf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=400,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
def regnet_x_800mf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_X_800MF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_x_800mf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=672,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def regnet_x_8gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_X_8GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_x_8gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=1920,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_y_128gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_128GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_128gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=7392,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
def regnet_y_16gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_16GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_16gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=3024,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
def regnet_y_1_6gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_1_6GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_1_6gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=888,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_y_32gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_32GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_32gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=3712,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_y_3_2gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_3_2GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_3_2gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=1512,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_y_400mf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_400mf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=440,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def regnet_y_800mf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_800mf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=784,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def regnet_y_8gf(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.RegNet_Y_8GF_Weights.DEFAULT

    #model
    model=torchvision.models.regnet_y_8gf(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2016,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnet101(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNet101_Weights.DEFAULT

    #model
    model=torchvision.models.resnet101(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model



    

def resnet152(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNet152_Weights.DEFAULT

    #model
    model=torchvision.models.resnet152(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnet18(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNet18_Weights.DEFAULT

    #model
    model=torchvision.models.resnet18(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=512,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnet34(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNet34_Weights.DEFAULT

    #model
    model=torchvision.models.resnet34(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=512,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def resnet34(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNet34_Weights.DEFAULT

    #model
    model=torchvision.models.resnet34(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=512,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnet50(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNet50_Weights.DEFAULT

    #model
    model=torchvision.models.resnet50(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnext101_32x8d(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT

    #model
    model=torchvision.models.resnext101_32x8d(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnet50(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNet50_Weights.DEFAULT

    #model
    model=torchvision.models.resnet50(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnext101_64x4d(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT

    #model
    model=torchvision.models.resnext101_64x4d(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def resnext50_32x4d(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT

    #model
    model=torchvision.models.resnext50_32x4d(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def shufflenet_v2_x0_5(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT

    #model
    model=torchvision.models.shufflenet_v2_x0_5(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.fc=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def shufflenet_v2_x0_5(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT

    #model
    model=torchvision.models.shufflenet_v2_x0_5(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.fc=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def shufflenet_v2_x1_0(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT

    #model
    model=torchvision.models.shufflenet_v2_x1_0(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.fc=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def shufflenet_v2_x1_5(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT

    #model
    model=torchvision.models.shufflenet_v2_x1_5(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.fc=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def shufflenet_v2_x2_0(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT

    #model
    model=torchvision.models.shufflenet_v2_x2_0(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def squeezenet1_0(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT

    #model
    model=torchvision.models.squeezenet1_0(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
        nn.Dropout(p=0.5,inplace=False),
        nn.Conv2d(in_channels=512,out_channels=out_features,
                  kernel_size=1,
                  stride=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1,1))

    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def squeezenet1_1(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT

    #model
    model=torchvision.models.squeezenet1_1(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
        nn.Dropout(p=0.5,inplace=False),
        nn.Conv2d(in_channels=512,out_channels=out_features,
                  kernel_size=1,
                  stride=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1,1))

    ).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def swin_b(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Swin_B_Weights.DEFAULT

    #model
    model=torchvision.models.swin_b(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.norm=nn.LayerNorm((1024,),eps=0.00001,elementwise_affine=True).to(device)
    model.head=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def swin_t(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Swin_T_Weights.DEFAULT

    #model
    model=torchvision.models.swin_t(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.norm=nn.LayerNorm((768,),eps=0.00001,elementwise_affine=True).to(device)
    model.head=nn.Linear(in_features=768,out_features=out_features,bias=True).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def swin_s(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Swin_S_Weights.DEFAULT

    #model
    model=torchvision.models.swin_s(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.norm=nn.LayerNorm((768,),eps=0.00001,elementwise_affine=True).to(device)
    model.head=nn.Linear(in_features=768,out_features=out_features,bias=True).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def swin_v2_b(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Swin_V2_B_Weights.DEFAULT

    #model
    model=torchvision.models.swin_v2_b(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.norm=nn.LayerNorm((1024,),eps=0.00001,elementwise_affine=True).to(device)
    model.head=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def swin_v2_s(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Swin_V2_S_Weights.DEFAULT

    #model
    model=torchvision.models.swin_v2_s(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.norm=nn.LayerNorm((768,),eps=0.00001,elementwise_affine=True).to(device)
    model.head=nn.Linear(in_features=768,out_features=out_features,bias=True).to(device) 
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
       

def swin_v2_t(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Swin_V2_T_Weights.DEFAULT

    #model
    model=torchvision.models.swin_v2_t(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.norm=nn.LayerNorm((1024,),eps=0.00001,elementwise_affine=True).to(device)
    model.head=nn.Linear(in_features=1024,out_features=out_features,bias=True).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def vgg11(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG11_Weights.DEFAULT

    #model
    model=torchvision.models.vgg11(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def vgg11_bn(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG11_BN_Weights.DEFAULT

    #model
    model=torchvision.models.vgg11_bn(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    



def vgg13_bn(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG13_BN_Weights.DEFAULT

    #model
    model=torchvision.models.vgg13_bn(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def vgg13(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG13_Weights.DEFAULT

    #model
    model=torchvision.models.vgg13(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

    

def vgg16_bn(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG16_BN_Weights.DEFAULT

    #model
    model=torchvision.models.vgg16_bn(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def vgg16(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG16_Weights.DEFAULT

    #model
    model=torchvision.models.vgg16(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def vgg19_bn(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG19_BN_Weights.DEFAULT

    #model
    model=torchvision.models.vgg19_bn(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


def vgg19(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.VGG19_Weights.DEFAULT

    #model
    model=torchvision.models.vgg19(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.classifier=nn.Sequential(
         nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
       nn.Linear(in_features=4096, out_features=4096, bias=True),
       nn.ReLU(inplace=True),
       nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=out_features, bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    


    

def vit_b_16(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ViT_B_16_Weights.DEFAULT

    #model
    model=torchvision.models.vit_b_16(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.heads=nn.Sequential(
        nn.Linear(in_features=768,out_features=out_features,bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def vit_b_32(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ViT_B_32_Weights.DEFAULT

    #model
    model=torchvision.models.vit_b_32(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.heads=nn.Sequential(
        nn.Linear(in_features=768,out_features=out_features,bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    
def vit_h_14(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ViT_H_14_Weights.DEFAULT

    #model
    model=torchvision.models.vit_h_14(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.heads=nn.Sequential(
        nn.Linear(in_features=1280,out_features=out_features,bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def vit_l_16(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ViT_L_16_Weights.DEFAULT

    #model
    model=torchvision.models.vit_l_16(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.heads=nn.Sequential(
        nn.Linear(in_features=1024,out_features=out_features,bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    

def vit_l_32(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.ViT_L_32_Weights.DEFAULT

    #model
    model=torchvision.models.vit_l_32(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.heads=nn.Sequential(
        nn.Linear(in_features=1024,out_features=out_features,bias=True)
    ).to(device)
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    



def wide_resnet101_2(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Wide_ResNet101_2_Weights.DEFAULT

    #model
    model=torchvision.models.wide_resnet101_2(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model

  

def wide_resnet50_2(out_features:int,device:torch.device ="cuda",
            show_summary:bool=False,get_transforms:bool=False):
    #docstring
    """
    Input parameter
    out_features : no of out put you wants
    device : on which device you want to make model recommended: "cuda" default is cuda
    show_summary: if you want to see summary:show_summary=True default is false
    get_transforms: Also want to get auto_transforms in return default is false


    Return
            
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model
    """

    #weights
    weights=torchvision.models.Wide_ResNet50_2_Weights.DEFAULT

    #model
    model=torchvision.models.wide_resnet50_2(weights=weights).to(device)

    #freezing layer

    for param in model.parameters():

        param.requires_grad=False

    
    #computing auto_transforms

    auto_transform=weights.transforms()
    
    #unfreezing classifier layer
    model.avgpool=nn.AdaptiveAvgPool2d(output_size=(1,1)).to(device)
    model.fc=nn.Linear(in_features=2048,out_features=out_features,bias=True).to(device)
    
    if show_summary is True:

      print( summary(model=model,
                input_size=(32,3,224,224),
                col_width=25,
                col_names=["input_size","output_size","num_params","trainable"])
        )
    if get_transforms is True:

        return model,auto_transform
    
    else:
        return model

  


    


    



    

    

    
    

    






    


    


    









    



    



    


       


    
    


    


    


    

    
    

    





    
    

    


    
    


   
