from torch import nn

class minifood101(nn.Module):
    def __init__(self,input_unit:int,
                 hidden_unit:int,
                 output_unit:int,
                 biased:int):
        super().__init__()

        #create layer
        self.layer_1=nn.Sequential(
            nn.Conv2d(in_channels=input_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=hidden_unit,
                      out_channels=hidden_unit,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classification_layer=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_unit*biased,
                      out_features=output_unit)
        )
    
    def forward(self,x):
        x=self.layer_1(x)
        x=self.layer_2(x)
        x=self.classification_layer(x)
        return x

        