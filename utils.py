import torch
from pathlib import Path

def save_model(model:torch.nn.Module,save_file_path:str):

 save_dir=Path("models")
 
 model_save_path=save_dir/save_file_path

 # checking if directory is already available
 if save_dir.is_dir():
  print("skipping")

 else:
  save_dir.mkdir(parents=True,exist_ok=True)

 #saving your model

 torch.save(obj=model.state_dict(),f=model_save_path)

 print(f"your model has been saved in {model_save_path}")
