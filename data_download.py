from pathlib import Path
import requests
import zipfile
import os
#download the load filen name
save_filename=Path("data/")

save_filename_path=save_filename/"pizza_steak_sushi"

if save_filename.is_dir():
    print("skipping making your directory it already exists")

else:
    save_filename.mkdir(parents=True, exist_ok=True)
    print("Directory created successfully")

    with open(save_filename/"pizza_steak_sushi.zip","wb") as f:
     request=requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
     f.write(request.content)

# extract the zip file
with zipfile.ZipFile(save_filename/"pizza_steak_sushi.zip","r") as zip_ref:
    zip_ref.extractall(save_filename_path)

print("Files extracted successfully")

os.remove(save_filename/"pizza_steak_sushi.zip")
