import zipfile
from pathlib import Path
import requests
import os
def data_downloader(target_dir:str,target_path:str,
                    uri:str):

  target_dir=Path(target_dir)

  if target_dir.is_dir():
    print("[INFO] directory exists already")

  else:
    print(f"[INFO] can not found your direcory making new one")
    target_dir.mkdir(parents=True,exist_ok=True)

  #download and extract
  with open(target_dir/"zip01.zip","wb") as f:
    request=requests.get(url=uri)
    f.write(request.content)

  #unzipping
  print("extracting your file")
  with zipfile.ZipFile(target_dir/"zip01.zip","r") as zip_ref:
    zip_ref.extractall(target_dir/target_path)

  #removing zipfile
  os.remove(target_dir/"zip01.zip")
  print("removal of zipfile is done")

  return target_dir/target_path
