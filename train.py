import data_loader,utils,model_creation,engine
from pathlib import Path
import os
import torch
import matplotlib.pyplot as plt
import argparse

#here we will not always create new model

train_dir=Path("data/pizza_steak_sushi/train")
test_dir=Path("data/pizza_steak_sushi/test")

if train_dir.is_dir() and test_dir.is_dir():
    print("got your directory processing further...")
    #setting parameters
    parser=argparse.ArgumentParser(description="setting up parameters")
    parser.add_argument('--hidden_unit', type=int, default=10,
                    help='hidden unit in the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch_size of training")
    parser.add_argument('--biased', type=int, default=1,
                        help="extra bias for training")
    parser.add_argument('--learning_rate', type=float,default=0.0001,
                        help="learnin rate of training")
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of epochs for training")
    
    args=parser.parse_args()
    HIDDEN_UNIT=args.hidden_unit
    BATCH_SIZE=args.batch_size
    BIASED=args.biased
    LEARNING_INDEX=args.learning_rate
    EPOCHS=args.epochs
    #preparing data for you
    resize_len,resize_width=224,224
    num_workers=os.cpu_count()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #create data loader
    class_names,train_dataloader,test_dataloader=data_loader.dataloader_creator(train_dir=train_dir,test_dir=test_dir,
                                          resize_len=resize_len,resize_width=resize_width,
                                          batch_size=BATCH_SIZE,augmentation=True)

    #check if model already exists
    model_save_path=Path("model/mini_foodvision_101v1")
    if model_save_path.is_file():
        print("model already exists using model state_dict")
        model=model_creation.minifood101(input_unit=3,
                                         hidden_unit=HIDDEN_UNIT,
                                         output_unit=len(class_names),
                                         biased=BIASED).to(device)
        
        model.load_state_dict(torch.load(model_save_path))


    else:
        print("model does not exist making brand new")
        model=model_creation.minifood101(input_unit=3,
                                         hidden_unit=HIDDEN_UNIT,
                                         output_unit=len(class_names),
                                         biased=BIASED).to(device)
        
    
    #training time
    results=engine.train(model=model,
                          device=device,
                          train_dataloader=train_dataloader,
                          test_dataloader=test_dataloader,
                          learning_index=LEARNING_INDEX,
                          epoches=EPOCHS,
                          )
    


    #save model
    utils.save_model(model)
      