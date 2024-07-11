from tqdm.auto import tqdm
import torch
from helper_script import engine
def updated_train(model:torch.nn.Module,
                  test_dataloader:torch.utils.data.DataLoader,
                  train_dataloader:torch.utils.data.DataLoader,
                  loss_fn:torch.nn.Module,
                  optimizer:torch.optim.Optimizer,
                  epoches:int,
                  writer:torch.utils.tensorboard.writer.SummaryWriter,
                  device:torch.device="cuda"
                  ):

  torch.manual_seed(42)
  torch.cuda.manual_seed(42)
  results={"train_loss":[],
           "test_loss":[],
           "train_acc":[],
           "test_acc":[]}

  epoches=epoches

  for epoch in tqdm(range(epoches)):

   train_loss,train_acc=engine.train_step(model=model,
               train_data=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               device=device)

   test_loss,test_acc=engine.test_step(model=model,
                                      test_data=test_dataloader,
                                      loss_fn=loss_fn,
                                      device=device)

   print(f"train loss : {train_loss:.4f} | train acc: {train_acc:.4f} | test acc : {test_acc:.4f} | test loss : {test_loss:.4f}")

   #storing data

   results["train_loss"].append(train_loss)
   results["train_acc"].append(train_acc)
   results["test_loss"].append(test_loss)
   results["test_acc"].append(test_acc)


   if writer:

    writer.add_scalars(main_tag="train Loss",
                       tag_scalar_dict={
                           "train loss":train_loss
                       },
                       global_step=epoch)
    writer.add_scalars(main_tag="test Loss",
                       tag_scalar_dict={
                           "test loss":test_loss
                       },
                       global_step=epoch)

    writer.add_scalars(main_tag="test accuracy",
                       tag_scalar_dict={
                           "test loss":test_acc
                       },
                       global_step=epoch)

    writer.add_scalars(main_tag="train accuracy",
                       tag_scalar_dict={
                           "test loss":train_acc
                       },
                       global_step=epoch)

    writer.close()

  else:
    pass

  return results
