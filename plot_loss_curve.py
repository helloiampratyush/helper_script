import torch
import matplotlib.pyplot as plt
from typing import Dict, List
def single_model_curve(results:dict[str,list]):


    #docstring

    """
    results should contain results["test_loss"],
    results["train_loss"] , results["test_acc"],
    results["train_acc"]

    """

    #results ploting
    x=range(len(results["train_loss"]))
    
    plt.figure(figsize=(10,7))
    plt.subplot(2,2,1)
    
    plt.title("loss of v2")
    plt.plot(x,results["train_loss"],c="r",label="train loss")
    plt.plot(x,results["test_loss"],c="b",label="test loss")
    plt.xlabel("epoches")
    plt.subplot(2,2,2)
    
    plt.title("accuracy of v2")
    plt.plot(x,results["train_acc"],c="r",label="train accuracy")
    plt.plot(x,results["test_acc"],c="b",label="test accuracy")
    plt.xlabel("epoches")
    plt.legend()

    plt.show()

def double_model_compare(model_1_results:dict[str,list],
                         model_2_results:dict[str,list],
                         model_1_name:str,
                         model_2_name:str):
    
    """
    results should contain results["test_loss"],
    results["train_loss"] , results["test_acc"],
    results["train_acc"]

    [Recommended] both model should be train on same epoches

    """

    x=range(len(model_1_results["train_loss"]))

    plt.figure(figsize=(10,9))

    plt.subplot(2,2,1)

    plt.title(" Model Train Loss")
    plt.plot(x,model_1_results["train_loss"],label=model_1_name,c="r")
    plt.plot(x,model_2_results["train_loss"],label=model_2_name,c="b")
    plt.xlabel("Epoches")

    plt.subplot(2,2,2)

    plt.title(" Model Test Loss")
    plt.plot(x,model_1_results["test_loss"],label=model_1_name,c="r")
    plt.plot(x,model_2_results["test_loss"],label=model_2_name,c="b")
    plt.xlabel("Epoches")

    plt.subplot(2,2,3)

    plt.title(" Model Train Accuracy")
    plt.plot(x,model_1_results["train_acc"],label=model_1_name,c="r")
    plt.plot(x,model_2_results["train_acc"],label=model_2_name,c="b")
    plt.xlabel("Epoches")

    plt.subplot(2,2,4)

    plt.title(" Model Test Accuracy")
    plt.plot(x,model_1_results["test_acc"],label=model_1_name,c="r")
    plt.plot(x,model_2_results["test_acc"],label=model_2_name,c="b")
    plt.xlabel("Epoches")

    plt.legend()

    plt.show()


def multi_model_compare(results_list: Dict[str, Dict[str, List[float]]]):
    """
    [Important]
    Your results_list must contain input like this : dict[dict[str,list],str]

    results should contain results["test_loss"],
    results["train_loss"] , results["test_acc"],
    results["train_acc"]

    results_list[results:"model_name"]

    [NOTE]! Every model should be trained on the same epochs
    """

    # Ensure that the input dictionary is not empty
    if not results_list:
        raise ValueError("results_list is empty")

    # Determine the number of epochs from the first model
    first_key = next(iter(results_list))
    num_epochs = len(results_list[first_key]["train_loss"])
    x = range(num_epochs)

    plt.figure(figsize=(10, 9))

    # Plotting Model Train Loss
    plt.subplot(2, 2, 1)
    plt.title("Model Train Loss")
    for model_name, results in results_list.items():
        plt.plot(x, results["train_loss"], label=f"{model_name} train loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plotting Model Test Loss
    plt.subplot(2, 2, 2)
    plt.title("Model Test Loss")
    for model_name, results in results_list.items():
        plt.plot(x, results["test_loss"], label=f"{model_name} test loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plotting Model Train Accuracy
    plt.subplot(2, 2, 3)
    plt.title("Model Train Accuracy")
    for model_name, results in results_list.items():
        plt.plot(x, results["train_acc"], label=f"{model_name} train accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Plotting Model Test Accuracy
    plt.subplot(2, 2, 4)
    plt.title("Model Test Accuracy")
    for model_name, results in results_list.items():
        plt.plot(x, results["test_acc"], label=f"{model_name} test accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()


    
 