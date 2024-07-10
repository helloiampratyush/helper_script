import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def torchmetrics_analysis(class_names,
                          predicted_label_tensor:list,
                          true_label_tensor:list

                          ):
    #docstring
    """
    Required input : class_names: list of classes of your data,
                     predicted_label_tensor:overall list of predicted label:on test data,
                     true_label_tensor:overall list of true label:on test data
    """
    
    conf_mat=ConfusionMatrix(num_classes=len(class_names),task="multiclass")

    conf_matTensor=conf_mat(preds=predicted_label_tensor,
                            target=true_label_tensor)

    plot_confusion_matrix(conf_mat=conf_matTensor.numpy(),
                          class_names=class_names,
                          figsize=(10,7)
                          )



