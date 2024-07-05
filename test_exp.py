import model_creation,sample_data
from pathlib import Path
import torch
import matplotlib.pyplot as plt
image_path=Path("Test_sample/pizza.jpg")

img=sample_data.test_sample_prepare(sample_path=image_path)

model=model_creation.minifood101(input_unit=3,hidden_unit=10,output_unit=3,biased=3136)

model_save_path=Path("model/mini_foodvision_101v1")

model.load_state_dict(torch.load(model_save_path))

pred_prob=model(img.unsqueeze(dim=0))

pred_prob=torch.softmax(pred_prob,dim=1)

prediction=pred_prob.max().item()

pred_label=torch.argmax(pred_prob,dim=1)

class_names={0:"pizza",
             1:"steak",
             2:"sushi"}
print(pred_label.item())
print(f"Predicted class: {class_names[pred_label.item()]}")
plt.figure(figsize=(10,7))
plt.imshow(img.permute(1,2,0))
plt.axis(False)
plt.title(f"Predicted class: {class_names[pred_label.item()]}| probability: {prediction:.4f}")
plt.show()


