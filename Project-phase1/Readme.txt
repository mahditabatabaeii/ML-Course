import torch
model = CIFAR10Classifier()
state_dict = torch.load("model_state_dict.pth")
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('_module.', '')  
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()