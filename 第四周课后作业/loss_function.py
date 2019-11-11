import torch
import torch.nn as nn
# fake data
inputs = torch.tensor([[1, 2], [1, 3], [1, 3]], dtype=torch.float)
target = torch.tensor([0, 1, 1], dtype=torch.long)

# ----------------------------------- CrossEntropy loss: reduction -----------------------------------
# def loss function
weights = torch.tensor([1, 2], dtype=torch.float)
loss_f_mean = nn.CrossEntropyLoss(weight=weights, reduction='mean')

# forward
loss_mean = loss_f_mean(inputs, target)

# view
print("\nweights: \n", weights)

# view
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
# loss(x,class)=weight[class](−x[class]+log(∑ exp(x[j]))) 如果reduction='sum', 那么返回loss(x, class)
# loss_mean = loss(x,class)/ ∑ weight[class]
print("\nCross Entropy Loss:\n ", loss_mean)