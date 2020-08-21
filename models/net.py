import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinNet(nn.Module):
  def __init__(self, params):
    super(ProteinNet, self).__init__()
    self.infeatures = params.infeatures
    self.hidden = params.hidden
    self.num_classes1 = params.classes1
    self.num_classes2 = params.classes2
    self.linear1 = nn.Linear(self.infeatures, self.hidden)
    self.linear2 = nn.Linear(self.hidden, self.num_classes1)
    self.linear3 = nn.Linear(self.hidden, self.num_classes2)
    self.dropout = nn.Dropout(p = 0.25)
    self.batchnorm = nn.BatchNorm1d(self.hidden)
    self.relu = nn.ReLU()
    
  def forward(self,x):
    x = self.linear1(x)
    x = self.dropout(x)
    x = self.batchnorm(x)
    x = self.relu(x)
    x = self.dropout(x)
    loc = self.linear2(x)
    mem = self.linear3(x)

    return F.log_softmax(loc, -1), F.log_softmax(mem,-1)
    
def loss_fn(log_probs, targ1hot):
    # (N x 10 or 3) * (N x 10 or 3)
    loss = nn.NLLLoss()
    return loss

def accuracy(outputs, labels):
    # N x 10(3) -> N x 1
    outputs = np.argmax(outputs, axis=-1)
    outputs = outputs.flatten()
    #N x 1
    labels = labels.flatten()
    return np.sum(outputs==labels)/float(outputs.shape[0])


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'Loc_accuracy': accuracy, 'Mem_accuracy': accuracy, 
    # could add more metrics such as accuracy for each token type
}