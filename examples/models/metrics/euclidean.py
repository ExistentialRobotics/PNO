import torch
import numpy as np

class EuclideanMetric(torch.nn.Module):
  def forward(self, x, y):
    return torch.norm(x - y, dim=-1)
