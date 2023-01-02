import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(log_dir="run")

for i in range(100):
    writer.add_scalar('Loss/train', np.random.random(), i)
    writer.add_scalar('Loss/test', np.random.random(), i)
    writer.add_scalar('Accuracy/train', np.random.random(), i)
    writer.add_scalar('Accuracy/test', np.random.random(), i)
    

