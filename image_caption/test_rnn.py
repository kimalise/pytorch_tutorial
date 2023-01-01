import torch
import torch.nn as nn

# lstm: input_size, hidden_size, num_layers
rnn = nn.LSTM(10, 20, 2)

# input: [seq_len, batch_size, input_size]
input = torch.randn(5, 1, 10) 

outputs, states = rnn(input)

print(outputs.shape)
# print(states.shape)
print(outputs[-1])
print(states)
for s in states:
    print(s.shape)

