import torch
import torch.nn as nn
from torch import optim
import torchaudio
import random

class AudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AudioRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
       # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
       # output = self.softmax(output)
        return output, hidden

class PositionalEncoding(nn.Module):

class ResidualBlock(nn.Module):
    pass

class DilatedConvolution(nn.Module0):
    pass

def preprocess(input):
    mu = 255
    convData = np.sign(input)*(np.log(1 + mu * np.abs(input)))/(np.log(1+mu))
    convData = (convData + 1) * 255 / 2
    convData = convData.type(torch.int64)
    inputData = np.zeros((convData.shape[1], 255))
    for i in range(inputData.shape[0]):
        inputData[i][convData[0][i]] = 1

    return inputData.reshape(inputData.shape[0], 1, 256)

n_hidden = 128
rnn = RNN(256, n_hidden, 256)
#hidden_state = torch.zeros(1, 128)
#output, next_hidden = rnn(inputData[0], hidden_state)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.01) 

y, fr  = torchaudio.load('/content/Dataset.wav')
#Now we Train (Let's do random 5 second clips at a time, this is fr * 5 in length)
training_segments = 1000
segment_duration = 5

data = preprocess(y)
rnn.train()
for i in range(training_segments):
    start_index = random.randint(0, y[0] - (fr * segment_duration) - 1)
    hidden_state = torch.zeros(1, n_hidden)

    for j in range(fr * segment_duration):
        output, hidden_state = rnn(data[j + start_index], hidden_state)

    loss = loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




