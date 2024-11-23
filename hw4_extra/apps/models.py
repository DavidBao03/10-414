import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def ConvBN(in_channels, out_channels, kernel_size, stride, device, dtype):
    return nn.Sequential(
        nn.Conv(in_channels, out_channels, kernel_size, stride=stride, bias=True, device=device, dtype=dtype),
        nn.BatchNorm2d(out_channels, device=device, dtype=dtype),
        nn.ReLU()
    )

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.convBN1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.convBN2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)

        module1 = nn.Sequential(
            ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
            ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        )
        self.residual1 = nn.Residual(module1)

        self.convBN3 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.convBN4 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)

        module2 = nn.Sequential(
            ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
            ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        )
        self.residual2 = nn.Residual(module2)

        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)

        self.flatten = nn.Flatten()
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.convBN1(x)
        x = self.convBN2(x)
        x = self.residual1(x)

        x = self.convBN3(x)
        x = self.convBN4(x)
        x = self.residual2(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=20, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'transformer':
            self.seq_model = nn.Transformer(
                embedding_size,
                hidden_size,
                num_layers, 
                device=device,
                dtype=dtype,
                sequence_len=seq_len,
            )
        else:
            raise ValueError("Not Support Seq Model")
        
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype) if seq_model != 'transformer' \
                      else nn.Linear(embedding_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size = x.shape
        embedded = self.embedding(x)
        out, h = self.seq_model(embedded, h)

        if isinstance(self.seq_model, nn.Transformer):
            out = out.reshape((seq_len * batch_size, self.embedding_size))
            out = self.linear(out)
        else:
            out = out.reshape((seq_len * batch_size, self.hidden_size))
            out = self.linear(out)

        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)