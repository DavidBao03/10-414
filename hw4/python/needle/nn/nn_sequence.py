"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return  (1.0 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = np.sqrt(1.0 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, 
                                        low=-bound, high=bound, 
                                        device=device, dtype=dtype,requires_grad=True))
        
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, 
                                        low=-bound, high=bound, 
                                        device=device, dtype=dtype,requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, 
                                            low=-bound, high=bound, 
                                            device=device, dtype=dtype,requires_grad=True))
            
            self.bias_hh = Parameter(init.rand(hidden_size, 
                                            low=-bound, high=bound, 
                                            device=device, dtype=dtype,requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.nonlinearity = nonlinearity
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.device = device
        self.dtype = dtype
        self.bias = bias

        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        if h is None:
            h = init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        
        if self.bias:
            bias_hh = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((batch_size, self.hidden_size))
            bias_ih = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((batch_size, self.hidden_size))
            out = X @ self.W_ih + h @ self.W_hh + bias_hh + bias_ih
        else:
            out = X @ self.W_ih + h @ self.W_hh
        
        if self.nonlinearity == "tanh":
            out = ops.tanh(out)
        elif self.nonlinearity == "relu":
            out = ops.relu(out)
        else:
            raise ValueError("unsupported nonlinearity function. Only support ReLU and Tanh.")
        
        return out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.device = device
        self.dtype = dtype
        self.bias = bias

        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)] + \
                         [RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size= X.shape[1]
        if h0 is None:
            h0 = [init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        else:
            h0 = list(ops.split(h0, axis=0))

        h_n = []
        inputs = list(ops.split(X, axis=0))
        for i, layer in enumerate(self.rnn_cells):
            for t, input in enumerate(inputs):
                h0[i] = layer(input, h0[i])
                inputs[t] = h0[i]
            h_n.append(h0[i])
        
        return ops.stack(inputs, axis=0), ops.stack(h_n, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = np.sqrt(1.0 / hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size * 4, 
                                        low=-bound, high=bound, 
                                        device=device, dtype=dtype,requires_grad=True))
        
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size * 4, 
                                        low=-bound, high=bound, 
                                        device=device, dtype=dtype,requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size * 4, 
                                            low=-bound, high=bound, 
                                            device=device, dtype=dtype,requires_grad=True))
            
            self.bias_hh = Parameter(init.rand(hidden_size * 4, 
                                            low=-bound, high=bound, 
                                            device=device, dtype=dtype,requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.device = device
        self.dtype = dtype
        self.bias = bias

        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        if h is None:
            h0 = init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        
        if self.bias:
            bias_ih = self.bias_ih.reshape((1, self.hidden_size * 4)) \
                                  .broadcast_to((batch_size, self.hidden_size * 4))
            bias_hh = self.bias_hh.reshape((1, self.hidden_size * 4)) \
                                  .broadcast_to((batch_size, self.hidden_size * 4))
            out = X @ self.W_ih + bias_ih + h0 @ self.W_hh + bias_hh
        else:
            out = X @ self.W_ih + h0 @ self.W_hh

        out_all_split = list(ops.split(out, axis = 1))
        outs = []
        for i in range(4):
            outs.append(ops.stack(out_all_split[i * self.hidden_size : (i + 1) * self.hidden_size], axis = 1))

        i = self.sigmoid(outs[0])
        f = self.sigmoid(outs[1])
        g = ops.tanh(outs[2])
        o = self.sigmoid(outs[3])

        c_ = f * c0 + i * g
        h_ = o * ops.tanh(c_)

        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.device = device
        self.dtype = dtype
        self.bias = bias

        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)] + \
                         [LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size= X.shape[1]
        if h is None:
            h0s = [init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
            c0s = [init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        else:
            h0, c0 = h
            h0s = list(ops.split(h0, axis=0))
            c0s = list(ops.split(c0, axis=0))

        h_n = []
        c_n = []
        inputs = list(ops.split(X, axis=0))
        for layer, h0, c0 in zip(self.lstm_cells, h0s, c0s):
            for t, input in enumerate(inputs):
                h0, c0 = layer(input, (h0, c0))
                inputs[t] = h0
            h_n.append(h0)
            c_n.append(c0)
        
        return ops.stack(inputs, axis=0), (ops.stack(h_n, axis=0), ops.stack(c_n, axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size = x.shape
        one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype) # (seq_len, bs, num_embeddings)
        one_hot_reshape = one_hot.reshape((seq_len * batch_size, self.num_embeddings))
        output = (one_hot_reshape @ self.weight).reshape((seq_len, batch_size, self.embedding_dim))
        return output
        ### END YOUR SOLUTION