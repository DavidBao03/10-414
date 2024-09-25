"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(kernel_size * kernel_size * in_channels, kernel_size * kernel_size * out_channels, 
                                        (kernel_size, kernel_size, in_channels, out_channels)), 
                                       device=device, dtype=dtype, requires_grad=True)
        if bias:
            bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(init.rand(out_channels, low=-bound, high=bound),
                                  device=device, dtype=dtype, requires_grad=True)
            
        self.padding = (kernel_size - 1) // 2
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = x.transpose((1,3)).transpose((1,2))
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)

        if self.bias:
            out += ops.broadcast_to(self.bias.reshape((1, 1, 1, self.out_channels)), out.shape)
        
        return out.transpose((1,2)).transpose((1,3))
        ### END YOUR SOLUTION