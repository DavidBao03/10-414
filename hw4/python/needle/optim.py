"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad  = self.u.get(param, 0) * self.momentum + (1-self.momentum) * (param.grad.data + self.weight_decay * param.data)
            grad = ndl.Tensor(grad, dtype = param.dtype)
            self.u[param] = grad
            param.data =  param.data - self.lr * grad.data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data 
            grad1 = self.m.get(param, 0) * self.beta1 + (1 - self.beta1) * grad.data
            grad1 = ndl.Tensor(grad1, dtype = param.dtype)
            self.m[param] = grad1

            grad2 = self.v.get(param, 0) * self.beta2 + (1 - self.beta2) * (grad.data ** 2)
            self.v[param] = grad2

            ut = grad1.data / (1 - self.beta1 ** self.t)
            vt = grad2.data / (1 - self.beta2 ** self.t)

            out = param.data - self.lr * ut / (vt ** 0.5 + self.eps)
            out = ndl.Tensor(out, dtype=param.dtype)
            param.data = out
        ### END YOUR SOLUTION
