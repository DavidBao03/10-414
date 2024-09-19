import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    module1 =  nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    
    module2 = nn.Sequential(
        nn.Residual(module1),
        nn.ReLU()
    )

    return module2
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    module1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
    )

    module2 = nn.Sequential(*[ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)])

    module3 = nn.Linear(hidden_dim, num_classes)

    return nn.Sequential(
        module1,
        module2,
        module3
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()

    loss_fuc = nn.SoftmaxLoss()
    acc_num = 0
    losses = []
    for X, y in dataloader:
        out = model(X)
        loss = loss_fuc(out, y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
            
        losses.append(loss.numpy())
        acc_num += (out.numpy().argmax(axis=1) == y.numpy()).sum()

    
    return 1 - acc_num / len(dataloader.dataset), np.mean(losses)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    )

    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = ndl.data.MNISTDataset(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )

    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, shuffle=False)


    shape = test_dataloader.dataset.img.shape
    dim = shape[1] * shape[2]

    model = MLPResNet(dim, hidden_dim)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_err,train_loss = 0, 0
    test_err,test_loss = 0, 0

    for i in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        print("Epoch %d: Train err: %f, Train loss: %f" % (
            i + 1, train_err, train_loss
        ))

    test_err, test_loss = epoch(test_dataloader, model)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
