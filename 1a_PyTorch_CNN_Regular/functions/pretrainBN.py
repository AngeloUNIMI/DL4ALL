import torch


def pretrainBN(model, dataloader, cuda, numBatchesPretrain):

    model.train()
    dataiter = iter(dataloader)
    batch_num = -1

    while batch_num < numBatchesPretrain:

        try:
            inputs, label = dataiter.next()
            batch_num = batch_num + 1
        except StopIteration:
            break

        # cuda
        if cuda:
            inputs = inputs.to('cuda')

        # forward pass with no error
        with torch.set_grad_enabled(False):
            dummy, outputs = model(inputs)

    return model

def pretrainBN_singleBatch(model, inputs, cuda):

    model.train()

    # cuda
    if cuda:
        inputs = inputs.to('cuda')

        # forward pass with no error
        with torch.set_grad_enabled(False):
            dummy, outputs = model(inputs)

    return model