import torch
import torch.nn as nn
from util.computeClassWeights import computeClassWeights


# training with validation
def warmup(dataloader1_train, dataloader1_val, dataloader2_train, dataloader2_val,
                    model, optimizer,
                    dataset1_sizes, dataset2_sizes,
                    classVec, num_classes,
                    cuda, dirResults):

    # class weights computed on train and val together
    print('\tGetting class weights...')
    weightsBCE1, weightsBCE2 = computeClassWeights(dataset1_sizes, dataset2_sizes,
                                                   dataloader2_train, dataloader2_val,
                                                   classVec, cuda, dirResults)

    # loop on epochs
    for dbnum in range(0, 2):

        # Each epoch has a training and validation phase
        # ---------------------------------------------------TRAINING
        model.train()  # Set model to training mode

        # choose dataloader
        if dbnum == 0:
            dataloaders_train = dataloader1_train
        else:
            dataloaders_train = dataloader2_train

        # Iterate over data.
        dataiter = iter(dataloaders_train)
        batch_num1 = -1
        # 1 epoch
        while True:

            if dbnum == 0:
                try:
                    inputs, dummyTargets, filename, label = dataiter.next()
                except StopIteration:
                    break
            else:
                try:
                    inputs, label = dataiter.next()
                except StopIteration:
                    break

            # cuda
            if cuda:
                inputs = inputs.to('cuda')
                label = label.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                if dbnum == 0:
                    outputs, dummy = model(inputs)
                    if cuda:
                        outputs = outputs.to('cuda')
                    m = nn.Sigmoid()
                    preds = (m(outputs) > 0.5).int()
                    # Forward Data and Calculate Loss#
                    criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weightsBCE1).cuda()
                    loss = criterion(outputs.float(), label.float())
                else:
                    dummy, outputs = model(inputs)
                    if cuda:
                        outputs = outputs.to('cuda')
                    _, preds = torch.max(outputs, 1)
                    # Forward Data and Calculate Loss#
                    criterion = nn.CrossEntropyLoss(weight=weightsBCE2).cuda()
                    loss = criterion(outputs, label)

                # Back Propagation and Update #
                loss.backward()
                optimizer.step()

    # del
    del inputs, label

    return model
