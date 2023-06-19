import torch
import time
import copy
import numpy as np
import os
import pickle
import torch.nn as nn
from util.computeClassWeights import computeClassWeights
from util import print_pers
from functions.pretrainBN import pretrainBN
from functions.pretrainBN import pretrainBN_singleBatch

# training with validation
def train_model_val(dataloader1_train, dataloader1_val, dataloader2_train, dataloader2_val,
                    data2_train, data2_val,
                    model, optimizer, scheduler,
                    dataset1_sizes, dataset2_sizes, batchSize1, batchSize2, classVec, num_classes,
                    num_epochs, numBatchesPretrain, cuda, log, dirResults, fileResultNameFull):

    # check if final already exists
    fileNameSaveFinal = 'modelsave_final.pt'
    if os.path.isfile(os.path.join(dirResults, fileNameSaveFinal)):
        # display
        if log:
            print_pers('\tModel loaded', fileResultNameFull)
        model.load_state_dict(torch.load(os.path.join(dirResults, fileNameSaveFinal)))
        return model

    # check if partial results
    entries = os.listdir(dirResults)
    max_epoch = -1
    for entry in entries:
        if entry.endswith(".pt"):
            entry2 = os.path.splitext(entry)
            temp = entry2[0].split('_')
            entry3 = temp[-1]
            saved_epoch = int(entry3)
            if saved_epoch > max_epoch:
                max_epoch = saved_epoch

    if max_epoch > -1:
        fileNameSave = 'modelsave_epoch_{0}.pt'.format(max_epoch)
        model.load_state_dict(torch.load(os.path.join(dirResults, fileNameSave)))

    # init best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # class weights computed on train and val together
    print('\tGetting class weights...')
    weightsBCE1, weightsBCE2 = computeClassWeights(dataset1_sizes, dataset2_sizes,
                                                   dataloader2_train, dataloader2_val,
                                                   classVec, cuda, dirResults)

    # compute num batches
    numBatches1 = {}
    numBatches1['train'] = np.round(dataset1_sizes['train'] / batchSize1['train'])
    numBatches1['val'] = np.round(dataset1_sizes['val'] / batchSize1['val'])

    # loop on epochs
    for epoch in range(num_epochs):

        # continue from saved epoch
        if epoch <= max_epoch:
            continue

        # display
        if log:
            print_pers('\tEpoch {}/{}'.format(epoch+1, num_epochs), fileResultNameFull)

        # Each epoch has a training and validation phase
        # ---------------------------------------------------TRAINING
        model.train()  # Set model to training mode

        # init losses and corrects
        # loss_all = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        # running_loss_all = 0.0
        running_corrects1 = 0.0
        running_corrects2 = 0.0
        amountOfDB1 = 0
        amountOfDB2 = 0

        # Iterate over data.
        dataiter1 = iter(dataloader1_train)
        dataiter2 = iter(dataloader2_train)
        batch_num1 = -1
        while True:
        # for batch_num, (inputs1, dummyTargets, filename, label1) in enumerate(dataloaders1_chosen):

            try:
                inputs1, dummyTargets, filename, label1 = dataiter1.next()
                batch_num1 = batch_num1 + 1
            except StopIteration:
                break

            try:
                inputs2, label2 = dataiter2.next()
            except StopIteration:
                break

            # display
            # if batch_num1 % 200 == 0:
                # print_pers("\t\tBatch n. {0} / {1}".format(batch_num1, int(numBatches1[phase])), fileResultNameFull)

            ##################
            #if batch_num > 10:
                #break
            ##################

            # get size of current batch
            # if different, one of them is about to finish (no longer true)
            sizeCurrentBatch1 = inputs1.size(0)
            sizeCurrentBatch2 = inputs2.size(0)
            # if sizeCurrentBatch1 != sizeCurrentBatch2:
                # break

            # we need to keep track of how much db we are processing
            amountOfDB1 += sizeCurrentBatch1
            amountOfDB2 += sizeCurrentBatch2

            # cuda
            if cuda:
                inputs1 = inputs1.to('cuda')
                inputs2 = inputs2.to('cuda')
                label1 = label1.to('cuda')
                label2 = label2.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # pretrain for ADP: forward few batches to update batch normalization parameters
            # model = pretrainBN(model, dataloader1_train, cuda, numBatchesPretrain)

            # forward 1
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs1, dummy = model(inputs1)
                if cuda:
                    outputs1 = outputs1.to('cuda')

                m = nn.Sigmoid()
                preds1 = (m(outputs1) > 0.5).int()

                # Forward Data and Calculate Loss#
                criterion1 = torch.nn.MultiLabelSoftMarginLoss(weight=weightsBCE1).cuda()
                loss1 = criterion1(outputs1.float(), label1.float())

                # Back Propagation and Update #
                # loss1 = loss1 / 2  # weight loss by 0.5
                # loss1.backward()
                # optimizer.step()

            # Initialize Optimizer #
            # optimizer.zero_grad()

            # pretrain for CNMC: forward few batches to update batch normalization parameters
            model = pretrainBN_singleBatch(model, inputs2, cuda)

            # forward 2
            # track history if only in train
            with torch.set_grad_enabled(True):
                dummy, outputs2 = model(inputs2)
                if cuda:
                    outputs2 = outputs2.to('cuda')

                # softmax
                _, preds2 = torch.max(outputs2, 1)

                # Forward Data and Calculate Loss#
                criterion2 = nn.CrossEntropyLoss(weight=weightsBCE2).cuda()
                # criterion2 = nn.CrossEntropyLoss().cuda()
                loss2 = criterion2(outputs2, label2)

                # Back Propagation and Update #
                # loss2 = loss2 / 2  # weight loss by 0.5
                # loss2.backward()
                loss_all = 0.5 * loss1 + 0.5 * loss2
                loss_all.backward()

                # rescale gradients
                for p in model.fc1.parameters():
                    p.grad *= (batchSize1['train']+batchSize2['train']) / batchSize1['train']
                for p in model.fc2.parameters():
                    p.grad *= (batchSize1['train']+batchSize2['train']) / batchSize2['train']

                optimizer.step()

            # Record Data #
            # accuracy evaluated on db 2
            with torch.no_grad():
                running_loss1 += loss1.item() * sizeCurrentBatch1
                running_loss2 += loss2.item() * sizeCurrentBatch2
                # running_loss_all += loss_all.item() * sizeCurrentBatch1 * sizeCurrentBatch2
                running_corrects1 += torch.sum(preds1 == label1.data.int())
                running_corrects2 += torch.sum(preds2 == label2.data.int())

        # update schedulers
        if len(scheduler) > 0:
            for schedulerSingle in scheduler:
                schedulerSingle.step()

        # compute epochs losses
        with torch.no_grad():
            epoch_loss1 = running_loss1 / amountOfDB1
            epoch_loss2 = running_loss2 / amountOfDB2
            # epoch_loss_all = running_loss_all / (dataset1_sizes[phase] * dataset2_sizes[phase])
            # epoch_acc = running_corrects / (dataset2_sizes[phase] * (num_classes + 2))
            epoch_acc1 = running_corrects1 / (amountOfDB1 * num_classes)
            epoch_acc2 = running_corrects2 / amountOfDB2

        # display
        if log:
            print_pers('\t\t{} Loss1: {:.4f} Loss2: {:.4f} Acc1 (1-HD): {:.4f} Acc2 (1-HD): {:.4f}'.format(
                'Train', epoch_loss1, epoch_loss2, epoch_acc1, epoch_acc2),
                fileResultNameFull)
            # util.print_pers('\t\t{} Loss (all): {:.4f} Acc1 (1-HD): {:.4f} Acc2 (1-HD): {:.4f}'.format(
                # phase, epoch_loss_all, epoch_acc1, epoch_acc2),
                # fileResultNameFull)

        # ---------------------------------------------------VALIDATION

        # pretrain for CNMC: forward few batches to update batch normalization parameters
        model = pretrainBN(model, dataloader2_train, cuda, numBatchesPretrain)

        # ---------------------------------
        # validation
        model.eval()  # Set model to evaluation mode
        # reset iterator
        dataiter = iter(dataloader2_val)
        # init
        amountOfDB = 0
        running_loss = 0
        running_corrects = 0
        while True:

            try:
                inputs, label = dataiter.next()
            except StopIteration:
                break

            # get size of current batch
            sizeCurrentBatch = inputs.size(0)

            # we need to keep track of how much db we are processing
            amountOfDB += sizeCurrentBatch

            if cuda:
                inputs = inputs.to('cuda')
                label = label.to('cuda')

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                dummy, outputs = model(inputs)
                if cuda:
                    outputs = outputs.to('cuda')
                _, preds = torch.max(outputs, 1)
                # Forward Data and Calculate Loss#
                criterion = nn.CrossEntropyLoss(weight=weightsBCE2).cuda()
                # criterion = nn.CrossEntropyLoss().cuda()
                loss = criterion(outputs, label)

            # Record Data #
            # accuracy evaluated on db 2
            with torch.no_grad():
                running_loss += loss.item() * sizeCurrentBatch
                running_corrects += torch.sum(preds == label.data.int())

        # compute epochs losses
        with torch.no_grad():
            epoch_loss = running_loss / amountOfDB
            epoch_acc = running_corrects / amountOfDB

        # display
        if log:
            print_pers('\t\t{} Loss: {:.4f} Acc (1-HD): {:.4f} '.format(
                'Val', epoch_loss, epoch_acc),
                fileResultNameFull)

        # if greater val accuracy, deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print('', end='')

        # del
        del inputs1, inputs2, inputs, label1, label2, label
        torch.cuda.empty_cache()

    print_pers('\tBest val Acc: {:4f}'.format(best_acc), fileResultNameFull)

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save final
    torch.save(model.state_dict(), os.path.join(dirResults, fileNameSaveFinal))

    # del
    torch.cuda.empty_cache()

    return model
