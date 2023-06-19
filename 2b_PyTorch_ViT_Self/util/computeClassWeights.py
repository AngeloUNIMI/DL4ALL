import os
import pickle
import torch
from util import getClassCount1, getClassCount2


def computeClassWeights(dataset1_sizes, dataset2_sizes,
                        dataloader1_train, dataloader1_val, numClasses1,
                        dataloader2_train, dataloader2_val, numClasses2,
                        cuda, dirResults):

    fileNameSaveWeights = os.path.join(dirResults, 'weights.dat')
    if os.path.isfile(fileNameSaveWeights):
        # read
        fileSaveWeights = open(fileNameSaveWeights, 'rb')
        weightsBCE1, weightsBCE2 = pickle.load(fileSaveWeights)
        fileSaveWeights.close()
    else:
        # adp
        datasetSizeAll1 = dataset1_sizes['train'] + dataset1_sizes['val']
        classCountAll1 = getClassCount2(dataloader1_train, numClasses1) + getClassCount2(dataloader1_val, numClasses1)
        # if no count, put 1
        numSub = 10
        for listc, tt in enumerate(classCountAll1):
            if tt < numSub:
                classCountAll1[listc] = numSub
        weightsBCE1 = torch.FloatTensor(datasetSizeAll1 / classCountAll1)
        # cnmc
        datasetSizeAll2 = dataset2_sizes['train'] + dataset2_sizes['val']
        classCountAll2 = getClassCount2(dataloader2_train, numClasses2) + getClassCount2(dataloader2_val, numClasses2)
        # if no count, put 1
        numSub = 10
        for listc, tt in enumerate(classCountAll2):
            if tt < numSub:
                classCountAll2[listc] = numSub
        weightsBCE2 = torch.FloatTensor(datasetSizeAll2 / classCountAll2)
        # save
        fileSaveWeights = open(fileNameSaveWeights, 'wb')
        pickle.dump([weightsBCE1, weightsBCE2], fileSaveWeights)
        fileSaveWeights.close()
    # cuda
    if cuda:
        weightsBCE1 = weightsBCE1.to('cuda')
        weightsBCE2 = weightsBCE2.to('cuda')

    return weightsBCE1, weightsBCE2
