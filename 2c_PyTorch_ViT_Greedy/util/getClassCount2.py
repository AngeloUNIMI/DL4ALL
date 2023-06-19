import numpy as np
from util import pause

def getClassCount2(trainLoader):

    classCounts = np.zeros(2)

    for _, target in trainLoader:
        classCounts += np.bincount(target)

    return classCounts