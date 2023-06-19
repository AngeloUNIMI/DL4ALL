import numpy as np
from util import pause

def getClassCount2(trainLoader, numClasses):

    classCounts = np.zeros(numClasses)

    for _, target in trainLoader:
        classCounts += np.bincount(target, minlength=numClasses)

    return classCounts