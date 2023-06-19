import math


def computeBatchSize(dataset1_sizes, dataset2_sizes, batch_sizeP):

    batchSize1 = {}
    batchSize2 = {}

    for mode in ['train', 'val', 'test']:

        # train
        if dataset1_sizes[mode] > dataset2_sizes[mode]:
            batchSize1[mode] = batch_sizeP
            ratioSizeDBs = dataset1_sizes[mode] / dataset2_sizes[mode]
            batchSize2[mode] = math.ceil(batchSize1[mode] / ratioSizeDBs)  # less batches in db2 (we use all db2)
        else:
            if dataset2_sizes[mode] > dataset1_sizes[mode]:
                batchSize2[mode] = batch_sizeP
                ratioSizeDBs = dataset2_sizes[mode] / dataset1_sizes[mode]
                batchSize1[mode] = math.floor(batchSize2[mode] / ratioSizeDBs)  # more batches in db1 (we use all db2)
            else:
                batchSize1[mode] = batch_sizeP
                batchSize2[mode] = batch_sizeP

    return batchSize1, batchSize2