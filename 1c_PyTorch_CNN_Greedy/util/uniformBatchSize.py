def uniformBatchSize(batch_sizeP):

    batchSize1 = {}
    batchSize2 = {}

    for mode in ['train', 'val', 'test']:

        batchSize1[mode] = batch_sizeP
        batchSize2[mode] = batch_sizeP

    return batchSize1, batchSize2
