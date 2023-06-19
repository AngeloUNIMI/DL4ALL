# --------------------------
# IMPORT
# from torchvision import models
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import warnings
# warnings.filterwarnings("ignore")
import os
import shutil
from sklearn.metrics import confusion_matrix
import splitfolders
import random
from datetime import datetime
import pickle
import PIL
from pytorch_pretrained_vit import ViT
# --------------------------
# PRIV FUNCTIONS
import util
import functions
# --------------------------
# CLASSES
# rom classes.classesADP import classesADP
# --------------------------
# PARAMS
from params.dirs import dirs
# --------------------------
# MODELS
from models.vitGeno import vitGeno
from utils import init_weights_normal, init_weights_xavier, init_weights_kaiming


def count_parameters(model):
    all_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, train_params


# --------------------------
# MAIN
if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    # params
    plotta = False
    log = True
    extOrig = 'tif'
    extNew = 'png'
    num_iterations = 1
    nFolds = 10
    # nFolds = 10
    batch_sizeP = 8  # 32
    batch_sizeP_norm = 1024
    numWorkersP = 0
    n_neighborsP = 1
    fontSize = 22
    padSize = 30
    num_epochs = 100  # 100
    numBatchesPretrain = 100
    seed = 42

    # --------------
    evalMode = False
    # --------------


    # ------------------------------------------------------------------- db info 1
    dirWorkspace1Orig = 'D:/Workspace/DB HTT - Public (orig) (Mahdi)/'
    dirWorkspace1Test = 'D:/Workspace/DB HTT - Public (test) (2) (Mahdi)/'
    dbName1 = 'ADP'
    ROI1 = 'img_res_1um_bicubic'
    csvFile = 'ADP_EncodedLabels_Release1_Flat.csv'
    imageSize = 224
    dLabel = 'dummyLabel'
    dataset1_sizes = {}
    #
    dirDb1Orig = dirWorkspace1Orig + dbName1 + '/' + ROI1 + '/'
    dirDb1Test = dirWorkspace1Test + dbName1 + '_' + ROI1 + '/datastore/'
    dirOutTrainTest1 = dirWorkspace1Test + dbName1 + '_' + ROI1 + '/datastore_train_test/'
    csvFileFull = dirWorkspace1Orig + dbName1 + '/' + csvFile
    os.makedirs(dirDb1Test, exist_ok=True)


    # ------------------------------------------------------------------- db info 2
    dirWorkspace2 = 'D:/Workspace/DB HEM - Public (test)/'
    dbName2 = 'C-NMC_Leukemia'
    dirDbOrig2 = dirWorkspace2 + dbName2 + '/'
    dirDbTest2 = dirWorkspace2 + dbName2 + '/datastore/'
    dirOutTrainTest2 = dirWorkspace2 + dbName2 + '/datastore_train_test/'
    if not os.path.exists(dirDbTest2):
        os.makedirs(dirDbTest2)
    if not os.path.exists(dirOutTrainTest2):
        os.makedirs(dirOutTrainTest2)
    dataset2_sizes = {}


    # ------------------------------------------------------------------- Enable CUDA
    cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.cuda.DoubleTensor if cuda else torch.cuda.DoubleTensor
    if cuda:
        torch.cuda.empty_cache()
        device = 'cuda'
    else:
        device = 'cpu'
    print("Cuda is {0}".format(cuda))
    #util.pause()

    # For Reproducibility #
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    # ------------------------------------------------------------------- define all models we want to try
    modelNamesAll = list()
    modelNamesAll.append({'name': 'vit',
                          'init': 'he',
                          'in_channels': 3,
                          'embed_dim': 768,
                          'patch_size': 16,
                          'num_layers': 12,
                          'num_heads': 12,
                          'mlp_dim': 3072,
                          'drop_out': 0.1
                          })

    # - ADP or IMAGENet
    trainModes = []
    trainModes.append('imagenet')
    trainModes.append('scratch')
    for trainMode in trainModes:

        # display
        if log:
            print()
            print("Train mode: {0}".format(trainMode))

        # get all classes
        allLabelsStr, allLabelsInt, columnNames, numSamples = util.getAllClassesVec(csvFileFull, dirDb1Orig, dirDb1Test, log, writeFile=False)
        num_classes = len(allLabelsInt)


        # ------------------------------------------------------------------- loop on models
        for i, (modelData) in enumerate(modelNamesAll):

            # dir results
            dirResult = './results/' + trainMode + '/' + modelData['name'] + '/'
            if not os.path.exists(dirResult):
                os.makedirs(dirResult)

            # result file
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            fileResultName = current_time + '.txt'
            fileResultNameFull = os.path.join(dirResult, fileResultName)
            fileResult = open(fileResultNameFull, "x")
            fileResult.close()

            # display
            if log:
                print()
                util.print_pers("Model: {0}".format(modelData['name']), fileResultNameFull)


            # ------------------------------------------------------------------- loop on iterations
            # init
            dataset_sizes = {}
            accuracyALL = np.zeros(num_iterations)
            CM_all = np.zeros((2, 2))
            CM_perc_all = np.zeros((2, 2))
            for r in range(0, num_iterations): #(nFolds-1)

                # display
                if log:
                    util.print_pers("", fileResultNameFull)
                    util.print_pers("Iteration n. {0}".format(r + 1), fileResultNameFull)
                    util.print_pers("", fileResultNameFull)

                if modelData['name'] == 'vit':

                    if trainMode == 'imagenet':
                        currentModel = vitGeno('B_16_imagenet1k', pretrained=True)
                        # fine tune for imagenet pretrained
                        for param in currentModel.parameters():
                            param.requires_grad = False  # frozen for warm-up

                    if trainMode == 'scratch':
                        currentModel = vitGeno('B_16_imagenet1k', pretrained=False)
                        # fine tune for imagenet pretrained
                        for param in currentModel.parameters():
                            param.requires_grad = False  # frozen for warm-up

                    # change last layer
                    # Multi-task learning
                    # fc1
                    new_classifier1 = nn.Linear(modelData['embed_dim'], num_classes)
                    currentModel.fc1 = new_classifier1
                    new_classifier2 = nn.Linear(modelData['embed_dim'], 2)
                    currentModel.fc2 = new_classifier2

                    image_size = currentModel.image_size

                    # Weight Initialization #
                    if trainMode == 'scratch':
                        if modelData['init'] == 'normal':
                            currentModel.apply(init_weights_normal)
                        elif modelData['init'] == 'xavier':
                            currentModel.apply(init_weights_xavier)
                        elif modelData['init'] == 'he':
                            currentModel.apply(init_weights_kaiming)
                        else:
                            raise NotImplementedError

                    all_params, train_params = count_parameters(currentModel)
                    print(all_params, train_params)
                    print()

                #currentModel.double()
                # cuda
                if cuda:
                    currentModel.to('cuda')
                    if torch.cuda.device_count() > 1:
                        currentModel = nn.DataParallel(currentModel, device_ids=[0, 1])
                # log
                if log:
                    print(currentModel)
                    print()

                # ---------------------- split db
                if not evalMode:
                    print("Random splitting DBs...")

                    # random split target db1
                    # first delete dir
                    if os.path.exists(dirOutTrainTest1):
                        shutil.rmtree(dirOutTrainTest1)
                    # create
                    if not os.path.exists(dirOutTrainTest1):
                        os.makedirs(dirOutTrainTest1)
                    # split db
                    splitfolders.ratio(dirDb1Test, output=dirOutTrainTest1, seed=seed, ratio=(.7, .2, .1))

                    # random split target db2
                    # first delete dir
                    if os.path.exists(dirOutTrainTest2):
                        shutil.rmtree(dirOutTrainTest2)
                    # create
                    if not os.path.exists(dirOutTrainTest2):
                        os.makedirs(dirOutTrainTest2)
                    # split db
                    splitfolders.ratio(dirDbTest2, output=dirOutTrainTest2, seed=seed, ratio=(.7, .2, .1))
                    util.print_pers("", fileResultNameFull)
                # ---------------------- split db

                # normalization
                print("Normalization...")
                # transforms
                # adp
                transform_adp = {}
                transform_adp['train'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                        # mean=[meanNorm, meanNorm, meanNorm],
                        # std=[stdNorm, stdNorm, stdNorm]),
                ])
                transform_adp['val'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                        # mean=[meanNorm, meanNorm, meanNorm],
                        # std=[stdNorm, stdNorm, stdNorm]),
                ])
                # cnmc
                transform_cnmc = {}
                transform_cnmc['train'] = transforms.Compose([
                    transforms.CenterCrop(450),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                        # mean=[meanNorm, meanNorm, meanNorm],
                        # std=[stdNorm, stdNorm, stdNorm]),
                ])
                transform_cnmc['val'] = transforms.Compose([
                    transforms.CenterCrop(450),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                        # mean=[meanNorm, meanNorm, meanNorm],
                        # std=[stdNorm, stdNorm, stdNorm]),
                ])
                #data loaders
                # adp
                adp_train = datasets.ImageFolder(os.path.join(dirOutTrainTest1, 'train'),
                                                      transform=transform_adp['train'])
                adp_train_loader = torch.utils.data.DataLoader(adp_train,
                                                               batch_size=batch_sizeP_norm, shuffle=True,
                                                               num_workers=numWorkersP, pin_memory=True,
                                                               drop_last=True)
                dataset1_sizes['train'] = len(adp_train)
                adp_val = datasets.ImageFolder(os.path.join(dirOutTrainTest1, 'val'),
                                                      transform=transform_adp['val'])
                adp_val_loader = torch.utils.data.DataLoader(adp_val,
                                                               batch_size=batch_sizeP_norm, shuffle=True,
                                                               num_workers=numWorkersP, pin_memory=True,
                                                               drop_last=True)
                dataset1_sizes['val'] = len(adp_val)
                # cnmc
                cnmc_train = datasets.ImageFolder(os.path.join(dirOutTrainTest2, 'train'),
                                                  transform=transform_cnmc['train'])
                cnmc_train_loader = torch.utils.data.DataLoader(cnmc_train,
                                                                batch_size=batch_sizeP_norm, shuffle=True,
                                                                num_workers=numWorkersP, pin_memory=True,
                                                                drop_last=True)
                dataset2_sizes['train'] = len(cnmc_train)
                cnmc_val = datasets.ImageFolder(os.path.join(dirOutTrainTest2, 'val'),
                                                  transform=transform_cnmc['val'])
                cnmc_val_loader = torch.utils.data.DataLoader(cnmc_val,
                                                                batch_size=batch_sizeP_norm, shuffle=True,
                                                                num_workers=numWorkersP, pin_memory=True,
                                                                drop_last=True)
                dataset2_sizes['val'] = len(cnmc_val)

                # adp
                fileNameSaveNorm1 = {}
                fileSaveNorm1 = {}
                meanNorm1 = {}
                stdNorm1 = {}
                dataloaders_all = list()
                dataloaders_all.append(adp_train_loader)
                dataloaders_all.append(adp_val_loader)
                dataset_sizes_all = dataset1_sizes['train'] + dataset1_sizes['val']
                fileNameSaveNorm1 = os.path.join(dirResult, 'norm1.dat')
                # if file exist, load
                if os.path.isfile(fileNameSaveNorm1):
                    # read
                    fileSaveNorm1 = open(fileNameSaveNorm1, 'rb')
                    meanNorm1, stdNorm1 = pickle.load(fileSaveNorm1)
                    fileSaveNorm1.close()
                # else, compute normalization
                else:
                    # compute norm for all channels together
                    meanNorm1, stdNorm1 = util.computeMeanStd(dataloaders_all, dataset_sizes_all, batch_sizeP_norm, cuda)
                    # save
                    fileSaveNorm1 = open(fileNameSaveNorm1, 'wb')
                    pickle.dump([meanNorm1, stdNorm1], fileSaveNorm1)
                    fileSaveNorm1.close()

                # cnmc
                fileNameSaveNorm2 = {}
                fileSaveNorm2 = {}
                meanNorm2 = {}
                stdNorm2 = {}
                dataloaders_all = list()
                dataloaders_all.append(cnmc_train_loader)
                dataloaders_all.append(cnmc_val_loader)
                dataset_sizes_all = dataset2_sizes['train'] + dataset2_sizes['val']
                fileNameSaveNorm2 = os.path.join(dirResult, 'norm2.dat')
                # if file exist, load
                if os.path.isfile(fileNameSaveNorm2):
                    # read
                    fileSaveNorm2 = open(fileNameSaveNorm2, 'rb')
                    meanNorm2, stdNorm2 = pickle.load(fileSaveNorm2)
                    fileSaveNorm2.close()
                # else, compute normalization
                else:
                    # compute norm for all channels together
                    meanNorm2, stdNorm2 = util.computeMeanStd(dataloaders_all, dataset_sizes_all, batch_sizeP_norm, cuda)
                    # save
                    fileSaveNorm2 = open(fileNameSaveNorm2, 'wb')
                    pickle.dump([meanNorm2, stdNorm2], fileSaveNorm2)
                    fileSaveNorm2.close()

                # update transforms
                # adp train
                transform_adp['train'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm1, meanNorm1, meanNorm1],
                        std=[stdNorm1, stdNorm1, stdNorm1]),
                ])
                # val
                transform_adp['val'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm1, meanNorm1, meanNorm1],
                        std=[stdNorm1, stdNorm1, stdNorm1]),
                ])
                # test
                transform_adp['test'] = transforms.Compose([
                    transforms.CenterCrop(272),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm1, meanNorm1, meanNorm1],
                        std=[stdNorm1, stdNorm1, stdNorm1]),
                ])
                # cnmc train
                transform_cnmc['train'] = transforms.Compose([
                    transforms.CenterCrop(450),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # ---------------------------
                    transforms.RandomRotation(90),  # ?
                    # ---------------------------
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm2, meanNorm2, meanNorm2],
                        std=[stdNorm2, stdNorm2, stdNorm2]),
                ])
                # val
                transform_cnmc['val'] = transforms.Compose([
                    transforms.CenterCrop(450),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm2, meanNorm2, meanNorm2],
                        std=[stdNorm2, stdNorm2, stdNorm2]),
                ])
                # test
                transform_cnmc['test'] = transforms.Compose([
                    transforms.CenterCrop(450),
                    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                    # transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[meanNorm2, meanNorm2, meanNorm2],
                        std=[stdNorm2, stdNorm2, stdNorm2]),
                ])
                print()

                # update data loaders
                # -----------------------------------------------------------------------------------------
                # ADP
                # train
                adp_train = datasets.ImageFolder(os.path.join(dirOutTrainTest1, 'train'),
                                                           transform=transform_adp['train'])
                dataset1_sizes['train'] = len(adp_train)
                util.print_pers("(ADP) Dimensione dataset train: {0}".format(dataset1_sizes['train']), fileResultNameFull)
                # val
                adp_val = datasets.ImageFolder(os.path.join(dirOutTrainTest1, 'val'),
                                                         transform=transform_adp['val'])
                dataset1_sizes['val'] = len(adp_val)
                util.print_pers("(ADP) Dimensione dataset val: {0}".format(dataset1_sizes['val']), fileResultNameFull)
                # test
                adp_test = datasets.ImageFolder(os.path.join(dirOutTrainTest1, 'test'),
                                                         transform=transform_adp['test'])
                dataset1_sizes['test'] = len(adp_test)
                util.print_pers("(ADP) Dimensione dataset test: {0}".format(dataset1_sizes['test']), fileResultNameFull)
                util.print_pers('', fileResultNameFull)
                # -----------------------------------------------------------------------------------------
                # CNMC
                # train
                cnmc_train = datasets.ImageFolder(os.path.join(dirOutTrainTest2, 'train'),
                                                           transform=transform_cnmc['train'])
                dataset2_sizes['train'] = len(cnmc_train)
                util.print_pers("(CNMC) Dimensione dataset train: {0}".format(dataset2_sizes['train']), fileResultNameFull)
                # val
                cnmc_val = datasets.ImageFolder(os.path.join(dirOutTrainTest2, 'val'),
                                                         transform=transform_cnmc['val'])
                dataset2_sizes['val'] = len(cnmc_val)
                util.print_pers("(CNMC) Dimensione dataset val: {0}".format(dataset2_sizes['val']), fileResultNameFull)
                # test
                cnmc_test = datasets.ImageFolder(os.path.join(dirOutTrainTest2, 'test'),
                                                         transform=transform_cnmc['test'])
                dataset2_sizes['test'] = len(cnmc_test)
                util.print_pers("(CNMC) Dimensione dataset test: {0}".format(dataset2_sizes['test']), fileResultNameFull)
                util.print_pers('', fileResultNameFull)

                # compute batch sizes
                batchSize1, batchSize2 = util.computeBatchSize(dataset1_sizes, dataset2_sizes, batch_sizeP)
                # batchSize1, batchSize2 = util.uniformBatchSize(batch_sizeP)
                for mode in ['train', 'val', 'test']:
                    # -------------------------------------
                    # Less data for ADP
                    # batchSize1[mode] = int(batchSize1[mode] / 2)
                    # -------------------------------------
                    util.print_pers('Batch size 1 (' + mode + '): {0}'.format(batchSize1[mode]), fileResultNameFull)
                for mode in ['train', 'val', 'test']:
                    util.print_pers('Batch size 2 (' + mode + '): {0}'.format(batchSize2[mode]), fileResultNameFull)
                util.print_pers("", fileResultNameFull)

                # numBatches1 = {}
                # numBatches2 = {}
                # numBatches1['train'] = dataset1_sizes['train'] / batchSize1['train']
                # numBatches2['train'] = dataset2_sizes['train'] / batchSize2['train']
                # print()

                # loaders
                adp_train_loader = torch.utils.data.DataLoader(adp_train,
                                                                    batch_size=batchSize1['train'], shuffle=True,
                                                                    num_workers=numWorkersP, pin_memory=True,
                                                                    drop_last=True)
                adp_val_loader = torch.utils.data.DataLoader(adp_val,
                                                                  batch_size=batchSize1['val'], shuffle=True,
                                                                  num_workers=numWorkersP, pin_memory=True,
                                                                  drop_last=True)
                adp_test_loader = torch.utils.data.DataLoader(adp_test,
                                                                  batch_size=batchSize1['test'], shuffle=True,
                                                                  num_workers=numWorkersP, pin_memory=True,
                                                                  drop_last=True)
                cnmc_train_loader = torch.utils.data.DataLoader(cnmc_train,
                                                                    batch_size=batchSize2['train'], shuffle=True,
                                                                    num_workers=numWorkersP, pin_memory=True,
                                                                    drop_last=True)
                cnmc_val_loader = torch.utils.data.DataLoader(cnmc_val,
                                                                  batch_size=batchSize2['val'], shuffle=True,
                                                                  num_workers=numWorkersP, pin_memory=True,
                                                                  drop_last=True)
                cnmc_test_loader = torch.utils.data.DataLoader(cnmc_test,
                                                                  batch_size=batchSize2['test'], shuffle=True,
                                                                  num_workers=numWorkersP, pin_memory=True,
                                                                  drop_last=True)

                # optim
                optimizer_ft = optim.SGD(currentModel.parameters(), lr=5 * 2e-4, momentum=0.9, weight_decay=5e-4)

                # warm-up
                if not evalMode:
                    util.print_pers('Warm-up', fileResultNameFull)
                    currentModel = functions.warmup(adp_train_loader, adp_val_loader, num_classes,
                                                    cnmc_train_loader, cnmc_val_loader, 2,
                                                    currentModel, optimizer_ft,
                                                    dataset1_sizes, dataset2_sizes,
                                                    cuda, dirResult)

                # different learning rates for base and classifiers
                my_list = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
                fc_params = list(
                    map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, currentModel.named_parameters()))))
                base_params = list(
                    map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, currentModel.named_parameters()))))

                #  uncertainty weighting, learnable loss weight param
                # log_sigma_A = torch.tensor([1.]).requires_grad_()
                # log_sigma_B = torch.tensor([1.]).requires_grad_()
                # loss_weight_list = [log_sigma_A, log_sigma_B]

                awl = functions.AutomaticWeightedLoss(2)

                optimizer_ft = optim.SGD([
                    {'params': base_params},  # lr for classifier is multiplied by 20
                    {'params': fc_params, 'lr': 5 * 2e-4},  # lr for classifier is multiplied by 20
                    # {'params': loss_weight_list}
                    {'params': awl.parameters(), 'weight_decay': 0}
                ], lr=2e-4, momentum=0.9, weight_decay=5e-4)
                # sched
                exp_lr_scheduler = list()
                # exp_lr_scheduler.append(lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.5))

                # re-enable grad
                for param in currentModel.parameters():
                    param.requires_grad = True  # re-enable gradients

                # train
                util.print_pers('Training', fileResultNameFull)
                # train net
                currentModel = functions.train_model_val(adp_train_loader, adp_val_loader, num_classes,
                                                         cnmc_train_loader, cnmc_val_loader, 2,
                                                         currentModel,
                                                         # log_sigma_A, log_sigma_B,
                                                         awl,
                                                         optimizer_ft, exp_lr_scheduler,
                                                         dataset1_sizes, dataset2_sizes, batchSize1, batchSize2,
                                                         num_epochs, numBatchesPretrain, cuda, log, dirResult, fileResultNameFull)

                # save model conf
                fileSaveModelIter = open(os.path.join(dirResult, 'model_{0}.dat'.format(r+1)), 'wb')
                pickle.dump([currentModel], fileSaveModelIter)
                fileSaveModelIter.close()


                # -------------------------------------- TEST -----------------------------------

                # display
                if log:
                    util.print_pers("Testing", fileResultNameFull)

                # pretrain for CNMC: forward few batches to update batch normalization parameters
                # currentModel = functions.pretrainBN(currentModel, cnmc_train_loader, cuda, numBatchesPretrain)

                # eval
                currentModel.eval()
                # zero the parameter gradients
                optimizer_ft.zero_grad()
                torch.no_grad()

                util.print_pers("\tDimensione dataset test: {0}".format(dataset2_sizes['test']), fileResultNameFull)

                numBatches = {}
                numBatches['test'] = np.round(dataset2_sizes['test'] / batchSize2['test'])

                # loop on images
                # init
                running_corrects = 0.0
                amountOfDB = 0
                predALL_test = torch.zeros(dataset2_sizes['test'])
                labelsALL_test = torch.zeros(dataset2_sizes['test'])
                dataiter = iter(cnmc_test_loader)
                batch_num = -1
                while True:

                    try:
                        inputs, label = dataiter.next()
                        batch_num = batch_num + 1
                    except StopIteration:
                        break

                    # get size of current batch
                    sizeCurrentBatch = label.size(0)
                    amountOfDB += sizeCurrentBatch

                    # if batch_num % 200 == 0:
                        # print("\t\tBatch n. {0} / {1}".format(batch_num, int(numBatches['test'])))

                    # stack
                    indStart = batch_num * batchSize2['test']
                    indEnd = indStart + sizeCurrentBatch

                    # extract features
                    if cuda:
                        inputs = inputs.to('cuda')
                        label = label.to('cuda')

                    # predict
                    with torch.set_grad_enabled(False):
                        dummy, outputs = currentModel(inputs)
                        if cuda:
                            outputs = outputs.to('cuda')

                        # softmax
                        _, preds = torch.max(outputs, 1)

                        predALL_test[indStart:indEnd] = preds
                        labelsALL_test[indStart:indEnd] = label

                    with torch.no_grad():
                        running_corrects += torch.sum(preds == label.data.int())

                # end for x,y

                with torch.no_grad():
                    test_acc = running_corrects / amountOfDB

                # confusion matrix
                CM = confusion_matrix(labelsALL_test, predALL_test)
                CM_perc = CM / dataset2_sizes['test']  # perc
                accuracyResult = util.accuracy(CM)
                CM_all = CM_all + CM
                CM_perc_all = CM_perc_all + CM_perc

                # print(output_test)
                util.print_pers("\tConfusion Matrix (%):", fileResultNameFull)
                util.print_pers("\t\t{0}".format(CM_perc * 100), fileResultNameFull)
                # util.print_pers("\tAccuracy (%): {0:.2f}".format(accuracyResult * 100), fileResultNameFull)
                util.print_pers("\tAccuracy (%): {:.4f}".format(test_acc), fileResultNameFull)

                # assign
                accuracyALL[r] = accuracyResult

                # newline
                util.print_pers("", fileResultNameFull)

                # save iter
                fileSaveIter = open(os.path.join(dirResult, 'results_{0}.dat'.format(r+1)), 'wb')
                pickle.dump([accuracyResult], fileSaveIter)
                fileSaveIter.close()

                # del
                if cuda:
                    del currentModel
                    del adp_train, adp_train_loader
                    del adp_val, adp_val_loader
                    del cnmc_train, cnmc_train_loader
                    del cnmc_val, cnmc_val_loader
                    del cnmc_test, cnmc_test_loader
                    del inputs, label
                    del outputs, preds
                    del optimizer_ft, exp_lr_scheduler
                    torch.cuda.empty_cache()

            # end loop on iterations

            # average accuracy
            meanAccuracy = np.mean(accuracyALL)
            stdAccuracy = np.std(accuracyALL)
            meanCM = CM_all / num_iterations
            meanCM_perc = CM_perc_all / num_iterations

            # display
            util.print_pers("", fileResultNameFull)
            util.print_pers("Mean classification accuracy over {0} iterations (%); {1:.2f}".format(num_iterations, meanAccuracy * 100),
                            fileResultNameFull)
            util.print_pers("Std classification accuracy over {0} iterations (%); {1:.2f}".format(num_iterations, stdAccuracy * 100),
                            fileResultNameFull)
            util.print_pers("\tMean Confusion Matrix over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0}".format(meanCM_perc * 100), fileResultNameFull)
            util.print_pers("\tTP (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[1, 1] * 100), fileResultNameFull)
            util.print_pers("\tTN (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[0, 0] * 100), fileResultNameFull)
            util.print_pers("\tFP (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[0, 1] * 100), fileResultNameFull)
            util.print_pers("\tFN (mean) over {0} iterations (%):".format(num_iterations), fileResultNameFull)
            util.print_pers("\t\t{0:.2f}".format(meanCM_perc[1, 0] * 100), fileResultNameFull)

            #close
            fileResult.close()

            # save
            fileSaveFinal = open(os.path.join(dirResult, 'resultsFinal.dat'), 'wb')
            pickle.dump([meanAccuracy], fileSaveFinal)
            fileSaveFinal.close()

            # del
            torch.cuda.empty_cache()




