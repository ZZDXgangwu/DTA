import sys, os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

import config
from gnn import GNNNet
from utils import *
from emetrics import *
from data_process import create_dataset
from config import *

# seed=1
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
datasets = [['davis', 'kiba'][config.dataset]]

cuda_name = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][config.cuda]
print('cuda_name:', cuda_name)
fold = [0, 1, 2, 3, 4][config.fold]
tune = config.tune
TRAIN_BATCH_SIZE = config.train_batch
TEST_BATCH_SIZE = config.test_batch
LR = config.learn_rate
NUM_EPOCHS = config.epochs

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Main program: iterate over different datasets
result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = GNNNet()
model.to(device)
model_st = GNNNet.__name__
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_rmse = 1000
best_test_rmse = 1000

best_mse = 1000
best_test_mse = 1000
best_pearson = 0
best_spearman = 0
best_ci = 0
best_test_pearson = 0
best_test_spearman = 0
best_test_ci = 0

for dataset in datasets:

    train_data, valid_data = create_dataset(dataset, fold, tune=tune)
    if tune:
        val_type = 'valid'
    else:
        val_type = 'test'

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                               collate_fn=collate)

    best_epoch = -1
    model_file_name = 'models/model_' + model_st + '_' + dataset + '_' + str(fold) + '.model'
    for epoch in range(NUM_EPOCHS):

        train(model, device, train_loader, optimizer, epoch + 1)
        print('predicting for ' + val_type + ' data')
        G, P = predicting(model, device, valid_loader)
        ret = [get_rmse(G, P), get_mse(G, P), get_pearson(G, P), get_spearman(G, P), get_ci(G, P)]
        if ret[1] < best_mse:
            best_mse = ret[1]
            best_rmse, best_mse, best_pearson, best_spearman, best_ci = ret
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
            print(val_type + ' set mse improved at epoch ', best_epoch, '; best ' + val_type + ' mse', best_mse,
                  model_st, dataset, fold)
            print('valid best rmse ', best_rmse, '; best mse ', best_mse, '; best pearson ', best_pearson,
                  '; best spearman ', best_spearman, '; best ci ', best_ci)

        else:
            print(val_type + ' set No improvement since epoch ', best_epoch, '; best ' + val_type + ' mse', best_mse,
                  model_st, dataset,
                  fold)
            print(val_type + ' set      rmse ', ret[0], ';      mse ', ret[1], ';      pearson ', ret[2], ';      '
                                                                                                          'spearman ',
                  ret[3], ';      ci ', ret[4])
            print(val_type + ' best rmse ', best_rmse, '; best mse ', best_mse, '; best pearson ', best_pearson,
                  '; best spearman ', best_spearman, '; best ci ', best_ci)
