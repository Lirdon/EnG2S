import logging
import argparse
import random
import os
import tqdm
import numpy as np
import math
import pandas as pd
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import my_model
from script import get_data, utility, earlystopping
import pickle
import torch.nn.functional as F
from tqdm import tqdm
# from itertools import islice
# from torch.utils.tensorboard import SummaryWriter
def set_env(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='IES')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--need_grad', type=bool, default=False, help='default as True')
    parser.add_argument('--n_vertex', type=int, default=31)
    parser.add_argument('--n_steam', type=int, default=24)
    parser.add_argument('--n_air', type=int, default=7)
    parser.add_argument('--n_mid_vertex', type=int, default=15)
    parser.add_argument('--n_his', type=int, default=12)
    parser.add_argument('--n_pred', type=int, default=3,
                        help='the number of time interval for predcition, default as 3')
    parser.add_argument('--steam_dim', type=int, default=2)
    parser.add_argument('--electricity_dim', type=int, default=2)
    parser.add_argument('--steam_edge_dim', type=int, default=2)
    parser.add_argument('--ele_edge_dim', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=12)
    parser.add_argument('--mid_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--time_dim', type=int, default=12)
    parser.add_argument('--time_intvl', type=int, default=10)
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--stblock_num', type=int, default=2)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--pregraph_conv_type', type=str, default='preGraphLayer_nomid',
                        choices=['preGraphLayer', 'preGraphLayer_nomid'])
    parser.add_argument('--weight_type', type=int, default=3, help='1 for none, 2 for *, 3 for /')
    parser.add_argument('--weight_decay_rate', type=float, default=0.1, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=40, help='epochs, default as 10000')
    parser.add_argument('--opt', type=str, default='adam', help='optimizer, default as adam')
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--lam', type=float, default=0.05)
    parser.add_argument('--Tb', type=float, default=549.9658495560284)
    parser.add_argument('--Ta', type=float, default=288.15)
    parser.add_argument('--R', type=float, default=0.4615)
    parser.add_argument('--R_air', type=float, default=0.287)
    parser.add_argument('--T0', type=float, default=273.15)
    parser.add_argument('--exp_data', type=bool, default=False)

    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # For stable experiment results
    set_env(args.seed)

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num


    blocks = []
    blocks.append([args.embed_dim])
    for l in range(args.stblock_num):
        blocks.append([args.mid_dim, args.mid_dim, args.mid_dim])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([2])

    return args, device, blocks

def get_iter(G_df, P_df, len_train, len_val):
    train_G, val_G, test_G = get_data.load_data(G_df, len_train, len_val)
    train_P, val_P, test_P = get_data.load_data(P_df, len_train, len_val)
    zscore_G = preprocessing.StandardScaler()
    zscore_P = preprocessing.StandardScaler()
    train_G = zscore_G.fit_transform(train_G)
    val_G = zscore_G.transform(val_G)
    test_G = zscore_G.transform(test_G)
    train_P = zscore_P.fit_transform(train_P)
    val_P = zscore_P.transform(val_P)
    test_P = zscore_P.transform(test_P)
    zscore = [zscore_G, zscore_P]

    x_train_G, y_train_G = get_data.data_transform(train_G, args.n_his, args.n_pred, device, args)
    x_val_G, y_val_G = get_data.data_transform(val_G, args.n_his, args.n_pred, device, args)
    x_test_G, y_test_G = get_data.data_transform(test_G, args.n_his, args.n_pred, device, args)
    x_train_P, y_train_P = get_data.data_transform(train_P, args.n_his, args.n_pred, device, args)
    x_val_P, y_val_P = get_data.data_transform(val_P, args.n_his, args.n_pred, device, args)
    x_test_P, y_test_P = get_data.data_transform(test_P, args.n_his, args.n_pred, device, args)
    x_train = torch.cat((x_train_G, x_train_P), dim=1)
    y_train = torch.cat((y_train_G, y_train_P), dim=1)
    x_val = torch.cat((x_val_G, x_val_P), dim=1)
    y_val = torch.cat((y_val_G, y_val_P), dim=1)
    x_test = torch.cat((x_test_G, x_test_P), dim=1)
    y_test = torch.cat((y_test_G, y_test_P), dim=1)
    return x_train, y_train, x_val, y_val, x_test, y_test, zscore

def data_preparate(args, GATG1, GATP1, GATG2, GATP2, device):
    G_df, P_df, E1_df, E2_df, edge_index, S1, S2, E1, E2, t = get_data.get_train_data(GATG1, GATP1, GATG2, GATP2, device, args)

    data_col = G_df.shape[0]
    val_and_test_rate = 0.1
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    S_x_train, S_y_train, S_x_val, S_y_val, S_x_test, S_y_test, S_score = get_iter(G_df, P_df, len_train, len_val)
    E_x_train, E_y_train, E_x_val, E_y_val, E_x_test, E_y_test, E_score = get_iter(E1_df, E2_df, len_train, len_val)

    S_x_train = S_x_train.cpu().numpy()
    S_y_train = S_y_train.cpu().numpy()
    E_x_train = E_x_train.cpu().numpy()
    E_y_train = E_y_train.cpu().numpy()
    S_x_val = S_x_val.cpu().numpy()
    S_y_val = S_y_val.cpu().numpy()
    E_x_val = E_x_val.cpu().numpy()
    E_y_val = E_y_val.cpu().numpy()
    S_x_test = S_x_test.cpu().numpy()
    S_y_test = S_y_test.cpu().numpy()
    E_x_test = E_x_test.cpu().numpy()
    E_y_test = E_y_test.cpu().numpy()

    data = {
        'S_x_train': S_x_train,
        'S_y_train': S_y_train,
        'S_x_val': S_x_val,
        'S_y_val': S_y_val,
        'S_x_test': S_x_test,
        'S_y_test': S_y_test,
        'E_x_train': E_x_train,
        'E_y_train': E_y_train,
        'E_x_val': E_x_val,
        'E_y_val': E_y_val,
        'E_x_test': E_x_test,
        'E_y_test': E_y_test
    }

    # 使用pickle将数据保存到一个文件中
    with open('data1.pickle', 'wb') as f:
        pickle.dump(data, f)
    with open('S_score1.pkl', 'wb') as f:
        pickle.dump(S_score, f)
    with open('E_score1.pkl', 'wb') as f:
        pickle.dump(E_score, f)
def load_data(args, data_path, Sscore_path, Escore_path):
    edge_index, S1, S2, E1, E2, t = get_data.get_load_data(args)
    device = torch.device("cuda" if args.enable_cuda else "cpu")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    S_x_train = torch.from_numpy(data['S_x_train']).to(device)
    S_y_train = torch.from_numpy(data['S_y_train']).to(device)
    S_x_val = torch.from_numpy(data['S_x_val']).to(device)
    S_y_val = torch.from_numpy(data['S_y_val']).to(device)
    S_x_test = torch.from_numpy(data['S_x_test']).to(device)
    S_y_test = torch.from_numpy(data['S_y_test']).to(device)
    E_x_train = torch.from_numpy(data['E_x_train']).to(device)
    E_y_train = torch.from_numpy(data['E_y_train']).to(device)
    E_x_val = torch.from_numpy(data['E_x_val']).to(device)
    E_y_val = torch.from_numpy(data['E_y_val']).to(device)
    E_x_test = torch.from_numpy(data['E_x_test']).to(device)
    E_y_test = torch.from_numpy(data['E_y_test']).to(device)
    with open(Sscore_path, 'rb') as f:
        S_score = pickle.load(f)
    with open(Escore_path, 'rb') as f:
        E_score = pickle.load(f)
    data_train = get_data.data_split(S_x_train, S_y_train, E_x_train, E_y_train, edge_index, S1, S2, E1, E2, t)
    data_val = get_data.data_split(S_x_val, S_y_val, E_x_val, E_y_val, edge_index, S1, S2, E1, E2, t)
    data_test = get_data.data_split(S_x_test, S_y_test, E_x_test, E_y_test, edge_index, S1, S2, E1, E2, t)
    
    # data_train = Data(x=x_train, edge_index=edge_index, edge_attr=edge_weights, y=y_train)
    train_iter = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=utility.custom_collate)
    # data_val = Data(x=x_val, edge_index=edge_index, edge_attr=edge_weights, y=y_val)
    val_iter = DataLoader(data_val, batch_size=args.batch_size, shuffle=True, collate_fn=utility.custom_collate)
    # data_test = Data(x=x_test, edge_index=edge_index, edge_attr=edge_weights, y=y_test)
    test_iter = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, collate_fn=utility.custom_collate)

    return S_score, E_score, train_iter, val_iter, test_iter

def prepare_model(args, blocks):
    loss = nn.MSELoss()
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    model = my_model.STGCNGraphConv(args, blocks).to(device)

    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer {args.opt} is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(S_score, E_score, loss, args, optimizer, scheduler, es, model, train_iter, val_iter):

    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for batch in tqdm(train_iter):

            #输入的x节点应该分成多个类型，特征的维度和数值应该也不一样，按照x_1,x_2的格式输入，然后先通过一层embed层扩展到相同维度，然后拼起来
            #后面计算pde时只需要使用蒸汽节点部分，即x_1
            #对边特征也一样先嵌入，然后看看GCN可不可以处理多维度的边特征，不行的话还是归一到一个维度

            t = batch.t

            #x = batch.x
            xs = batch.xs
            xe = batch.xe

            edge_index = batch.edge_index
            S1 = batch.S1  #蒸汽管道长度
            S2 = batch.S2

            E1 = batch.E1
            E2 = batch.E2

            ys = batch.ys
            ye = batch.ye
            y = torch.cat((ys,ye), dim=-1)

            y_pred = model(xs, xe, t, edge_index, S1, S2, E1, E2, False)  # [batch_size, num_nodes]
            y_pred = y_pred.squeeze(2)
            if args.need_grad:
                l_grad_steam = utility.caculate_grad(args, y_pred, S1, t, S_score, True)
                l_grad_air = utility.caculate_grad(args, y_pred, E1, t, E_score, False)
                indices = torch.tensor([0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23, 24, 25, 28, 29, 30], device=device)
                y = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices).reshape(y.shape[0], 2, len(indices))
                y_pred = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices).reshape(y_pred.shape[0], 2,
                                                                                               len(indices))
                l_p = loss(y_pred, y)
                l = l_grad_steam[0] / 1000000 + l_grad_steam[1] / 10000 + l_grad_air[0] / 1000000 + l_grad_air[
                    1] / 10000 + l_p
            else:
                indices = torch.tensor([0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23, 24, 25, 28, 29, 30], device=device)
                y = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices).reshape(y.shape[0], 2, len(indices))
                y_pred = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices).reshape(y_pred.shape[0], 2,
                                                                                                   len(indices))

                l = loss(y_pred, y)


            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        scheduler.step()
        val_loss = val(model, val_iter, loss)


        # GPU memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

        print('Epoch: {:03d} | Lr: {:.20f} |Train loss: {:.6f} | Val loss: {:.6f} | GPU occupy: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        if es.step(val_loss):
            print('Early stopping.')
            break

        if (epoch + 1) % args.save_interval == 0:
            save_path = f"model_nofno_notime_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")


@torch.no_grad()
def val(model, val_iter, loss):
    model.eval()
    l_sum, n = 0.0, 0
    for batch in val_iter:
        xs = batch.xs
        xe = batch.xe
        edge_index = batch.edge_index
        S1 = batch.S1  # 蒸汽管道长度
        S2 = batch.S2
        E1 = batch.E1
        E2 = batch.E2
        t = batch.t
        ys = batch.ys
        ye = batch.ye
        y = torch.cat((ys, ye), dim=-1)


        y_pred = model(xs, xe, t, edge_index, S1, S2, E1, E2, False) # [batch_size, num_nodes]
        y_pred = y_pred.squeeze(2)
        indices = torch.tensor([0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23, 24, 25, 28, 29, 30], device=device)
        y = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices).reshape(y.shape[0], 2, len(indices))
        y_pred = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices).reshape(y_pred.shape[0], 2,
                                                                                           len(indices))
        l = loss(y_pred, y)
        '''
        indices_steam = torch.tensor([0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23], device=device)
        indices_air = torch.tensor([24, 25, 28, 29, 30], device=device)
        y_steam = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices_steam).reshape(y.shape[0], 2, len(indices_steam))
        y_pred_steam = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices_steam).reshape(y_pred.shape[0], 2, len(indices_steam))
        y_air = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices_air).reshape(y.shape[0], 2, len(indices_air))
        y_pred_air = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices_air).reshape(y_pred.shape[0], 2, len(indices_air))
        l_steam = loss(y_pred_steam, y_steam)
        l_air = loss(y_pred_air, y_air)
        l = l_steam + l_air
        '''
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)

@torch.no_grad()
def test(S_score, E_score, loss, model, test_iter, args):
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter, args)
    steam_MAE_G, steam_RMSE_G, steam_WMAPE_G, steam_MAE_P, steam_RMSE_P, steam_WMAPE_P, air_MAE_G, air_RMSE_G, air_WMAPE_G, air_MAE_P, air_RMSE_P, air_WMAPE_P  = utility.evaluate_metric(model,
                                                                                                         test_iter,
                                                                                                         S_score,
                                                                                                         E_score,
                                                                                                         args)
    print(
        f' Test loss {test_MSE:.6f} \n'
        f'steam_MAE_G {steam_MAE_G:.6f} | steam_RMSE_G {steam_RMSE_G:.6f} | steam_WMAPE_G {steam_WMAPE_G:.8f} | steam_MAE_P {steam_MAE_P:.6f} | steam_RMSE_P {steam_RMSE_P:.6f} | steam_WMAPE_P {steam_WMAPE_P:.8f}\n'
        f'air_MAE_G {air_MAE_G:.6f} | air_RMSE_G {air_RMSE_G:.6f} | air_WMAPE_G {air_WMAPE_G:.8f} | air_MAE_P {air_MAE_P:.6f} | air_RMSE_P {air_RMSE_P:.6f} | air_WMAPE_P {air_WMAPE_P:.8f}')



if __name__ == "__main__":
    # writer = SummaryWriter('logs')

    # Logging
    # logger = logging.getLogger('stgcn')
    # logging.basicConfig(filename='stgcn.log', level=logging.INFO)
    logging.basicConfig(level=logging.INFO)
    args, device, blocks = get_parameters()
    print(f'Using device: {device}')

    '''
    GATG = 'data\steam_G.xlsx'
    GATP = "data\steam_P.xlsx"
    GATE1 = 'data\preair_G.xlsx'
    GATE2 = "data\preair_P.xlsx"
    '''

    #data_preparate(args, GATG, GATP, GATE1, GATE2, device)

    S_score, E_score, train_iter, val_iter, test_iter = load_data(args, 'data/data.pickle', 'data/S_score.pkl', 'data/E_score.pkl')

    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)

    train(S_score, E_score, loss, args, optimizer, scheduler, es, model, train_iter, val_iter)
    test(S_score, E_score, loss, model, test_iter, args)


