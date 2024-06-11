import pandas as pd
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np


def get_train_data(excel_path_G, excel_path_P, excel_path_e1, excel_path_e2, device, args):

    edge_index = [[0, 1, 2, 2, 1, 5, 5, 7, 8, 9, 9, 11, 11, 13, 7, 15, 16, 15, 18, 18, 20, 16, 22, 24, 25, 26, 27, 27, 26, 0],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 24]]
    edge_index = torch.tensor(edge_index).to(device)
    steam1 = [59, 273, 118, 99, 120, 29, 110, 42, 282, 53, 23, 20, 28, 23, 151, 314, 114, 90, 19, 10, 44, 14, 102]
    steam2 = [0.6, 0.35, 0.25, 0.3, 0.6, 0.35, 0.6, 0.6, 0.6, 0.25, 0.45, 0.35, 0.45, 0.35, 0.6, 0.45, 0.4, 0.45, 0.35, 0.45, 0.3, 0.45, 0.35]
    e1 = [1340, 2919, 2834, 2300, 1350, 1290]
    e2 = [0.8, 0.8, 0.6, 0.4, 0.4, 0.4]
    t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    t = torch.tensor(t, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    t = torch.unsqueeze(t, dim=-1)
    t = t.expand(12, args.n_vertex)
    steam1 = torch.tensor(steam1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    steam2 = torch.tensor(steam2, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e1 = torch.tensor(e1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e2 = torch.tensor(e2, dtype=torch.float32, requires_grad=args.need_grad).to(device)

    G_df= pd.read_excel(excel_path_G)
    G_df = G_df.abs()
    P_df= pd.read_excel(excel_path_P)
    e1_df = pd.read_excel(excel_path_e1)
    e1_df = e1_df.abs()
    e2_df = pd.read_excel(excel_path_e2)

    return G_df, P_df, e1_df, e2_df, edge_index, steam1, steam2, e1, e2, t


def get_load_data(args):

    edge_index = [[0, 1, 2, 2, 1, 5, 5, 7, 8, 9, 9, 11, 11, 13, 7, 15, 16, 15, 18, 18, 20, 16, 22, 24, 25, 26, 27, 27, 26, 0],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 24]]
    device = torch.device("cuda" if args.enable_cuda else "cpu")
    edge_index = torch.tensor(edge_index).to(device)
    steam1 = [59, 273, 118, 99, 120, 29, 110, 42, 282, 53, 23, 20, 28, 23, 151, 314, 114, 90, 19, 10, 44, 14, 102]
    steam2 = [0.6, 0.35, 0.25, 0.3, 0.6, 0.35, 0.6, 0.6, 0.6, 0.25, 0.45, 0.35, 0.45, 0.35, 0.6, 0.45, 0.4, 0.45, 0.35, 0.45, 0.3, 0.45, 0.35]
    e1 = [1340, 2919, 2834, 2300, 1350, 1290]
    e2 = [0.8, 0.8, 0.6, 0.4, 0.4, 0.4]

    t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    steam1 = torch.tensor(steam1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    steam2 = torch.tensor(steam2, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e1 = torch.tensor(e1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e2 = torch.tensor(e2, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    t = torch.unsqueeze(t, dim=-1)
    t = t.expand(12, args.n_vertex)
    return edge_index, steam1, steam2, e1, e2, t

def load_data(dataset_name, len_train, len_val):
    train = dataset_name[: len_train]
    val = dataset_name[len_train: len_train + len_val]
    test = dataset_name[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, device, args):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, 1, n_vertex])

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]
    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=args.need_grad)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    return x, y

def data_split(S_data_x, S_data_y, E_data_x, E_data_y, edge_index, S1, S2, E1, E2, t):
    data_list = []
    for i in range(S_data_x.shape[0]):
        xs = S_data_x[i]
        ys = S_data_y[i]
        xe = E_data_x[i]
        ye = E_data_y[i]

        data = Data(x=xs, edge_index=edge_index, edge_attr=S1, y=ys)
        data.xe = xe
        data.ye = ye
        data.S2 = S2
        data.E1 = E1
        data.E2 = E2
        data.t = t
        data_list.append(data)
    return data_list


