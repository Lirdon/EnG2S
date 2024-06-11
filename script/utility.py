import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
import pandas as pd
import torch.nn.functional as F

def evaluate_model(model, loss, data_iter, args):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in data_iter:
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

            device = torch.device('cuda' if args.enable_cuda else 'cpu')


            y_pred = model(xs, xe, t, edge_index, S1, S2, E1, E2, True)# [batch_size, num_nodes]
            y_pred = y_pred.squeeze(2)
            indices = torch.tensor([0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23, 24, 25, 28, 29, 30], device=y.device)
            y = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices).reshape(y.shape[0], 2, len(indices))
            y_pred = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices).reshape(y_pred.shape[0], 2,
                                                                                               len(indices))
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_sub(indices, y, y_pred, scaler):
    y1, y2 = torch.split(y, split_size_or_sections=1, dim=1)
    # 将张量从第二个维度上挤压，得到形状为 [32, 24] 的张量
    y1 = y1.squeeze(dim=1)
    y2 = y2.squeeze(dim=1)

    original_shape = (y1.shape[0], y1.shape[1])
    y1 = scaler[0].inverse_transform(y1.cpu().numpy()).reshape(-1)
    y2 = scaler[1].inverse_transform(y2.cpu().numpy()).reshape(-1)
    y1 = y1.reshape(original_shape)
    y2 = y2.reshape(original_shape)
    y1 = y1[:, indices].reshape(-1)
    y2 = y2[:, indices].reshape(-1)

    y_pred1, y_pred2 = torch.split(y_pred, split_size_or_sections=1, dim=1)
    # 将张量从第二个维度上挤压，得到形状为 [32, 24] 的张量
    y_pred1 = y_pred1.squeeze(dim=1)
    y_pred2 = y_pred2.squeeze(dim=1)
    y_pred1 = scaler[0].inverse_transform(y_pred1.cpu().numpy()).reshape(-1)
    y_pred2 = scaler[1].inverse_transform(y_pred2.cpu().numpy()).reshape(-1)
    y_pred1 = y_pred1.reshape(original_shape)
    y_pred2 = y_pred2.reshape(original_shape)
    y_pred1 = y_pred1[:, indices].reshape(-1)
    y_pred2 = y_pred2[:, indices].reshape(-1)

    d1 = np.abs(y1 - y_pred1)
    d2 = np.abs(y2 - y_pred2)
    return d1, d2, y1, y2


def evaluate_metric(model, data_iter, S_score, E_score, args):
    model.eval()
    with torch.no_grad():
        steam_mae_G, steam_sum_y_G, steam_mape_G, steam_mse_G, steam_mae_P, steam_sum_y_P, steam_mape_P, steam_mse_P, = [], [], [], [], [], [], [], []
        air_mae_G, air_sum_y_G, air_mape_G, air_mse_G, air_mae_P, air_sum_y_P, air_mape_P, air_mse_P, = [], [], [], [], [], [], [], []
        for batch in data_iter:
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

            #device = torch.device('cuda' if args.enable_cuda else 'cpu')


            y_pred = model(xs, xe, t, edge_index, S1, S2, E1, E2, False) # [batch_size, num_nodes]
            y_pred = y_pred.squeeze(2)

            indices_steam = [0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23]
            #indices_air = [24, 25, 28, 29, 30]
            indices_air = [0, 1, 4, 5, 6]
            y_steam = y[:, :, :args.n_steam]
            y_air = y[:, :, args.n_steam:]
            y_pred_steam = y_pred[:, :, :args.n_steam]
            y_pred_air = y_pred[:, :, args.n_steam:]

            steam_d1, steam_d2, steam_y1, steam_y2 = evaluate_sub(indices_steam, y_steam, y_pred_steam, S_score)
            air_d1, air_d2, air_y1, air_y2 = evaluate_sub(indices_air, y_air, y_pred_air, E_score)

            steam_mae_G += steam_d1.tolist()
            steam_mae_P += steam_d2.tolist()
            steam_sum_y_G += steam_y1.tolist()
            steam_sum_y_P += steam_y2.tolist()
            steam_mape_G += (steam_d1 / steam_y1).tolist()
            steam_mape_P += (steam_d2 / steam_y2).tolist()
            steam_mse_G += (steam_d1 ** 2).tolist()
            steam_mse_P += (steam_d2 ** 2).tolist()

            air_mae_G += air_d1.tolist()
            air_mae_P += air_d2.tolist()
            air_sum_y_G += air_y1.tolist()
            air_sum_y_P += air_y2.tolist()
            air_mape_G += (air_d1 / air_y1).tolist()
            air_mape_P += (air_d2 / air_y2).tolist()
            air_mse_G += (air_d1 ** 2).tolist()
            air_mse_P += (air_d2 ** 2).tolist()
        steam_MAE_G = np.array(steam_mae_G).mean()
        steam_MAE_P = np.array(steam_mae_P).mean()
        steam_RMSE_G = np.sqrt(np.array(steam_mse_G).mean())
        steam_RMSE_P = np.sqrt(np.array(steam_mse_P).mean())
        steam_WMAPE_G = np.sum(np.array(steam_mae_G)) / np.sum(np.array(steam_sum_y_G))
        steam_WMAPE_P = np.sum(np.array(steam_mae_P)) / np.sum(np.array(steam_sum_y_P))
        air_MAE_G = np.array(air_mae_G).mean()
        air_MAE_P = np.array(air_mae_P).mean()
        air_RMSE_G = np.sqrt(np.array(air_mse_G).mean())
        air_RMSE_P = np.sqrt(np.array(air_mse_P).mean())
        air_WMAPE_G = np.sum(np.array(air_mae_G)) / np.sum(np.array(air_sum_y_G))
        air_WMAPE_P = np.sum(np.array(air_mae_P)) / np.sum(np.array(air_sum_y_P))

        #return MAE, MAPE, RMSE
        return steam_MAE_G, steam_RMSE_G, steam_WMAPE_G, steam_MAE_P, steam_RMSE_P, steam_WMAPE_P, air_MAE_G, air_RMSE_G, air_WMAPE_G, air_MAE_P, air_RMSE_P, air_WMAPE_P

def custom_collate(data_list):
    # 提取x, y, edge_index, edge_attr
    xs = torch.stack([data.x for data in data_list]) # 堆叠x
    xe = torch.stack([data.xe for data in data_list])
    ys = torch.stack([data.y for data in data_list]) # 堆叠y
    ye = torch.stack([data.ye for data in data_list])
    t = torch.stack([data.t for data in data_list])
    # 由于每个data的edge_index和edge_attr都是相同的，
    # 只需要从第一个元素中获取它们
    edge_index = data_list[0].edge_index
    S1 = data_list[0].edge_attr
    S2 = data_list[0].S2
    E1 = data_list[0].E1
    E2 = data_list[0].E2

    # 创建一个空的Data对象并填充数据
    batch_data = Batch()
    batch_data.xs = xs
    batch_data.ys = ys
    batch_data.xe = xe
    batch_data.ye = ye
    batch_data.S1 = S1
    batch_data.S2 = S2
    batch_data.E1 = E1
    batch_data.E2 = E2
    batch_data.t = t
    batch_data.edge_index = edge_index

    return batch_data

def calculate_pde_loss(dpdt, dpdx, dgdt, dgdx, P, G, d, A, args, Flag):
    #steam = IAPWS97(P=P, T=T)
    # 计算定压比热 c_p (单位kJ/kgK)
    #cp = steam.cp
    # 计算定容比热 c_v (单位kJ/kgK)
    #cv = steam.cv
    # 计算比焓 h (单位kJ/kg)
    #h = steam.h
    # 计算比内能 u (单位kJ/kg)
    #u = steam.u
    if Flag:
        loss1 = dgdx + A * dpdt / (args.R * args.Tb)
        loss2 = dpdx + dgdt / A + args.lam * args.R * args.Tb * G ** 2 / (2 * A ** 2 * d * P)
    else:
        loss1 = dgdx + A * dpdt / (args.R_air * args.Tb)
        loss2 = dpdx + dgdt / A + args.lam * args.R_air * args.Ta * G ** 2 / (2 * A ** 2 * d * P)
    #loss3 = dtdt + dgdx*((R**3*G**2*T**3)/(2*A_pipe[pipe_cnt]**3*P**3*cv) * ((h-u)*R*T)/(A_pipe[pipe_cnt]*P*cv))-G*R**2*T**2*dpdx/(A_pipe[pipe_cnt]*P**2*cv) + cp*T*R*G*dtdx/(A_pipe[pipe_cnt]*P*cv) + 4*R*K*T*(T-T0)/(D[pipe_cnt]*P*cv) - lam*R**3*G**3*T**3/(2*A_pipe[pipe_cnt]*D[pipe_cnt]*P**3*cv)
    #loss = loss1 + loss2
    #print("内层函数", loss2)
    return loss1, loss2

def caculate_grad(args, y_pred, edge_weights, t, zscore, Flag):
    if args.need_grad:
        if Flag:
            g = y_pred[:, 0, :args.n_steam]
            p = y_pred[:, 1, :args.n_steam]
            indices = [0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23]
        else:
            g = y_pred[:, 0, args.n_steam:]
            p = y_pred[:, 1, args.n_steam:]
            indices = [0, 4, 5, 6]

        g_np = zscore[0].inverse_transform(g.detach().cpu().numpy())
        p_np = zscore[1].inverse_transform(p.detach().cpu().numpy())
        factor_g = g_np / g.detach().cpu().numpy()
        factor_p = p_np / p.detach().cpu().numpy()

        # 使用这个因子对 PyTorch 张量进行乘法操作
        g = g * torch.from_numpy(factor_g).to(g.device)
        p = p * torch.from_numpy(factor_p).to(g.device)


        #indices = [0, 3, 4, 6, 10, 12, 14, 17]
        g = g[:, indices]
        p = p[:, indices]
        l_grad1 = 0
        l_grad2 = 0
        # 提取第 node_num 个节点两个特征值对第 node_num-1 个权重的梯度
        for batch_cnt in range(g.shape[0]):
            l_batch_grad1 = 0
            l_batch_grad2 = 0
            for i in range(g.shape[-1]-1):
                # dgdx = torch.autograd.grad(outputs=g[batch_cnt, i + 1], inputs=edge_weights[i],retain_graph=True)
                #print(edge_weights.requires_grad)

                dgdx = torch.autograd.grad(outputs=g[batch_cnt, i + 1], inputs=edge_weights, retain_graph=True)[0][indices[i+1]-1]
                dpdx = torch.autograd.grad(outputs=p[batch_cnt, i + 1], inputs=edge_weights, retain_graph=True)[0][indices[i+1]-1]
                dgdt = torch.autograd.grad(outputs=g[batch_cnt, i + 1], inputs=t, retain_graph=True)[0][batch_cnt].squeeze(
                    -1).sum()
                dpdt = torch.autograd.grad(outputs=p[batch_cnt, i + 1], inputs=t, retain_graph=True)[0][batch_cnt].squeeze(
                    -1).sum()

                A = 3.14 * edge_weights[i] ** 2 / 4
                l_grad_temp1, l_grad_temp2 = calculate_pde_loss(dpdt, dpdx, dgdt, dgdx, p[batch_cnt, i + 1],
                                                         g[batch_cnt, i + 1], edge_weights[indices[i+1]-1], A, args, Flag)
                l_batch_grad1 += torch.abs(l_grad_temp1)
                l_batch_grad2 += torch.abs(l_grad_temp2)
            l_grad1 += l_batch_grad1
            l_grad2 += l_batch_grad2
        return l_grad1/g.shape[0], l_grad2/g.shape[0]
    else:
        return 0, 0

def return_grad(args, y_pred, edge_weights, t, zscore, node_num, cnt):
    if args.need_grad:
        g = y_pred[:, 0, :]
        p = y_pred[:, 1, :]
        g_np = zscore[0].inverse_transform(g.detach().cpu().numpy())
        p_np = zscore[1].inverse_transform(p.detach().cpu().numpy())
        factor_g = g_np / g.detach().cpu().numpy()
        factor_p = p_np / p.detach().cpu().numpy()

        # 使用这个因子对 PyTorch 张量进行乘法操作
        g = g * torch.from_numpy(factor_g).to(g.device)
        p = p * torch.from_numpy(factor_p).to(g.device)
        indices = [0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23]
        g = g[:, indices]
        p = p[:, indices]
        dgdx_grad = []
        dpdx_grad = []
        dgdt_grad = []
        dpdt_grad = []
        g_pred = []
        p_pred = []
        # 提取第 node_num 个节点两个特征值对第 node_num-1 个权重的梯度
        for batch_cnt in range(g.shape[0]):
            dgdx = torch.autograd.grad(outputs=g[batch_cnt, node_num + 1], inputs=edge_weights, retain_graph=True)[0][
                indices[node_num + 1] - 1]
            dpdx = torch.autograd.grad(outputs=p[batch_cnt, node_num + 1], inputs=edge_weights, retain_graph=True)[0][
                indices[node_num + 1] - 1]
            dgdt = torch.autograd.grad(outputs=g[batch_cnt, node_num + 1], inputs=t, retain_graph=True)[0][batch_cnt].squeeze(
                -1).sum()
            dpdt = torch.autograd.grad(outputs=p[batch_cnt, node_num + 1], inputs=t, retain_graph=True)[0][batch_cnt].squeeze(
                -1).sum()
            dgdx_grad.append(dgdx.cpu().detach().tolist())
            dpdx_grad.append(dpdx.cpu().detach().tolist())
            dgdt_grad.append(dgdt.cpu().detach().tolist())
            dpdt_grad.append(dpdt.cpu().detach().tolist())
            g_pred.append(g[batch_cnt, node_num + 1].cpu().detach().tolist())
            p_pred.append(p[batch_cnt, node_num + 1].cpu().detach().tolist())
        df = pd.DataFrame({'g': g_pred, 'p': p_pred, 'dgdx': dgdx_grad, 'dpdx': dpdx_grad, 'dgdt': dgdt_grad, 'dpdt': dpdt_grad})
        df.to_excel('origin{}.xlsx'.format(cnt), index=False)
        #df.to_excel(excel_writer, sheet_name=f'Sheet_{node_num}', index=False)

def weight_loss(edge_weight, dl):
    x = edge_weight / dl

    loss_extra = x.var().sum()
    return loss_extra