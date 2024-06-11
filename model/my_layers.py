import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.init as init
import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class NodeEmbedding(nn.Module):
    def __init__(self, feature_dim):
        super(NodeEmbedding, self).__init__()
        self.fno = FNO1d(feature_dim)

    def forward(self, x):
        output_list = []
        for i in range(x.shape[-1]):

            node_data = x[..., i]

            node_data = node_data.permute(0, 2, 1)

            node_output = self.fno(node_data)

            node_output = node_output.permute(0, 2, 1)

            output_list.append(node_output)


        output_data = torch.stack(output_list, dim=-1)
        return output_data

class FNO1d(nn.Module):
    def __init__(self, feature_dim, modes=16, width=12):
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width


        self.fc0 = nn.Linear(feature_dim, 128)  # input channel is 2: (a(x), x)
        init.xavier_normal_(self.fc0.weight)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        init.xavier_normal_(self.w0.weight)
        init.xavier_normal_(self.w1.weight)
        init.xavier_normal_(self.w2.weight)
        init.xavier_normal_(self.w2.weight)

        self.fc1 = nn.Linear(128, 128)
        init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 12)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, x):

        x = self.fc0(x)     #[32, 12, 2]->[32,12,64]
        #x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.tanh(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        #x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device,
                             dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def compl_mul1d(self, input, weights):

        return torch.einsum("bix,iox->box", input, weights)

class TimeEncode(torch.nn.Module):
    def __init__(self, args, factor=5):
        super(TimeEncode, self).__init__()
        #init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = args.time_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        self.dense = torch.nn.Linear(time_dim, time_dim, bias=False)
        torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)# [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1) # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)
        return harmonic**2 / torch.sqrt(torch.Tensor([12]).cuda())
        #return self.dense(harmonic**2)


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))
        init.xavier_normal_(self.align_conv.weight)
    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x

        return x


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                           padding=self.__padding, dilation=dilation, groups=groups, bias=bias)
        init.xavier_normal_(self.weight)

        if bias:
            # 初始化bias为0
            init.constant_(self.bias, 0)
    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]

        return result


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1,
                 bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0,
                                           dilation=dilation, groups=groups, bias=bias)
        init.xavier_normal_(self.weight)
        if bias:
            # 初始化bias为0
            init.constant_(self.bias, 0)
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.align_t = Align(c_in, c_out)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1),
                                            enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1),
                                            enable_padding=False, dilation=1)
        self.act_func = act_func
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]

        x_causal_conv = self.causal_conv(x)



        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))


            else:

                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)

        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')

        return x


class edge_embed(nn.Module):
    def __init__(self, steam_feature_num, droprate):
        super(edge_embed, self).__init__()
        self.fc1 = nn.Linear(2, 8, bias=False)
        self.fc2 = nn.Linear(8, 1, bias=False)
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, weight1, weight2, ratio):

        x = torch.stack((weight2/weight1*ratio, weight2**2), dim=0)
        x = x.permute(1, 0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        '''
        x = weight2/weight1   #[23]
        x = x.unsqueeze(-1)      #[23,1]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        #x = self.leaky_relu(x)
        x = self.tanh(x)
         '''
        return torch.squeeze(x, 0)



class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, weight_type):
        super(GraphConvLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv3 = GCNConv(64, out_channels)
        self.tanh = nn.Tanh()


    def forward(self, x, edge_index, edge_weight):
        batch_size, num_features, time_steps, vertex = x.size()

        # Initialize an empty list to store the results for each time step
        outputs = []

        for t in range(time_steps):

            x_t = x[:, :, t, :].transpose(1, 2)
            x_t = self.conv1(x_t, edge_index, edge_weight)

            x_t = self.dropout1(x_t)
            x_t = self.conv3(x_t, edge_index, edge_weight)

            x_t = self.tanh(x_t)
            x_t = x_t.transpose(1, 2).contiguous()

            outputs.append(x_t)


        outputs = torch.stack(outputs, dim=2)

        return outputs


class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, n_vertex, last_block_channel, channels, act_func, droprate, weight_type):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(channels[0], channels[1], weight_type)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x, edge_index, edge_weight):

        x = self.tmp_conv1(x)

        x = self.graph_conv(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.tanh(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x


class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.tc1_ln_t = nn.LayerNorm([1, channels[0]])
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x
