import torch
import torch.nn as nn
from model import my_layers


class STGCNGraphConv(nn.Module):

    def __init__(self, args, blocks):
        super(STGCNGraphConv, self).__init__()

        self.steam_embed = my_layers.NodeEmbedding(args.steam_dim)
        self.e_embed = my_layers.NodeEmbedding(args.electricity_dim)
        self.steam_edge_embed = my_layers.edge_embed(args.steam_edge_dim, 0.01)

        self.virtual_edge = nn.Parameter(torch.randn(1, 1))
        self.stblock1 = my_layers.STConvBlock(args.Kt, args.n_vertex, blocks[0][-1], blocks[0 + 1], args.act_func,
                                               args.droprate, args.weight_type)
        self.stblock2 = my_layers.STConvBlock(args.Kt, args.n_vertex, blocks[1][-1], blocks[1 + 1], args.act_func,
                                               args.droprate, args.weight_type)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = my_layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], args.n_vertex, args.act_func,
                                             args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x_steam, x_e, t, edge_index, steam_weight, steam_d, e_weight, e_d):
        
        steam_feature = self.steam_edge_embed(steam_weight, steam_d, 1E2)       #l/d 1E2,1E3  d/l 1E2 1E3 l*d 1E1 1E3
        e_feature = self.steam_edge_embed(e_weight, e_d, 1E3)                   #d2 1E2 1E3
        virtual_feature = self.virtual_edge
        edge_weight = torch.cat((steam_feature, e_feature, virtual_feature), dim=0)

        x_steam = self.steam_embed(x_steam)
        x_e = self.e_embed(x_e)

        x = torch.cat((x_steam, x_e), dim=-1)

        x = self.stblock1(x, edge_index, edge_weight)
        x = self.stblock2(x, edge_index, edge_weight)

        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        return x




