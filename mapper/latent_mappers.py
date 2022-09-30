import torch
from torch import nn
from torch.nn import Module
from models.stylegan2.model import EqualLinear, PixelNorm


class Mapper(Module):

    def __init__(self, opts, num_layer=4, latent_dim=512):
        super(Mapper, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(num_layer):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x


class LevelsMapper(Module):

    def __init__(self, opts, mlp_list):
        super(LevelsMapper, self).__init__()

        self.opts = opts
        self.mlp_list = mlp_list
        self.mlps_pre = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.w_t = nn.ModuleList()
        self.mappings = nn.ModuleList()
        self.scales = nn.ModuleList()
        self.factory_kwargs = {'device': 'cuda:0', 'dtype': torch.float32}
        self.weights = [nn.Parameter(torch.ones(1, **self.factory_kwargs)) for i in range(len(mlp_list) * 4)]
        self.soft = nn.Softmax(dim=1)
        for i in range(len(mlp_list)):
            self.mappings.append(SubHairMapper(opts))
            self.mappings.append(SubHairMapper(opts))
            self.mappings.append(SubHairMapper(opts))
            self.mappings.append(SubHairMapper(opts))
            self.w_t.append(
                nn.Sequential(
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                    EqualLinear(512, 2048, lr_mul=0.01, activation='fused_lrelu'),
                )
            )
            self.mlps_pre.append(
                nn.Sequential(
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                    EqualLinear(512, 2048, lr_mul=0.01, activation='fused_lrelu'),
                )
            )
            self.mlps_pos.append(
                nn.Sequential(
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                    EqualLinear(512, 512, lr_mul=0.01, activation='fused_lrelu'),
                )
            )
        # self.mapping = SubHairMapper(opts)

    def forward(self, x, text_embedding):
        # tmp = []
        out = []
        j = 0
        weights = torch.FloatTensor(self.weights).reshape(-1, 4)
        weights = self.soft(weights)
        for i in range(18):
            if i in self.mlp_list:
                # tmp_out = []
                tmp = self.mlps_pre[j](x[:, i, :]).reshape(-1, 4, 512)   # 1x4x512
                text_embed = self.w_t[j](text_embedding.float()).reshape(-1, 4, 512)  # 1x4x512
                # tmp = torch.cat([tmp, text_embedding], dim=1).view(1, -1)  # 1x5x512->1x2560
                tmp_1 = weights[j][0] * self.mappings[4 * j + 0](tmp[:, 0, :], text_embed[:, 0, :])
                tmp_2 = weights[j][1] * self.mappings[4 * j + 1](tmp[:, 1, :], text_embed[:, 1, :])
                tmp_3 = weights[j][2] * self.mappings[4 * j + 2](tmp[:, 2, :], text_embed[:, 2, :])
                tmp_4 = weights[j][3] * self.mappings[4 * j + 3](tmp[:, 3, :], text_embed[:, 3, :])
                tmp = tmp_1 + tmp_2 + tmp_3 + tmp_4
                out.append(self.mlps_pos[j](tmp).view(-1, 1, 512))
                j += 1
            else:
                out.append(x[:, i:i+1, :])
        out = torch.cat(out, dim=1)
        return out


