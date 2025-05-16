import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from ..backbones.cal_resnets import C2DResNet50 as CALResNet
from ..backbones.cal_classifiers import Classifier as CALClassifier


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class GaitGL(nn.Module):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, model_cfg, *args, **kargs):
        super(GaitGL, self).__init__(*args, **kargs)
        self.build_network(model_cfg)  # Call the build_network function to build the model with model_cfg

    def build_network(self, model_cfg):

        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = model_cfg['dataset_name']

        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

            self.TP = PackSequenceWrapper(torch.max)
            self.HPP = GeMHPP()

            self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

            if 'SeparateBNNecks' in model_cfg.keys():
                self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
                self.Bn_head = False
            else:
                self.Bn = nn.BatchNorm1d(in_c[-1])
                self.Head1 = SeparateFCs(64, in_c[-1], class_num)
                self.Bn_head = True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(sils)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]
            embed = gait

        # n, _, s, h, w = sils.size()
        # retval = {
        #     'training_feat': {
        #         'triplet': {'embeddings': embed, 'labels': labs},
        #         'softmax': {'logits': logi, 'labels': labs}
        #     },
        #     'visual_summary': {
        #         #'image/sils': sils.view(n*s, 1, h, w)
        #     },
        #     'inference_feat': {
        #         'embeddings': embed
        #     }
        # }
        # return retval

        return embed, logi

class ProjectionHead(nn.Module):
    def __init__(
        self,
        config
    ):

        super().__init__()
        
        embedding_dim = config["embedding_dim"]
        projection_dim = config["projection_dim"]
        dropout = config["dropout"]
        norm_type = config["norm"]
        
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(projection_dim)
        else:
            self.norm = nn.LayerNorm(projection_dim)
        
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.norm(x)
        return x

    
class GaitFusion(BaseModel):
    
    def __init__(self, *args, **kargs):
        super(GaitFusion, self).__init__(*args, **kargs)
        
        self.probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},}
        self.gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],}
        
    def build_network(self, model_cfg):
        
        self.sil_encoder = GaitGL(model_cfg=model_cfg["backbone_cfg"]["GaitGL"])
        self.rgb_encoder = CALResNet(config=model_cfg["backbone_cfg"]["CAL"])
        self.rgb_classifier = CALClassifier(2048,74)

        self.sil_proj = ProjectionHead(config=model_cfg["proj_cfg"]["sil"])
        self.rgb_proj = ProjectionHead(config=model_cfg["proj_cfg"]["rgb"])
        
    
    def setup_encoders(self, model_cfg):
        if "weights" in model_cfg["backbone_cfg"]["GaitGL"]:
            checkpoint = torch.load(model_cfg["backbone_cfg"]["GaitGL"]["weights"], map_location=torch.device(
            "cuda", self.device))
            model_state_dict = checkpoint['model']
            self.sil_encoder.load_state_dict(model_state_dict, strict=False)
            self.sil_encoder.requires_grad_(False)
            self.msg_mgr.log_info("Loaded and Froze GaitGL weights")
            #print(self.sil_encoder)

        if "weights" in model_cfg["backbone_cfg"]["CAL"]:
            checkpoint = torch.load(model_cfg["backbone_cfg"]["CAL"]["weights"], map_location=torch.device(
            "cuda", self.device))
            model_state_dict = checkpoint['model_state_dict']
            
            self.rgb_encoder.load_state_dict(model_state_dict, strict=True)
            self.rgb_classifier.load_state_dict(checkpoint['classifier_state_dict'])
            self.rgb_encoder.requires_grad_(False)
            self.rgb_classifier.requires_grad_(False)
            self.msg_mgr.log_info("Loaded and Froze CAL weights")
            #print(self.rgb_encoder)
        
    def forward(self, inputs):
        
        _, labs, cond, _, _ = inputs
        
        if self.training:
            sil_embed, sil_logi = self.sil_encoder(inputs)
            sil_embed_flat = sil_embed.view(sil_embed.size(0), -1)
            sil_feats = self.sil_proj(sil_embed_flat)

            rgb_embed = self.rgb_encoder(rearrange(inputs[0][1],"b f h w c-> b c f h w"))
            rgb_logi = self.rgb_classifier(rgb_embed)

            rgb_feats = self.rgb_proj(rgb_embed)

            ###----------------Sum of features-----------------
            fused_feats = sil_feats + rgb_feats
            fused_feats = rearrange(fused_feats, 'b c -> b 1 c')

            ###---------------Concatenation of features-----------------
            # fused_feats = torch.cat([sil_feats, rgb_feats])
            # fused_feats = rearrange(fused_feats, 'b c -> b 1 c')
            # print(fused_feats.size())

            retval = {
                'training_feat': {
                    'triplet': {'embeddings': fused_feats, 'labels': torch.cat([labs, labs])},
                    'softmax': {'logits': sil_logi, 'labels': labs}
                },
                'visual_summary': {
                    # 'image/sils': rearrange(inputs[0][0],"b f h w -> (b f) 1 h w"),
                    # 'image/rgb': rearrange(inputs[0][1],"b f c h w -> (b f) c h w")
                },
                'inference_feat': {
                    'embeddings': fused_feats
                }
            }
        
        else:
            sil_embed, sil_logi = self.sil_encoder(inputs)
            sil_embed_flat = sil_embed.view(sil_embed.size(0), -1)
            sil_feats = self.sil_proj(sil_embed_flat)

            rgb_embed = self.rgb_encoder(rearrange(inputs[0][1],"b f h w c-> b c f h w"))
            rgb_feats = self.rgb_proj(rgb_embed)

            ###---------Sum of features-----------------
            fused_feats = sil_feats + rgb_feats
            fused_feats = rearrange(fused_feats, 'b c -> b 1 c')

            ###---------------Concatenation of features----------
            # fused_feats = torch.cat([sil_feats, rgb_feats], dim=1)
            # fused_feats = rearrange(fused_feats, 'b c -> b 1 c')

            retval = {
                'visual_summary': {
                    # 'image/sils': rearrange(inputs[0][0],"b f h w -> (b f) 1 h w"),
                },
                'inference_feat': {
                    'embeddings': fused_feats
                }
            }
            
        return retval
        
