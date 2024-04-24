import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F

from objects import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from functools import partial


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class backbone(nn.Module):
    def __init__(self, arch, pretrained=True):
        super(backbone, self).__init__()
        self.arch = arch
        

        if self.arch=="vit":

            model = models_vit.__dict__['vit_large_patch16'](
                num_classes=11,
                drop_path_rate=0.2,
                global_pool=True,
            )
            # load RETFound weights
            checkpoint = torch.load('/root/autodl-tmp/classification/RETFound_cfp_weights.pth', map_location='cpu')
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)

            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)

            # print("Model = %s" % str(model))

        else:
            model = models.__dict__[arch](pretrained=pretrained)

        if arch.startswith('resnet'):
            # print(list(model.children()))
            # print(model.fc.in_features)
            # exit(0)

            self.out_features = model.fc.in_features
            self.net_f = nn.Sequential(*(list(model.children())[:-1]))
        elif arch.startswith('vgg'):
            self.out_features = model.classifier[0].in_features
            self.model = nn.Sequential(*(list(model.children())[:-1]))
        elif arch.startswith('efficientnet'):
            self.out_features = model.classifier[1].in_features
            self.model = nn.Sequential(*(list(model.children())[:-1]))
        elif arch=="vit":
            # print(list(model.children())[:])
            # exit(0)
            # print(model.fc.out_features)
            # exit(0)
            self.out_features = 2156
            self.net_f = nn.Sequential(*(list(model.children())[:-1]))
        else:
            raise RuntimeError("backbone {} have not implemented!")

    def forward(self, x):
        if self.arch.startswith('resnet'):
            x = self.net_f(x)
        elif self.arch=="vit":
            x = self.net_f(x)
        else:
            x = self.model(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # exit(0)
        return x


class bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(bottleneck, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class classifier(nn.Module):
    def __init__(self, num_classes, bottleneck_dim=256, type="linear"):
        super(classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weight_norm(nn.Linear(bottleneck_dim, num_classes), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, num_classes)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class weight(nn.Module):
    def __init__(self, feature_dim):
        super(weight, self).__init__()
        self.w = nn.Parameter(torch.tensor(2) * torch.rand([feature_dim, 1]) - 1)

    def forward(self, taus):
        mu = F.softmax(F.relu(self.w.T.matmul(taus.T)), dim=1)  # [1, len(args.src)]
        return mu

if __name__ == '__main__':
    f = backbone("resnet50").cuda()
    b1 = bottleneck(feature_dim=f.out_features).cuda()
    b2 = bottleneck(feature_dim=f.out_features).cuda()
    b3 = bottleneck(feature_dim=f.out_features).cuda()
    b4 = bottleneck(feature_dim=f.out_features).cuda()
    c = classifier(num_classes=5).cuda()
    input = torch.randn([16, 3, 224, 224]).cuda()
    # tgt_features =
    # src_features_all = torch.zeros(4, input, tgt_features.shape[1]).cuda()
    feature1 = b1(f(input))
    feature2 = b2(f(input))
    feature3 = b3(f(input))
    feature4 = b4(f(input))
    src_features_all = torch.stack([feature2, feature3, feature4], dim=0)
    # print(src_features_all.shape)
    # s0 = torch.mean(torch.mean(feature1 ** 2, dim=0))
    # print(s0)
    pw = torch.mean(torch.sqrt(torch.mean(torch.mean((src_features_all - feature1) ** 2, dim=1), dim=0)))
    print(pw)

    # s = torch.mean(torch.sum((src_features_all - feature1) ** 2, dim=2)).sum()
    # print(s)
    # taus = torch.randn([4, 256]).cuda()
    # print(taus.shape)
    # wl = torch.nn.DataParallel(weight(256)).cuda()
    # # logits = torch.randn([4, 64, 2]).cuda()
    # x = wl(taus)
    # print(x)
    # u = torch.sum(logits * x.T.unsqueeze(dim=2), dim=0)

    # print(u.shape)