import torch
from torch import nn
import os
from models.facial_recognition.model_irse import Backbone
import os
from mapper.options.train_options import TrainOptions
from PIL import Image
from torchvision import transforms


class IDS(nn.Module):
    def __init__(self, opts):
        super(IDS, self).__init__()
        # print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        similarity = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            similarity += diff_target
            count += 1

        return similarity / count


to_tensor = transforms.ToTensor()
pil_to_tensor = transforms.PILToTensor()


def calcIDS(iamges_path, ori_iamges_path, IDS):
    nums = 0
    result = 0.
    classes = os.listdir(iamges_path)
    for idx, label in enumerate(classes):
        single_images_path = os.path.join(iamges_path, label)
        # ori_single_images_path = os.path.join(ori_iamges_path, label)
        images = os.listdir(single_images_path)
        ori_images = os.listdir(ori_iamges_path)
        i = -1
        for image in images:
            i += 1
            nums += 1
            img = os.path.join(single_images_path, image)
            ori_img = os.path.join(ori_iamges_path, ori_images[i])
            tensor = to_tensor(Image.open(img).convert('RGB')).unsqueeze(0).cuda()
            ori_tensor = to_tensor(Image.open(ori_img).convert('RGB')).unsqueeze(0).cuda()
            result += IDS(tensor, ori_tensor)

    return result / nums

if __name__ == '__main__':
    opts = TrainOptions().parse()
    device = 'cuda:0'
    IDSimilarity = IDS(opts)
    iamges_path = 'I:\\codes\\StyleCLIP\\outputs\\emotion_wo_bg'
    ori_iamges_path = 'I:\\codes\\StyleCLIP\\Inputs'
    res = calcIDS(iamges_path, ori_iamges_path, IDSimilarity)
    print(res)