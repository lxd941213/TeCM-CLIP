import torch
import clip
import os
from mapper.options.train_options import TrainOptions
from PIL import Image
from torchvision import transforms


class Acc(torch.nn.Module):

    def __init__(self, opts, model):
        super(Acc, self).__init__()
        self.model = model
        # self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)

    def forward(self, image, text):

        image = self.avg_pool(self.upsample(image))
        return self.model(image, text)[0] / 100



to_tensor = transforms.ToTensor()
pil_to_tensor = transforms.PILToTensor()


def calcAcc(iamges_path, ACC):
    nums = 0
    result = 0.
    classes = os.listdir(iamges_path)
    for idx, label in enumerate(classes):
        text = clip.tokenize(label).cuda()
        single_images_path = os.path.join(iamges_path, label)
        images = os.listdir(single_images_path)
        for image in images:
            nums += 1
            img = os.path.join(single_images_path, image)
            tensor = to_tensor(Image.open(img).convert('RGB')).unsqueeze(0).cuda()
            result += ACC(tensor, text)

    return result / nums


if __name__ == '__main__':
    opts = TrainOptions().parse()
    device = 'cuda:0'
    model, preprocess = clip.load("ViT-B/32", device=device)
    ACC = Acc(opts, model)
    iamges_path = ''
    res = calcAcc(iamges_path, ACC)
    print(res)