import torch
import clip
import sys
sys.path.append("../")
from models.mtcnn import detect_faces
from torchvision.transforms import ToPILImage, PILToTensor


class FaceLoss(torch.nn.Module):

    def __init__(self, opts):
        super(FaceLoss, self).__init__()
        self.mse = torch.nn.MSELoss().cuda()
        self.toPIL = ToPILImage()
        self.opts = opts

    def forward(self, x, x_hat):
        x_clone = x.cpu().clone()
        # x_hat_clone = x_hat.cpu().clone()
        b, c, h, w = x.shape

        loss = 0.
        for i in range(b):
            image_x = self.toPIL(x_clone[i].squeeze(0)).convert("RGB")
            bounding_boxes = detect_faces(image_x)

            if len(bounding_boxes) == 1:
                x1, y1, x2, y2 = int(bounding_boxes[0][0]), int(bounding_boxes[0][1]), \
                                 int(bounding_boxes[0][2]), int(bounding_boxes[0][3])

                mask = torch.zeros(x[i].shape).cuda()
                mask[:, y1:y2, x1:x2] = 1
                # x_clone[i, :, y1:y2, x1:x2] = 0
                # x_hat_clone[i, :, y1:y2, x1:x2] = 0
                loss += self.mse(mask * x[i], mask * x_hat[i])
        # loss = self.mse(x_clone.float(), x_hat_clone.float()) / b
        return loss / b


class BackGroundLoss(torch.nn.Module):

    def __init__(self, opts):
        super(BackGroundLoss, self).__init__()
        self.mse = torch.nn.MSELoss().cuda()
        self.toPIL = ToPILImage()
        self.opts = opts

    def forward(self, x, x_hat):
        x_clone = x.cpu().clone()
        # x_hat_clone = x_hat.cpu().clone()
        b, c, h, w = x.shape

        loss = 0.
        for i in range(b):
            image_x = self.toPIL(x_clone[i].squeeze(0)).convert("RGB")
            bounding_boxes = detect_faces(image_x)

            if len(bounding_boxes) == 1:
                x1, y1, x2, y2 = int(bounding_boxes[0][0]), int(bounding_boxes[0][1]), \
                                 int(bounding_boxes[0][2]), int(bounding_boxes[0][3])

                mask = torch.ones(x[i].shape).cuda()
                mask[:, y1:y2, x1:x2] = 0
                # x_clone[i, :, y1:y2, x1:x2] = 0
                # x_hat_clone[i, :, y1:y2, x1:x2] = 0
                loss += self.mse(mask * x[i], mask * x_hat[i])
        # loss = self.mse(x_clone.float(), x_hat_clone.float()) / b
        return loss / b



if __name__ == '__main__':
    from PIL import Image
    x = Image.open('./00000.png')
    x = PILToTensor()(x).unsqueeze(0).cuda()
    ls = FaceLoss(opts=None).cuda()
    print(ls(x, x*2))