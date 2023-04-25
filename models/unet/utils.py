import cv2
import numpy as np
import torch
import torch.nn.functional as F


def preprocess(im, scale):
    w, h,_ = im.shape
    newW, newH = int(scale * w), int(scale * h)

    pil_img = cv2.resize(im, (newW, newH))
    img = np.asarray(pil_img)

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    if (img > 1).any():
        img = img / 255.0

        return img


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(preprocess(full_img, scale = 0.5))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

