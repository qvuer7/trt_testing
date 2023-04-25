import torch
from models.unet.unet_model import UNet
import cv2
import numpy as np
from models.unet.utils import predict_img
device = torch.device('cpu')


model = UNet(n_channels = 3, n_classes = 2)
checkpoint = torch.load(r'C:\Users\Andrii\PycharmProjects\trt_experiments\models\unet\weights\unet_carvana_scale0.5_epoch2.pth',
                        map_location = device)
model.load_state_dict(checkpoint)

image = cv2.imread(r'C:\Users\Andrii\PycharmProjects\trt_experiments\images\carvana_image_example.jpeg')
#
# mask = predict_img(net=model, full_img=image, device = device)
# mask = mask.astype(np.uint8)
# mask*=255
#
#

model.eval()
x = torch.rand(1, 3, 224, 224)

torch_out = model(x)

torch.onnx.export(model, x,
                  r'C:\Users\Andrii\PycharmProjects\trt_experiments\onnx_files\unet_carvana.onnx',
                  input_names=['input'], output_names=['output'])
