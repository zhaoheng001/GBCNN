import argparse
import torch
import os
import random
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
import torchvision
import cv2
from partial_conv_net import PartialConvUNet
from places2_train import unnormalize, MEAN, STDDEV
from loss import CalculateLoss
from loss import CalculateLoss
from partial_conv_net import PartialConvUNet
from places2_train import Places2Data, FFHQData, Places2Data_grey, Places2Data_inp, Places2Data_blur, Places2Data_mispix
from SDPF_net import SparseNet, SDPF
import time
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


class loss_fn_sparse(torch.nn.Module):
	def __init__(self):
		super(loss_fn_sparse, self).__init__()
		self.mse_loss = torch.nn.MSELoss()
	def forward(self, image, mask, output, gt):
		loss_dict = {}
		loss_dict["loss"] = self.mse_loss(output, gt)
		return loss_dict
def PSNR(output, gt):
	output = output.cpu().numpy()
	output = np.transpose(output, (0, 2, 3, 1))
	output = np.squeeze(output)
	output = output * 255
	output = output.astype(np.uint8)
	gt = gt.cpu().numpy()
	gt = np.transpose(gt, (0, 2, 3, 1))
	gt = np.squeeze(gt)
	gt = gt * 255
	gt = gt.astype(np.uint8)
	psnr = 0.0
	for i in range(output.shape[0]):
		psnr += cv2.PSNR(output[i], gt[i])
	psnr = psnr / output.shape[0]
	return psnr

def psnr(output, gt):
	#gt = gt.cpu().numpy()
	psnr = 0.0
	for i in range(output.shape[0]):
	    psnr += sk_psnr(output[i].squeeze().cpu().numpy(), gt[i].squeeze().cpu().numpy())
	psnr = psnr / output.shape[0]
	return psnr
def ssim(output, gt):
	ssim = 0.0
	for i in range(output.shape[0]):
	    s = sk_ssim(output[i].squeeze().cpu().numpy(), gt[i].squeeze().cpu().numpy(), data_range=1.0)
	    #print(s)
	    ssim += s
	ssim = ssim / output.shape[0]
	return ssim


image_num = str(random.randint(1, 328501)).zfill(8)
parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default="/project/labate/heng/inpainting-partial-conv/datasets/BSD68")
parser.add_argument("--mask", type=str, default="/project/labate/heng/inpainting-partial-conv/datasets/BSD68")
parser.add_argument("--noise", type=str, default="/project/labate/heng/inpainting-partial-conv/datasets/noise")
parser.add_argument("--pred", type=str, default="/project/labate/heng/inpainting-partial-conv/datasets/pred")
parser.add_argument("--model_dir", type=str, default="/project/labate/heng/inpainting-partial-conv/training_logs/2023-10-02 11:08:40.598874_sparsenet_places2_0.1/model/model_sparsenet.pth")
#ircnn /project/labate/heng/inpainting-partial-conv/training_logs/2023-08-31 02:01:23.785758_sparsenet_places2_0.1/model/model_sparsenet.pth
	  #/project/labate/heng/inpainting-partial-conv/training_logs/2023-08-31 19:00:52.119392_sparsenet_places2_0.196/model/model_sparsenet.pth
#gbcnn /project/labate/heng/inpainting-partial-conv/training_logs/2023-09-02 18:50:58.861398_sdpf_places2_0.1/model/model_sdpf.pth
	  #/project/labate/heng/inpainting-partial-conv/training_logs/2023-09-02 18:38:57.122438_sdpf_places2_0.196/model/model_sdpf.pth
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--network", type=str, choices=["pcnet", "sparsenet", "sdpf"], default="sparsenet")


args = parser.parse_args()

cwd = os.getcwd()
if args.gpu >= 0:
	device = torch.device("cuda:{}".format(args.gpu))
else:
	device = torch.device("cpu")

#load model and test on test data set
if args.network == "pcnet":
	model = PartialConvUNet()
elif args.network == "sparsenet":
	model = SparseNet().to(device)
	loss_fn = loss_fn_sparse().to(device)
elif args.network == "sdpf":
	model = SDPF().to(device)
	loss_fn = loss_fn_sparse().to(device)
model.to(device)
model.eval()
checkpoint = torch.load(args.model_dir)
model.load_state_dict(checkpoint["model"])


test_dataset = Places2Data_mispix(path_to_data=args.img, path_to_mask=args.mask, num = 68)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
print("test_loader:", len(test_loader))
psnr_all = []
ssim_all = []
loss_test = []
img_idx = 0
time_cnt = []
time_start = time.time()
with torch.no_grad():
    for i, (image, mask, gt) in enumerate(test_loader):
        image, mask, gt = [x.to(device) for x in [image, mask, gt]]
        output = model(image)
        psnr_all.append(psnr(output, gt))
        #ssim_all.append(ssim(output, gt))
        print("psnr:", psnr_all[-1])
        loss_fn = loss_fn_sparse().to(device)
        loss_dict = loss_fn(image, mask, output, gt)
        loss = 0.0
        for key, value in loss_dict.items():
            loss += value
        loss_test.append(loss.item())
        # print("loss:", loss_test[-1])
        # print("output:", output.shape)
        for j in range(output.shape[0]):
            torchvision.utils.save_image(output[j], args.pred + "/pred_{}.png".format(img_idx))
            torchvision.utils.save_image(image[j], args.noise + "/noise_{}.png".format(img_idx))
            img_idx += 1
time_end = time.time()

print("mean psnr:", np.mean(psnr_all))
#print("mean ssim:", np.mean(ssim_all))
print("mean loss:", np.mean(loss_test))
print("mean time:", np.mean(time_end - time_start))
# print("mean time:", np.mean(time_cnt))	
