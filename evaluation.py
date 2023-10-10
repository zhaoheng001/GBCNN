#calculate psnr and ssim of images from folders
import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity

img_path = "/project/labate/heng/Places2_gray/test/test_all/image"
noise_path = "/project/labate/heng/inpainting-partial-conv/images_blur_sdpf"
sdpf_pred_path = "/project/labate/heng/inpainting-partial-conv/pred_images_blur_sdpf"
sparse_pred_path = "/project/labate/heng/inpainting-partial-conv/pred_images_blur_sparse"
img_list = os.listdir(img_path)
noise_list = os.listdir(noise_path)
sdpf_pred_list = os.listdir(sdpf_pred_path)
sparse_pred_list = os.listdir(sparse_pred_path)


sdpf_psnr = []
sparse_psnr = []
sdpf_ssim = []
sparse_ssim = []

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
#define function calculate ssim
# def SSIM(output, gt):
# 	output = output.cpu().numpy()
# 	output = np.transpose(output, (0, 2, 3, 1))
# 	output = np.squeeze(output)



for i in range(len(img_list)):
    img = cv2.imread(os.path.join(img_path, img_list[i]))
    noise = cv2.imread(os.path.join(noise_path, noise_list[i]))
    sdpf_pred = cv2.imread(os.path.join(sdpf_pred_path, sdpf_pred_list[i]))
    sparse_pred = cv2.imread(os.path.join(sparse_pred_path, sparse_pred_list[i]))
    sdpf_psnr.append(cv2.PSNR(img, sdpf_pred))
    sparse_psnr.append(cv2.PSNR(img, sparse_pred))
    sdpf_ssim.append(structural_similarity(img, sdpf_pred, channel_axis=2))
    sparse_ssim.append(structural_similarity(img, sparse_pred, channel_axis=2))
    print("{}-th sdpf_psnr:".format(i), sdpf_psnr[-1])
    print("{}-th spar_psnr:".format(i), sparse_psnr[-1])
    print("{}-th sdpf_ssim:".format(i), sdpf_ssim[-1])
    print("{}-th spar_ssim:".format(i), sparse_ssim[-1])


print("mean sdpf_psnr:", np.mean(sdpf_psnr))
print("mean sparse_psnr:", np.mean(sparse_psnr))
print("mean sdpf_ssim:", np.mean(sdpf_ssim))
print("mean sparse_ssim:", np.mean(sparse_ssim))


