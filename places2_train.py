import random
import torch
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from torchvision import utils
import numpy as np

#ccalculate mean and std of the dataset in path
# def calculate_mean_std(path):
# 	img_paths = glob.glob(path + "/*.png", recursive=True)
# 	num_imgs = len(img_paths)
# 	img_transform = transforms.Compose([transforms.ToTensor()])
# 	imgs = torch.stack([img_transform(Image.open(img_path).convert('RGB')) for img_path in img_paths])
# 	print(imgs.shape)
# 	print(imgs.mean(dim=(0, 2, 3)))
# 	print(imgs.std(dim=(0, 2, 3)))
# 	return imgs.mean(dim=(0, 2, 3)), imgs.std(dim=(0, 2, 3))

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

# reverses the earlier normalization applied to the image to prepare output
def unnormalize(x):
	# x.transpose_(1, 3)
	# x = x * torch.Tensor(STDDEV) + torch.Tensor(MEAN)
	# x.transpose_(1, 3)
	return x




class Places2Data (torch.utils.data.Dataset):

	def __init__(self, path_to_data, path_to_mask, noise_level):
		super().__init__()

		self.img_paths = glob.glob(path_to_data + "/*.png", recursive=True)
		self.mask_paths = glob.glob(path_to_mask + "/*.png")
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)
		self.noise_level = noise_level
		# normalizes the image: (img - MEAN) / STD and converts to tensor
		self.img_transform = transforms.Compose([transforms.ToTensor()])
		self.mask_transform = transforms.ToTensor()

	def __len__(self):
		return self.num_imgs

	def __getitem__(self, index):
		gt_img = Image.open(self.img_paths[index])
		#gt_img = gt_img.filter(ImageFilter.GaussianBlur(radius = 3))
		# print(gt_img)
		gt_img = self.img_transform(gt_img.convert('RGB'))
		# print(gt_img)
		mask = Image.open(self.mask_paths[index])
		mask = self.mask_transform(mask.convert('RGB'))
		noise = torch.randn(gt_img[0].shape) * self.noise_level
		#extemd moise to 3 channels
		noise = torch.stack([noise, noise, noise])
		noise_img = gt_img + noise
		image_mask = gt_img * (1-mask) + mask
		#noise_img = torch.unsqueeze(image_mask, 0)
		
		return noise_img, 1-mask, gt_img


class Places2Data_grey (torch.utils.data.Dataset):

	def __init__(self, path_to_data, path_to_mask, noise_level, num):
		super().__init__()

		self.img_paths = glob.glob(path_to_data + "/*.png", recursive=True)[0:num]
		self.mask_paths = glob.glob(path_to_mask + "/*.png")[0:num]
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)
		self.noise_level = noise_level
		# normalizes the image: (img - MEAN) / STD and converts to tensor
		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
		self.mask_transform = transforms.ToTensor()

	def __len__(self):
		return self.num_imgs

	def __getitem__(self, index):
		gt_img = Image.open(self.img_paths[index])
		#gt_img_grey = ImageOps.grayscale(gt_img)
		#gt_img = gt_img.filter(ImageFilter.GaussianBlur(radius = 3))
		# print(gt_img)
		gt_img_grey = self.img_transform(gt_img.convert('RGB'))
		gt_img = self.img_transform(gt_img.convert('RGB'))
		
		# print(gt_img)
		#mask = Image.open(self.mask_paths[index])
		#mask = self.mask_transform(mask.convert('RGB'))
		noise = torch.randn(gt_img[0].shape) * self.noise_level
		#extemd moise to 3 channels
		#noise = torch.stack([noise, noise, noise])
		noise_img = gt_img_grey + noise
		#image_mask = gt_img * (1-mask) + mask
		#noise_img = torch.unsqueeze(image_mask, 0)
		
		return noise_img, gt_img_grey, gt_img_grey

class Places2Data_inp (torch.utils.data.Dataset):

	def __init__(self, path_to_data , path_to_mask, num):
		super().__init__()

		self.img_paths = glob.glob(path_to_data + "/*.png", recursive=True)[0:num]
		self.mask_paths = glob.glob(path_to_mask + "/*.png", recursive=True)[0:num]
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)
		# normalizes the image: (img - MEAN) / STD and converts to tensor
		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
		self.mask_transform = transforms.Compose([transforms.ToTensor()])
		# self.img_transform = transforms.Compose([transforms.ToTensor()])
		# self.mask_transform = transforms.Compose([transforms.ToTensor()])

	def __len__(self):
		return self.num_imgs

	def __getitem__(self, index):
		gt_img = Image.open(self.img_paths[index])
		#gt_img_grey = ImageOps.grayscale(gt_img)
		gt_img_grey = self.img_transform(gt_img.convert('RGB'))
		#gt_img = self.img_transform(gt_img.convert('RGB'))
		
		# print(gt_img)
		mask = Image.open(self.mask_paths[index])
		mask = self.mask_transform(mask.convert('RGB'))[0,:,:]
                mask = torch.unsqueeze(mask, 0)
		#noise_img = gt_img_grey + noise
		image_mask = gt_img_grey * (1-mask) + mask
		#noise_img = torch.unsqueeze(image_mask, 0)
		
		return image_mask, mask, gt_img_grey


class Places2Data_mispix (torch.utils.data.Dataset):

	def __init__(self, path_to_data , path_to_mask, num):
		super().__init__()

		self.img_paths = glob.glob(path_to_data + "/*.png", recursive=True)[0:num]
		self.mask_paths = glob.glob(path_to_mask + "/*.png", recursive=True)[0:num]
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)
		# normalizes the image: (img - MEAN) / STD and converts to tensor
		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
		self.mask_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
		# self.img_transform = transforms.Compose([transforms.ToTensor()])
		# self.mask_transform = transforms.Compose([transforms.ToTensor()])

	def __len__(self):
		return self.num_imgs

	def __getitem__(self, index):
		gt_img = Image.open(self.img_paths[index])
		#gt_img_grey = ImageOps.grayscale(gt_img)
		gt_img_grey = self.img_transform(gt_img.convert('RGB'))
		#gt_img = self.img_transform(gt_img.convert('RGB'))
		#generate 50% missing pixels mask 
		mask = torch.rand(gt_img_grey.shape) > 0.5
		mask = mask.float()
		# print(gt_img)
		#mask = Image.open(self.mask_paths[index])
		#mask = ImageOps.grayscale(mask)
		# mask = self.mask_transform(mask)
		#noise_img = gt_img_grey + noise
		image_mask = gt_img_grey * mask
		#noise_img = torch.unsqueeze(image_mask, 0)
		
		return image_mask, mask, gt_img_grey


class Places2Data_blur (torch.utils.data.Dataset):

	def __init__(self, path_to_data, path_to_mask, blur_level, num):
		super().__init__()

		self.img_paths = glob.glob(path_to_data + "/*.png", recursive=True)[0:num]
		self.mask_paths = glob.glob(path_to_mask + "/*.png")[0:num]
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)
		self.blur_level = blur_level
		
		# normalizes the image: (img - MEAN) / STD and converts to tensor
		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
		self.mask_transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])

	def __len__(self):
		return self.num_imgs

	def __getitem__(self, index):
		gt_img = Image.open(self.img_paths[index])
		blur = gt_img.filter(ImageFilter.GaussianBlur(radius = self.blur_level))
		blur = self.img_transform(blur.convert('RGB'))
		# print(gt_img)
		gt_img = self.img_transform(gt_img.convert('RGB'))
		# print(gt_img)
		# mask = Image.open(self.mask_paths[index])
		# mask = self.mask_transform(mask.convert('RGB'))
		# noise = torch.randn(gt_img[0].shape) * self.noise_level
		# #extemd moise to 3 channels
		# noise = torch.stack([noise, noise, noise])
		# noise_img = gt_img + noise
		#image_mask = gt_img * (1-mask) + mask
		

		
		return blur, blur, gt_img


class FFHQData (torch.utils.data.Dataset):

	def __init__(self, path_to_data="/project/labate/heng/ffhq256", path_to_mask="/project/labate/heng/ffhq256", phase = "test"):
		super().__init__()
		
		self.img_paths = glob.glob(path_to_data + "/*.png", recursive=True)
		self.mask_paths = glob.glob(path_to_mask + "/*.png")
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)
		if phase == "train":
			self.img_paths = self.img_paths[:int(0.9*self.num_imgs)]
			self.mask_paths = self.mask_paths[:int(0.9*self.num_masks)]
		elif phase == "val":
			self.img_paths = self.img_paths[int(0.9*self.num_imgs):]
			self.mask_paths = self.mask_paths[int(0.9*self.num_masks):]
		# normalizes the image: (img - MEAN) / STD and converts to tensor
		self.img_transform = transforms.Compose([transforms.ToTensor()])
		self.mask_transform = transforms.ToTensor()
		self.len= len(self.img_paths)

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		gt_img = Image.open(self.img_paths[index])
		#gt_img = gt_img.filter(ImageFilter.GaussianBlur(radius = 3))
		# print(gt_img)
		gt_img = self.img_transform(gt_img.convert('RGB'))
		# print(gt_img)
		#noise = torch.randn(gt_img.shape) * 0.2
		noise = torch.randn(gt_img[0].shape) * 0.1
		#extemd moise to 3 channels
		noise = torch.stack([noise, noise, noise])
		noise_img = gt_img + noise

		
		return noise_img, gt_img, gt_img

# Unit Test
if __name__ == '__main__':
	places2 = Places2Data_blur(path_to_data = "/project/labate/heng/Places2_gauss/test/test00/image", path_to_mask = "/project/labate/heng/Places2_gauss/test/test00/mask", blur_level = 3, num = 4)
	print(len(places2))
	img, mask, gt = zip(*[places2[i] for i in range(4)]) # returns tuple of a single batch of 3x256x256 images
	print(np.shape(img[0]))
	print(np.shape(mask[0]))
	print(np.shape(gt[0]))
	# print(mask)
	# print(gt)
	img = torch.stack(img) # --> i x 3 x 256 x 256
	i = img == 0
	print(i.sum())
	mask = torch.stack(mask)
	gt = torch.stack(gt)

	grid = utils.make_grid(torch.cat((unnormalize(img), unnormalize(gt)), dim=0))
	utils.save_image(grid, "test.jpg")
