import argparse
import os
import torch
import numpy as np
import cv2
from torch.utils import data
from tqdm import tqdm
#from tensorboardX import SummaryWriter
from loss import CalculateLoss
from partial_conv_net import PartialConvUNet
from places2_train import Places2Data, Places2Data_blur, FFHQData, Places2Data_grey, Places2Data_inp, Places2Data_mispix
from SDPF_net import SparseNet, SDPF
import torchvision
from loss import CharbonnierLoss, EdgeLoss
import datetime
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

class SubsetSampler(data.sampler.Sampler):
	def __init__(self, start_sample, num_samples):
		self.num_samples = num_samples
		self.start_sample = start_sample

	def __iter__(self):
		return iter(range(self.start_sample, self.num_samples))

	def __len__(self):
		return self.num_samples


def requires_grad(param):
	return param.requires_grad

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
#define a class to calculate MSE loss of a batch of images
class loss_fn_sparse(torch.nn.Module):
	def __init__(self):
		super(loss_fn_sparse, self).__init__()
		self.mse_loss = torch.nn.MSELoss()
	def forward(self, image, mask, output, gt):
		loss_dict = {}
		loss_dict["loss"] = self.mse_loss(output, gt)
		#loss_dict["tv"] = total_variation_loss(output)*0.05
		return loss_dict
def total_variation_loss(image):
    l1 = torch.nn.L1Loss()
    # shift one pixel and get loss1 difference (for both x and y direction)
    loss = l1(image[:, :, :, :-1], image[:, :, :, 1:]) + l1(image[:, :, :-1, :], image[:, :, 1:, :])
    return loss	
#define psnr function for a batch of images
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



#define a function to do testing
# def test(model, test_loader, device, args):
# 	model.eval()
# 	psnr = []
# 	loss_test = []
# 	with torch.no_grad():
# 		for i, (image, mask, gt) in enumerate(test_loader):
# 			image, mask, gt = [x.to(device) for x in [image, mask, gt]]
# 			print("image shape:", image.shape)
# 			output = model(image, mask)
# 			print("output shape:", output.shape)
# 			#calculate psnr
# 			psnr.append(PSNR(output, gt))
# 			print("psnr:", psnr[-1])
# 			loss_fn = CalculateLoss().to(device)
# 			loss_dict = loss_fn(image, mask, output, gt)
# 			loss = 0.0
# 			for key, value in loss_dict.items():
# 				loss += value
# 			loss_test.append(loss.item())
# 			print("loss:", loss_test[-1])
# 			output = output.cpu().numpy()
# 			output = np.transpose(output, (0, 2, 3, 1))
# 			output = np.squeeze(output)
# 			output = output * 255
# 			output = output.astype(np.uint8)
# 		for j in range(output.shape[0]):
# 			cv2.imwrite("/project/labate/heng/inpainting-partial-conv/pred_images" + "/image_{}.png".format(j), output[j])
# 			#print mean psnr of the whole test set
# 	return np.mean(psnr), np.mean(loss_test)




#sparsenet: 10epoch 29.1205
#sdpfnet: 10epoch 29.1307



parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="/project/labate/heng/Places2_gray/train/image")
parser.add_argument("--mask_path", type=str, default="/project/labate/heng/Places2_gray/train/image")
parser.add_argument("--val_path", type=str, default="/project/labate/heng/Places2_gray/test/test_all/image")
parser.add_argument("--val_mask", type=str, default="/project/labate/heng/Places2_gray/test/test_all/image")

parser.add_argument("--log_dir", type=str, default="/training_logs")
parser.add_argument("--save_dir", type=str, default="/project/labate/heng/inpainting-partial-conv/training_logs/2023-08-26 23:55:50.195505_sparsenet_places2_0.1/model/model_sparsenet.pth")
parser.add_argument("--load_model", type=int, default=0)
parser.add_argument('--lr', default=0.001, type=float)
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight_decay', '--wd', default=1e-3, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
#parser.add_argument("--lr", type=float, default=0.01)

parser.add_argument("--lr_drop_rate", type=float, default=0.90)
parser.add_argument("--fine_tune_lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--fine_tune", type = int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=32)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=5000)
parser.add_argument("--network", type=str, choices=["pcnet", "sparsenet", "sdpf"], default="sparsenet")
parser.add_argument("--data", type=str, choices=["ffhq", "places2"], default="places2")
parser.add_argument("--noise_level", type=int, default=0.1)
parser.add_argument("--blur_kernel", type=int, default=3)
parser.add_argument("--train_num", type=int, default=221000)
parser.add_argument("--val_num", type=int, default=4100)


args = parser.parse_args()

cwd = os.getcwd()
#IRCNN PSNR: 28.8813, SSIM: 0.8389
#RFCNN PSNR: 29.1927, SSIM: 0.8492


#Tensorboard SummaryWriter setup
# if not os.path.exists(cwd + args.log_dir):
# 	os.makedirs(cwd + args.log_dir)


# if not os.path.exists(cwd + args.save_dir):
# 	os.makedirs(cwd + args.save_dir)

if args.gpu >= 0:
	device = torch.device("cuda:{}".format(args.gpu))
else:
	device = torch.device("cpu")

data_train = Places2Data_mispix(args.train_path, args.mask_path, args.train_num)
data_size = len(data_train)
print("Loaded training dataset with {} samples and {} masks".format(data_size, data_size))

data_val = Places2Data_mispix(args.val_path, args.val_mask, args.val_num)
val_loader = data.DataLoader(data_val, args.batch_size, num_workers=args.num_workers)
print("number of validation samples:", len(data_val))

# assert(data_size % args.batch_size == 0)
iters_per_epoch = data_size // args.batch_size

# data_val = Places2Data(args.val_path, args.mask_path)
# print("Loaded validation dataset...")

# Move model to gpu prior to creating optimizer, since parameters become different objects after loading
if args.network == "sparsenet":
	model = SparseNet().to(device)
	loss_fn = loss_fn_sparse().to(device)
elif args.network == "pcnet":
	model = PartialConvUNet().to(device)
	loss_fn = CalculateLoss().to(device)
elif args.network == "sdpf":
	model = SDPF().to(device)
	loss_fn = loss_fn_sparse().to(device)
criterion_char = CharbonnierLoss()
criterion_edge = EdgeLoss()
print("load model from {}".format(args.save_dir))
print("Loaded model to device...")
print("Model:", str(args.network))
print(model)
print("number of parameters:", get_parameter_number(model))
print("noise level:", args.noise_level)
print("blur kernel:", args.blur_kernel)
print("data:", data_train)

# Set the fine tune learning rate if necessary
if args.fine_tune==1:
    lr = args.fine_tune_lr
    model.freeze_enc_bn = True
else:
    lr = args.lr

# Adam optimizer proposed in: "Adam: A Method for Stochastic Optimization"
# filters the model parameters for those with requires_grad == True
print("fine tune:", args.fine_tune)
print("load model:", args.load_model)
print("learning rate:{}".format(args.lr))
print("learning rate drop rate:{}".format(args.lr_drop_rate))
optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, cycle_momentum=False)

# optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], args.lr,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay)
# #lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_drop_rate)
print("Setup Adam optimizer...")
print("Setup loss function...")

# Resume training on model
if args.load_model:
	filename = args.save_dir 
	checkpoint_dict = torch.load(filename)

	model.load_state_dict(checkpoint_dict["model"])
	optimizer.load_state_dict(checkpoint_dict["optimizer"])

	print("Resume training on model:{}".format(args.load_model))

	# Load all parameters to gpu
	model = model.to(device)
	for state in optimizer.state.values():
		for key, value in state.items():
			if isinstance(value, torch.Tensor):
				state[key] = value.to(device)

#make a folder named with date and model name and data name to save the model and log
time = str(datetime.datetime.now())
log_dir = cwd + args.log_dir + "/" + time +"_"+ str(args.network) + "_" + str(args.data) + "_" + str(args.noise_level)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
#writer = SummaryWriter(args.log_dir)
noise_dir = log_dir + "/noise"
if not os.path.exists(noise_dir):
	os.makedirs(noise_dir)
pred_dir = log_dir + "/pred"
if not os.path.exists(pred_dir):
	os.makedirs(pred_dir)
model_dir = log_dir + "/model"
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
print("model saved in {}".format(model_dir))
#writer = SummaryWriter(cwd + args.log_dir)

for epoch in range(0, args.epochs):
	scheduler.step(epoch)
	iterator_train = iter(data.DataLoader(data_train, 
									batch_size=args.batch_size, 
									num_workers=args.num_workers, 
									sampler = SubsetSampler(0, data_size)))

	# TRAINING LOOP
	print("\nEPOCH:{} of {} - starting training loop from iteration:0 to iteration:{}\n".format(epoch, args.epochs, iters_per_epoch))
	
	for i in tqdm(range(0, iters_per_epoch)):
		
		# Sets model to train mode
		model.train()

		# Gets the next batch of images
		image, mask, gt = [x.to(device) for x in next(iterator_train)]
		
		# Forward-propagates images through net
		# Mask is also propagated, though it is usually gone by the decoding stage
		output = model(image)

		loss_dict = loss_fn(image, mask, output, gt)
		loss = 0.0
		# loss_char = torch.sum([criterion_char(output[j],image) for j in range(len(output))])
		# loss_edge = torch.sum([criterion_edge(output[j],image) for j in range(len(output))])
		# loss = loss_char# + (0.05*loss_edge)
		# sums up each loss value
		for key, value in loss_dict.items():
			loss += value
		# if (i + 1) % args.log_interval == 0:
		# 	writer.add_scalar(key, value.item(), (epoch * iters_per_epoch) + i + 1)
		# 	writer.file_writer.flush()
		# Resets gradient accumulator in optimizer
		optimizer.zero_grad()
		# back-propogates gradients through model weights
		loss.backward()
		# updates the weights
		optimizer.step()

		
	filename = model_dir + "/model_{}.pth".format(str(args.network))
	state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
	torch.save(state, filename)
		#print("epoch {}, batch {}, loss:{}".format(epoch, i, loss.item()))
	model.eval()
	psnr_avg1 = []
	psnr_avg2 = []
	loss_test_all = []
	test_idx = 0
	l2 = []
	tv = []
	with torch.no_grad():
		for i, (image, mask, gt) in enumerate(val_loader):
			image, mask, gt = [x.to(device) for x in [image, mask, gt]]
			#print("image shape:", image.shape)
			output = model(image)
			#print("output shape:", output.shape)
			#calculate psnr
			psnr_avg1.append(psnr(output, gt))
			psnr_avg2.append(PSNR(output, gt))

			#print("psnr:", psnr[-1])
			loss_dict = loss_fn(image, mask, output, gt)
			loss_test = 0.0
			# l2.append(loss_dict["loss"].item())
			# tv.append(loss_dict["tv"].item())
			for key, value in loss_dict.items():
				loss_test += value
			loss_test_all.append(loss_test.item())
			# #save output and gt as images using torchvision.utils.save_image
			for j in range(output.shape[0]):
				torchvision.utils.save_image(output[j], pred_dir + "/pred_{}.png".format(test_idx))
				torchvision.utils.save_image(image[j], noise_dir + "/noise_{}.png".format(test_idx))
				test_idx += 1




			#print mean psnr of the whole test set
	print("test at epoch {}, mean_psnr1:{}, mean_psnr2:{}, mean_loss:{}".format(epoch, np.mean(psnr_avg1), np.mean(psnr_avg2), np.mean(loss_test_all)))
	#print("l2:{}, tv:{}".format(np.mean(l2), np.mean(tv)))

