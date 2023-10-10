import torch
import numpy as np
import torch.nn as nn
from SDPF_net import SparseNet, SDPF
import matplotlib.pyplot as plt
#define a function to save all the learned conv filters
def save_filters(model_file, folder, network):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if network == "sparsenet":
	    model = SparseNet().to(device)
    elif network == "sdpf":
	    model = SDPF().to(device)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["model"])
    kernel_list = []
    weight_list = []
    for name, param in model.state_dict().items():
        if "kernels" in name:
            kernel_list.append(param)
        if "weight" in name:
            weight_list.append(param)
    for i in range(len(weight_list)):
        print(i)
        kernel_list[i] = torch.einsum("ijk, ijklm -> ijlm", weight_list[i], kernel_list[i].detach())
        print(kernel_list[i].size())
    print("kernel list shape:", kernel_list.shape)
model_file = "/project/labate/heng/inpainting-partial-conv/training_logs/2023-09-04 16:08:26.282329_sparsenet_places2_0.1/model/model_sparsenet.pth"
folder = "/project/labate/heng/inpainting-partial-conv/filters/test"
#save_filters(model_file, folder, "sdpf")
class Save_filter():
    def __init__(self, network, model_file, folder):
        super(Save_filter, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = network
        self.model_file = model_file
        self.folder = folder
        if network == "sparsenet":
            self.model = SparseNet().to(device)
        elif network == "sdpf":
            self.model = SDPF().to(device)
    def save(self):
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint["model"])
        span_layer = []
        weight_layer = []

        #save spaned kernels
        
        for name in self.model.state_dict():
            if "kernels" in name:
                span_layer.append(name.split(".")[0]+".0")
        for param in span_layer:
            conv_kernel = torch.einsum("ijk, ijklm -> ijlm", self.model.state_dict()[param+".weight"], self.model.state_dict()[param+".kernels"].detach())
            conv_kernel = conv_kernel.cpu().numpy()
            shape = conv_kernel.shape
            print(shape)
            #save the conv kernel as images with same small intervals
            for k in range(shape[1]):
                num_ker = shape[0]
                fig, ax = plt.subplots(int(num_ker/8), 8, figsize=(int(num_ker/8), 8))
                for i in range(int(num_ker/8)):
                    for j in range(8):
                        ax[i, j].imshow(conv_kernel[i*8+j,k,:,:], cmap='gray')
                        ax[i, j].axis('off')
                fig.tight_layout(pad=0.5)
                plt.savefig(self.folder + "/" + param + "_" + str(k) + "th" + ".png")

        #save conv kernels
        for name in self.model.state_dict():
            if name.split(".")[0]+".0" not in span_layer and "weight" in name:
                weight_layer.append(name.split(".")[0]+".0")
        for param in weight_layer:
            conv_kernel = self.model.state_dict()[param+".weight"].cpu().numpy()
            shape = conv_kernel.shape
            if shape[0] == 1:
                conv_kernel = conv_kernel.transpose(1, 0, 2, 3)
                shape = conv_kernel.shape
            print(shape)
            #save the conv kernel as images with same small intervals
            for k in range(shape[1]):
                num_ker = shape[0]
                fig, ax = plt.subplots(int(num_ker/8), 8, figsize=(int(num_ker/8), 8))
                for i in range(int(num_ker/8)):
                    for j in range(8):
                        ax[i, j].imshow(conv_kernel[i*8+j,k,:,:], cmap='gray')
                        ax[i, j].axis('off')
                fig.tight_layout(pad=0.5)
                plt.savefig(self.folder + "/" + param + "_" + str(k) + "th" + ".png")

save_filters = Save_filter("sparsenet", model_file, folder)
save_filters.save()







