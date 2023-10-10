#!/bin/bash
#SBATCH -J ip ### job name
#SBATCH -o ip.o%j ### output and error file name (%j expands to jobID)
#SBATCH --mail-user=hzhao25@cougarnet.uh.edu
####SBATCH --mem-per-cpu=32g ### How much memory in megabytes per cpu that you are using, Max is approximately 8800.

#SBATCH --mail-type=all #### email me when the job starts (changes its status on the queue
#### and gets the need resources resources),
#### or when the job fails or when it finishes

#SBATCH -p volta -t 96:00:00 -N 1 -n 8 --gres=gpu:1  ### time allocated to your job, num cpus, and num gpus to be used
### Tell SLURM which account (advisor) to charge this job to
#SBATCH -A labate


hostname
#env
#nvidia-smi
#module add cudatoolkit/10.2
#module add python/3.7 ### activate the python environment you need
#module add OpenCV/4.2.0-fosscuda-2019b-Python-3.7.4
#module add TensorFlow/2.3.2-fosscuda-2019b-Python-3.7.4
#### In my case i need pytorch (latest version

#module add anaconda3/python-3.8 
module add cudatoolkit/11.6
module add torchvision/0.15.2-foss-2022a-CUDA-11.7.0
module add PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module add OpenCV/4.5.3-fosscuda-2021a-Python-3.8.2
module add opencv-python
module add matplotlib
module add scikit-learn
module add tqdm
python places2_train.py