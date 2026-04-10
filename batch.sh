#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#### IMPORTANT check which account and partition you can use
#### on the machine you are running on (you can use the 'myallocation' command)
cd ~/mtgHPC
#Run the program with our input
# nvcc kmean.cu -o cuda
nvcc kmeanChatGPT.cu -o cuda
./cuda