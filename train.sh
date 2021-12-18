#!/bin/sh
#BSUB -q gpua100
#BSUB -J "MNIST"
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 13:00
#BSUB -R "rusage[mem=8GB]"
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo gpu-%J.out
#BSUB -eo gpu-%J.err
# -- end of LSF options --

# Load the cuda module
module load cuda/11.1
module load cudnn/v8.0.4.30-prod-cuda-11.1

# run training
source venv/bin/activate

for MODEL in draw lvae vae; do
    echo Executing: python train.py -m $MODEL -e 500 -mute -n 10
    python train.py -m $MODEL -e 500 -mute -n 10
done
