#!/bin/bash
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=15000
#SBATCH -J lighting_thesis    # name
#SBATCH -o hostname_%j.out # File to which STDOUT will be written
#SBATCH -e hostname_%j.err # File to which STDERR will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gp14958@my.bristol.ac.uk # Email to which notifications will be sent
module unload Python/2.7.11-foss-2016a
module load libs/cudnn/8.0-cuda-8.0
export PATH=$PATH:/mnt/storage/scratch/gp14958/blender-2.79-linux-glibc219-x86_64/
export SCENE_DIR=/mnt/storage/scratch/gp14958/scene_data
#source ~/Thesis/tensorflow/bin/activate

export PREFIX=b
python runner.py

