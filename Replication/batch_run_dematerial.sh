#!/bin/bash
#SBATCH -t 1-00:00 # Runtime in D-HH:MM
#SBATCH --account=comsm0018
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=15000
#SBATCH -J dematerial    # name
#SBATCH -o hostname_%j.out # File to which STDOUT will be written
#SBATCH -e hostname_%j.err # File to which STDERR will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gp14958@my.bristol.ac.uk # Email to which notifications will be sent
module load Python/2.7.11-foss-2016a
module load libs/cudnn/8.0-cuda-8.0
source ~/Thesis/tensorflow/bin/activate

python main.py --dematerial --log-dir='/mnt/storage/scratch/gp14958/train_logs' --batch-size=16 --learning-rate=1.7e-9 --train-dir='/mnt/storage/scratch/gp14958/scene_data_experiment/renders' --val-dir='/mnt/storage/scratch/gp14958/scene_data_experiment/val' --max-epochs=100
