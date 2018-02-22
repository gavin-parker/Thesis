#!/bin/bash
#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=15000
#SBATCH --account=comsm0018       # use the course account
#SBATCH -J adl_cwk    # name
#SBATCH -o hostname_%j.out # File to which STDOUT will be written
#SBATCH -e hostname_%j.err # File to which STDERR will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gp14958@my.bristol.ac.uk # Email to which notifications will be sent
module load libs/tensorflow/1.2

python main.py --dematerial --log-dir='/mnt/storage/scratch/gp14958/dm_logs' --batch-size=16 --test-model-dir='best_dematerial/model-1' --learning-rate=50e-6 --train-dir='/mnt/storage/scratch/gp14958/MultiNatIllum/data/multiple_materials_single_object/singlets/' --max-epochs=1