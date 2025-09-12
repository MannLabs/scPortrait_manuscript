#!/bin/sh

#SBATCH --job-name=ViTMAE_training
#SBATCH -o /home/icb/alioguz.can/projects/scPortrait4i/slurm_outputs/slurm_%j.txt ## UPDATE PATHS
#SBATCH -e /home/icb/alioguz.can/projects/scPortrait4i/slurm_outputs/slurm_error_%j.txt ## UPDATE PATHS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:2
#SBATCH --qos=gpu_long
#SBATCH --time=71:59:00
#SBATCH --nice=10000
#SBATCH --exclude=gpusrv26,gpusrv27,gpusrv28,gpusrv29,gpusrv30,gpusrv31,gpusrv32,gpusrv33,gpusrv34,gpusrv35,gpusrv38,gpusrv39,gpusrv40,gpusrv41,gpusrv42,gpusrv43,gpusrv44,gpusrv45,gpusrv46,gpusrv47,gpusrv48,gpusrv49,gpusrv50,gpusrv51,gpusrv52,gpusrv54,gpusrv57

source $HOME/.bashrc
chmod 600 $HOME/projects/scPortrait4i/slurm_outputs/slurm_$SLURM_JOB_ID.txt ## UPDATE PATHS

# Activate the correct conda environment using the passed ENV_NAME
conda deactivate
conda activate scportrait ## UPDATE ENV NAME

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
USER_COMMENT="${1:-No comment provided for this experiment}"
START_TIME=$(date +%s)

echo "==================== Job Started ===================="
echo "User Comment   : $USER_COMMENT"
echo "Start time     : $(date)"
echo "Job ID         : $SLURM_JOB_ID"
echo "Job Name       : $SLURM_JOB_NAME"
echo "Node List      : $SLURM_NODELIST"
echo "Num. of CPUs   : $SLURM_CPUS_ON_NODE"
echo "Num. of GPUs   : $NUM_GPUS"
echo "====================================================="

python /home/icb/alioguz.can/projects/scPortrait_manuscript/data_processing/VITMAE_Training_Xenium/train_xenium_ovarian_cancer.py --finetune 1 ## UPDATE PATHS

echo "==================== Job Finished ==================="
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "End time       : $(date)"
echo "Duration       : $(printf '%02dh:%02dm:%02ds\n' $(($DURATION/3600)) $(($DURATION%3600/60)) $(($DURATION%60)))"

# usage:
# sbatch /home/icb/alioguz.can/projects/scPortrait4i/scripts/run_train.sbatch ## UPDATE PATHS