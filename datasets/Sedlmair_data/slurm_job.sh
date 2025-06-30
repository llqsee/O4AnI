#!/bin/bash
#SBATCH --job-name=scatter-mpi
#SBATCH --output=logs/scatter_%j.out
#SBATCH --error=logs/scatter_%j.err
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

module load gcc/14.2.0
module load openmpi/5.0.6.gcc-14.2.0

# Just use the system default MPI plugin
srun python ./datasets/Sedlmair_data/scatterplot_generator_mpi.py
