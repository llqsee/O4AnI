#!/bin/bash
#SBATCH --job-name=scatter-mpi                  # Job name
#SBATCH --output=logs/scatter_%j.out            # Standard output log
#SBATCH --error=logs/scatter_%j.err             # Standard error log
#SBATCH --ntasks=100                            # Total number of MPI tasks
#SBATCH --cpus-per-task=1                       # CPU cores per task
#SBATCH --mem-per-cpu=4G                        # Memory per CPU
#SBATCH --time=02:00:00                         # Wall time limit


module load gcc/14.2.0
module load openmpi/5.0.6.gcc-14.2.0

# mpirun -np $SLURM_NTASKS python datasets/simulated_datasets/mpi_scatterplot_generator.py


srun python ./datasets/simulated_datasets/mpi_scatterplot_generator.py
