#!/bin/bash
#SBATCH --job-name=Abeer_job4
#SBATCH --cpus-per-task=12
#SBATCH --time=55:00:00          
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --output=sbatch_output/test_sbatch_output_4.txt  # %j (job_id) %x(job_name)    
#SBATCH --partition=leinegpu        # only for GPU jobs 
#SBATCH --gres=gpu:1                    # only for GPU jobs
#SBATCH --mail-type=ALL                   
#SBATCH --mail-user=mansour.abeer@mh-hannover.de

# node the job ran on + empty line
echo "Job ran on:" $(hostname)
echo "" 

python Hypergraph_classification.py --data_name "UWaveGestureLibrary"

wait

echo ""
echo "Job finished"  

# job will finish automatically as soon as the script has finished running, regardless of the specified job time.