ml load Python/3.10.4-GCCcore-11.3.0
pip install torch torchvision torchaudio
pip install pandas matplotlib numpy patool scikit-learn sktime sympy tqdm


pip install torch_geometric
pip uninstall torch-scatter torch-sparse torch-geometric torch-cluster  --y
pip install torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__}.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
pip install torch-householder


srun --job-name=Train --cpus-per-task=12 --mem=64G --time=24:00:00 --gres=gpu:1 --partition=leinegpu --pty /bin/bash

python run.py --model_id test_metr --data_name METR-LA --horizon 6