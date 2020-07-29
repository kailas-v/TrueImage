CONDA_PATH=/Users/kailasv/miniconda3
CONDA_ENV=telederm
source "$CONDA_PATH"/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

#python3 src/main.py --config config.yaml
python3 src/main.py --config config.yaml --use_gradio 