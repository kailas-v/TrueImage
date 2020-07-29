# -- Edit path below !! --
CONDA_PATH=/Users/kailasv/miniconda3



# TODO: create anaconda environment if not created

CONDA_ENV=telederm
source "$CONDA_PATH"/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# -- Make directory for outside repos --
OUTSIDE_REPOS=outside_repos
mkdir -p "$OUTSIDE_REPOS"
if [ ! -d "$OUTSIDE_REPOS"/SemanticSegmentation ]; then
    cd "$OUTSIDE_REPOS"
    git clone https://github.com/WillBrennan/SemanticSegmentation.git
fi

#pip install albumentations
#pip install tensorboard
#pip install pytorch-ignite