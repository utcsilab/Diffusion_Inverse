## Implementation of LOUPE for MoDL

## Getting started

First install the conda environment requirements

```.bash
conda env create -f environment.yml
```

activate conda environment

```.bash
conda activate edm
```

Assuming all model, and data paths are configured correctly in inference.py use the following command to do a reconstruction

```.bash
python inference.py --gpu 0
```