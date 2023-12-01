## Implementation of Posterior Sampling with Diffusion Models for MR Image Reconstruction

## Getting started

First install the conda environment requirements

```.bash
conda env create -f environment.yml
```

activate conda environment

```.bash
conda activate edm
```
## RGB Image Experiments
Download model weights from: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/. Specifically, download "edm-afhqv2-64x64-uncond-ve.pkl".

Download an example 64x64 image from the validation parition of the AFHQ dataset: https://utexas.app.box.com/v/utcsilab-data/folder/237821411608.
 

Next, in "inference_general_photo.py" change the path for (1) the model weights and (2) the test data .pt file to their respective locations on your local machine.

To run the compressed sensing task run the following command
```.bash
python inference_general_photo.py --task compsens --R 4
```
if you then run the results viewer notbook you should see results that look similar to the following

![front_page_sample](figures/CS_readme_ex.png)

To run the inpainting task run the following command
```.bash
python inference_general_photo.py --task inpaint --R 4
```
if you then run the results viewer notbook you should see results that look similar to the following
![front_page_sample](figures/IP_readme_ex.png)


