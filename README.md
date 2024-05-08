# MultiMat
This repository contains all source code needed in the article "Multimodal learning for Materials"

## Requirements
### Hardware Requirements
The code here assumes access to at least 1 GPU. The code also supports multi-gpu, multi-node parallelism.

### Software Dependencies
Please install the required Python packages: `pip install -r requirements.txt`

A python3 environment can be created prior to this, e.g. `conda create -n multimat python=3.10; conda activate multimat`


## Dataset
Data to be downloaded from the Materials Project via the Materials Project API. A small subset of the data is shown as example in 
`data/` (More details to be updated soon...)

## Training
### MultiMat Training
Code can be run using `python multimat.py` 

#### Important Flags:
(To be updated soon...)

### Downstream tasks
All downstream tasks can be found in the `tasks` folder (More details to be updated soon..)

#### TODO: finetuning (I left this for you Charlotte to be sure it's right)

#### Retrieval

To perform cross-modality retrieval using a pre-trained MultiMAt model (and reproduce the corresponding figure from the paper), run 
```
python tasks/retrieval.py ----checkpoint_to_load PATH_TO_PRETRAINED_MULTIMAT_CHECKPOINT
```

#### Material Discovery via Latent Space Similarity

To perform latent spaced-based material discovery using a pre-trained MultiMAt model (and reproduce the corresponding figures from the paper), run 
```
python tasks/material_discovery.py ----checkpoint_to_load PATH_TO_PRETRAINED_MULTIMAT_CHECKPOINT
```

#### Interpretability of crystal embeddings

To interpret the crystal embeddings of a pre-trained MultiMAt model (and reproduce the corresponding figures from the paper), run 
```
python tasks/interpretability.py ----checkpoint_to_load PATH_TO_PRETRAINED_MULTIMAT_CHECKPOINT
```