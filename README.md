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
Code can be run using 
```
python multimat.py --data_path PATH_TO_DATA_DIR --modalities_encoders 
crystal dos --exp NAME_OF_EXPERIMENT
``` 
The above example performs MultiMat with two modalities, the crystal structure and the DOS. To toggle 
between different modalities see the important flags below.

#### Important Flags:
--modalities_encoder: list all the modalities to use during MultiMat training. Available modalities are 
`crystal`, `dos`, `charge_density`, `text`.

--mask_non_int: creates a mask so that all pairwise losses will be accounted for in the batch. 
Missing entries for the DOS and charge_density modality will have losses set to zero. (only applicable 
when using 3 or more modalities)  

### Downstream tasks
All downstream tasks can be found in the `tasks` folder 

#### Property Prediction
We fine-tune the crystal encoder pretrained with MultiMat for property prediction
```
python tasks/prediction_finetune.py --data_path PATH_TO_DATA_DIR --decoder_task bulk_modulus 
--checkpoint_to_finetune PATH_TO_PRETRAINED_MULTIMAT_CHECKPOINT --exp NAME_OF_EXPERIMENT
```
the flag --decoder_task can be one of the following [`bulk_modulus`, `shear_modulus`, `elastic_tensor`, 
`bandgap`]. For bandgap prediction on the SNUMAT database, the data path should be changed to that of 
snumat data (e.g. ./example_data/snumat_data instead of ./example_data)

 #### Retrieval

To perform cross-modality retrieval using a pre-trained MultiMAt model (and reproduce the corresponding figure from the paper), run 
```
python tasks/retrieval.py --checkpoint_to_load PATH_TO_PRETRAINED_MULTIMAT_CHECKPOINT
```

#### Material Discovery via Latent Space Similarity

To perform latent spaced-based material discovery using a pre-trained MultiMAt model (and reproduce the corresponding figures from the paper), run 
```
python tasks/material_discovery.py --checkpoint_to_load PATH_TO_PRETRAINED_MULTIMAT_CHECKPOINT
```

#### Interpretability of crystal embeddings

To interpret the crystal embeddings of a pre-trained MultiMAt model (and reproduce the corresponding figures from the paper), run 
```
python tasks/interpretability.py --checkpoint_to_load PATH_TO_PRETRAINED_MULTIMAT_CHECKPOINT
```
