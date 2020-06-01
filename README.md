
## Instructions to setup training infrastructure
1. Install required python packages from requirements.txt file.
2. Place the input data images and json file as per samples in the repository.
3. Path for Json file is defined in `config_2.py` file and `dataset_500.json` contains relative paths for the input images.
4. Surrounding stream input size can be modified in `config_2.py` file through variables `image_size_w_p` & `image_size_h_p`, default values are 96,48 respectively.
5. Current data is splitted in 5 sets beforehand and Train:Validation:Test split is 2:1:2.
6. Data pre-processing parallelization can be controlled through `max_workers` parameter in `siamese_two_stream.py`, default = 10.
7. `suffix` and `model_name` parameters from `siamese_two_stream.py` can be modified to change the saved model names.

## Commands
### Train
`python siamese_two_stream.py train`

### Test
`python siamese_two_stream.py test models-car-96-96/`
## Test resuls on [Vehicle Reid dataset](https://arxiv.org/abs/1911.05541)
* **Combined Results** - 
  * **Precision:94.56  | Recall:91.08 | F-score:92.62**

## Model Architecture and input example
![Current Model](https://github.com/adich23/Object-Reidentification/blob/master/Surrounding-Siamese.png)
