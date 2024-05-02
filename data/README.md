To start, fetch the data using:
```sh
bash fetch_all_data.sh
```
This script will load all data sources, if you want to be more selective you can look inside the file and comment out the datasets you aren't interested in.

Next, split the data into train, validation and test sets. 
```sh
bash split_all_data.sh
```
This following command will also load the remaining datasets that already exist on [HuggingFace](https://huggingface.co/docs/datasets/en/index), once again take a look inside and comment out the datasets you would rather not load and split.

Finally, if you want to use it, generate the greedy decoding samples on the task datasets (either as a baseline, for tuning or for the Synthetic Dataset Creation Algorithm mentioned in the paper. This can be done using the command
```sh
python create_generation_data.py --model_name_or_path $model --data_dir task_data/$taskdata --max_points $max_points --n_gens $n_gens
```
The model parameter should be a HuggingFace model, max_points determines how many points get generated, and n_gens determines how many generations are made for each prompt. See [an example](data/gen_all_data.sh) for a better understanding
