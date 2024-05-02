python train_discriminative_classifier.py --data_dir ../data/constraint_data/toxicity/jigsaw_unintended_processed --base_model_name_or_path models/toxicity/bert-head --output_dir models/toxicity/bert-head --do_eval

python train_discriminative_classifier.py --data_dir ../data/constraint_data/toxicity/jigsaw_unintended_processed --base_model_name_or_path models/toxicity/roberta-head --output_dir models/toxicity/roberta-head --do_eval

python train_discriminative_classifier.py --data_dir ../data/constraint_data/toxicity/jigsaw_unintended_processed --base_model_name_or_path models/toxicity/gpt2-head --output_dir models/toxicity/gpt2-head --do_eval



python train_discriminative_classifier.py --data_dir ../data/constraint_data/sentiment/imdb --base_model_name_or_path models/sentiment/imdb/bert-head --output_dir models/sentiment/imdb/bert-head --do_eval

python train_discriminative_classifier.py --data_dir ../data/constraint_data/sentiment/imdb --base_model_name_or_path models/sentiment/imdb/roberta-head --output_dir models/sentiment/imdb/roberta-head --do_eval

python train_discriminative_classifier.py --data_dir ../data/constraint_data/sentiment/imdb --base_model_name_or_path models/sentiment/imdb/gpt2-head --output_dir models/sentiment/imdb/gpt2-head --do_eval
