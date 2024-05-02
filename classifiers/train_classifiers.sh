# skipping imdb because done
for dataset in sentiment/yelp_polarity clickbait/clickbait_news_detection clickbait/stop_clickbait genre/storycontrol genre/tagmybook formality/pavlick spam/sms_spam spam/spamassassin toxicity/jigsaw_unintended_processed
do
  python train_discriminative_classifier.py --data_dir ../data/constraint_data/$dataset --base_model_name_or_path roberta-base --output_dir models/$dataset/roberta-vocab-of-gpt2 --do_train --overwrite_output_dir --save_total_limit 2 --finetune_strategy full --num_train_epochs 5 --embedding_strategy vocab --embedding_model_name_or_path gpt2-large
done
