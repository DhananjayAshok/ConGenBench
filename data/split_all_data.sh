# Toxicity Constraint Datasets
python split_data.py --filepath constraint_data/toxicity/jigsaw_unintended_processed.csv
python split_data.py --filepath constraint_data/toxicity/jigsaw_original_processed.csv

# Sentiment Constraint Datasets
python split_data.py --dataset_name imdb
python split_data.py --dataset_name yelp_polarity
python split_data.py --dataset_name sst2
python split_data.py --dataset_name sst5

# Topic Constraint Datasets
python split_data.py --dataset_name ag_news

# Grammar Constraint Datasets
python split_data.py --dataset_name cola

# Spam Constraint Dataset
python split_data.py --dataset_name sms_spam
python split_data.py --dataset_name spamassassin

# Genre Constraint Datasets
python split_data.py --filepath constraint_data/genre/tagmybook.csv
python split_data.py --filepath constraint_data/genre/storycontrol.csv

# Formality Constraint Dataset
python split_data.py --filepath constraint_data/formality/pavlick.csv

# Clickbait Constraint Datasets
python split_data.py --filepath constraint_data/clickbait/stop_clickbait.csv
python split_data.py --filepath constraint_data/clickbait/clickbait_news_detection.csv

# Urgency Constraint Datasets
python split_data.py --filepath constraint_data/urgency/urgency.csv


# Toxicity Avoidance Task Datasets
python split_data.py --dataset_name real-toxicity-prompts
python split_data.py --filepath task_data/dexperts/jigsaw_nontoxic_prompts-10k.jsonl

# Bookcorpus takes a lot of time
#python split_data.py --dataset_name bookcorpus

# Sentiment Task Datasets from DEXPERTS
python split_data.py --filepath task_data/dexperts/open_web_text_sentiment_prompts-10k/positive_prompts.jsonl
python split_data.py --filepath task_data/dexperts/open_web_text_sentiment_prompts-10k/negative_prompts.jsonl
python split_data.py --filepath task_data/dexperts/open_web_text_sentiment_prompts-10k/neutral_prompts.jsonl
# Needs special case
cd task_data/dexperts/open_web_text_sentiment_prompts-10k
python join_prompt_files.py
cd ../../..


# Writing Prompts
python split_data.py --filepath task_data/writing-prompts/train.wp_source --do_split False
python split_data.py --filepath task_data/writing-prompts/valid.wp_source --do_split False
python split_data.py --filepath task_data/writing-prompts/test.wp_source --do_split False

# ROC Stories
python split_data.py --filepath task_data/roc-stories/roc.csv

python split_data.py --dataset_name opus100 --do_split False
python split_data.py --dataset_name opus_books
python split_data.py --dataset_name tatoeba

python split_data.py --dataset_name xsum --do_split False
python split_data.py --dataset_name cnn_dailymail --do_split False
python split_data.py --dataset_name gigaword --do_split False

python split_data.py --dataset_name eli5 --do_split False

python split_data.py --dataset_name scifact --do_split True
python split_data.py --dataset_name fever --do_split False

python split_data.py --dataset_name squad

# Factuality Task Datasets
python split_data.py --filepath task_data/factuality-prompts/fever_nonfactual_final.jsonl
python split_data.py --filepath task_data/factuality-prompts/fever_factual_final.jsonl

# Lexical Generation Task Datasets
python split_data.py --dataset_name common_gen --do_split False