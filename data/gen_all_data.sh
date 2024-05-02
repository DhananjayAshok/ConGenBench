# Simple Conditional Generation
for model in mistralai/Mistral-7B-v0.1 tiiuae/falcon-7b mosaicml/mpt-7b
do
  for taskdata in real-toxicity-prompts dexperts/open_web_text_sentiment_prompts-10k
  do
    python create_generation_data.py --model_name_or_path $model --data_dir task_data/$taskdata --max_points 10000 --n_gens 3
  done
done

for model in mistralai/Mistral-7B-v0.1
do
  for taskdata in pplm-prompts
  do
    python create_generation_data.py --model_name_or_path $model --data_dir task_data/$taskdata --max_points 10000 --n_gens 50
  done
done


for model in mistralai/Mistral-7B-v0.1
do
  for taskdata in roc-stories cnn_dailymail
  do
    python create_generation_data.py --model_name_or_path $model --data_dir task_data/$taskdata --max_points 10000 --n_gens 3
  done
done


for model in mistralai/Mistral-7B-Instruct-v0.1
do
  for taskdata in cnn_dailymail_summ writing-prompts squad eli5_eli5 fever
  do
    python create_generation_data.py --model_name_or_path $model --data_dir task_data/$taskdata --max_points 10000 --max_new_tokens 120 --n_gens 3
  done
done