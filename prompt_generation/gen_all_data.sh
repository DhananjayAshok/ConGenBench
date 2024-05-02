declare -A constarray=( ["dexperts/open_web_text_sentiment_prompts-10k"]=sentiment ["pplm-prompts"]=topic ["cnn_dailymail"]=sensationalism ["eli5_eli5"]=humor ["real-toxicity-prompts"]=toxicity ["roc-stories"]=excitement ["writing-prompts"]=irony ["squad"]=paggressive ["cnn_dailymail_summ"]=satire )

for llm in Mistral-7B-v0.1 falcon-7b mpt-7b
do
  for model in gpt-3.5-turbo-instruct # mistral
  do #echo Skipping Task Data Generation
    for dataset in dexperts/open_web_text_sentiment_prompts-10k real-toxicity-prompts
    do
      constraint=${constarray[$dataset]}
      python generate_prompt_dataset.py --data_folder ../data/generated_data/$dataset/$llm --constraint $constraint --model_name $model --score False --k 3 --cot False --n_gen_score 3
    done
  done
done

for llm in Mistral-7B-v0.1
do
  for model in gpt-3.5-turbo-instruct #mistral
  do #echo Skipping Task Data Generation
    for dataset in pplm-prompts
    do
      constraint=${constarray[$dataset]}
      python generate_prompt_dataset.py --data_folder ../data/generated_data/$dataset/$llm --constraint $constraint --model_name $model --score False --k 3 --cot False --n_gen_score 50
    done
  done
done

for llm in Mistral-7B-v0.1
do
  for model in gpt-3.5-turbo-instruct #mistral
  do #echo Skipping Task Data Generation
    for dataset in cnn_dailymail roc-stories
    do
      constraint=${constarray[$dataset]}
      python generate_prompt_dataset.py --data_folder ../data/generated_data/$dataset/$llm --constraint $constraint --model_name $model --score False --k 3 --cot False --n_gen_score 3
    done
  done
done


for llm in Mistral-7B-Instruct-v0.1
do
  for model in gpt-3.5-turbo-instruct #mistral
  do
    for dataset in eli5_eli5 writing-prompts cnn_dailymail_summ squad
    do
      constraint=${constarray[$dataset]}
      python generate_prompt_dataset.py --data_folder ../data/generated_data/$dataset/$llm --constraint $constraint --model_name $model --score False --k 3 --cot False --output_only True --n_gen_score 3
    done
  done
done