declare -A constarray=( ["dexperts/open_web_text_sentiment_prompts-10k"]=sentiment ["pplm-prompts"]=topic ["cnn_dailymail"]=sensationalism ["eli5_eli5"]=humor ["real-toxicity-prompts"]=toxicity ["roc-stories"]=excitement ["writing-prompts"]=irony ["squad"]=paggressive ["cnn_dailymail_summ"]=satire )
promptmodelconfig=_score_False_k_3_coT_False_author_None_max_points_None_seed_42
base_model=roberta-base
model_name=roberta-classifier
save_total_limit=2
finetune_strategy=full
num_train_epochs=5

for model in gpt-3.5-turbo-instruct #llama-2-70b-chat #platypus solar
do
  for llm in Mistral-7B-v0.1 falcon-7b mpt-7b Mistral-7B-Instruct-v0.1
  do    
    for dataset in dexperts/open_web_text_sentiment_prompts-10k bookcorpus cnn_dailymail eli5_eli5 real-toxicity-prompts roc-stories writing-prompts cnn_dailymail_summ squad
    do
      #echo Skipping
      constraint=${constarray[$dataset]}
      data_dir=../data/generated_data/$dataset/$llm/$constraint/"$model"$promptmodelconfig
      output_dir=models/$dataset/$llm/$constraint/"$model"$promptmodelconfig/$model_name
      #python ../prompt_generation/unravel_csv.py --data_dir $data_dir
      #python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir $output_dir --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      #rm -rf "$data_dir"_unravel

      #python ../prompt_generation/unravel_csv.py --data_dir $data_dir --fudge True
      #python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir "$output_dir"-fudge --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      #rm -rf "$data_dir"_unravel
    done
  done
done

for model in gpt-3.5-turbo-instruct #llama-2-70b-chat #platypus solar
do
  for llm in Mistral-7B-v0.1 falcon-7b mpt-7b
  do    
    for dataset in dexperts/open_web_text_sentiment_prompts-10k real-toxicity-prompts
    do
      constraint=${constarray[$dataset]}
      data_dir=../data/generated_data/$dataset/$llm/$constraint/"$model"$promptmodelconfig
      output_dir=models/$dataset/$llm/$constraint/"$model"$promptmodelconfig/$model_name
      python ../prompt_generation/unravel_csv.py --data_dir $data_dir
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir $output_dir --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel

      python ../prompt_generation/unravel_csv.py --data_dir $data_dir --fudge True
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir "$output_dir"-fudge --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel
    done
  done
done

for model in gpt-3.5-turbo-instruct
do
  for llm in Mistral-7B-v0.1
  do    
    for dataset in pplm-prompts
    do
      constraint=${constarray[$dataset]}
      data_dir=../data/generated_data/$dataset/$llm/$constraint/"$model"$promptmodelconfig
      output_dir=models/$dataset/$llm/$constraint/"$model"$promptmodelconfig/$model_name
      python ../prompt_generation/unravel_csv.py --data_dir $data_dir
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir $output_dir --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel

      python ../prompt_generation/unravel_csv.py --data_dir $data_dir --fudge True
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir "$output_dir"-fudge --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel
    done
  done
done



for model in gpt-3.5-turbo-instruct #llama-2-70b-chat #platypus solar
do
  for llm in Mistral-7B-v0.1
  do    
    for dataset in cnn_dailymail roc-stories
    do
      constraint=${constarray[$dataset]}
      data_dir=../data/generated_data/$dataset/$llm/$constraint/"$model"$promptmodelconfig
      output_dir=models/$dataset/$llm/$constraint/"$model"$promptmodelconfig/$model_name
      python ../prompt_generation/unravel_csv.py --data_dir $data_dir
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir $output_dir --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel

      python ../prompt_generation/unravel_csv.py --data_dir $data_dir --fudge True
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir "$output_dir"-fudge --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel
    done
  done
done


for model in gpt-3.5-turbo-instruct #llama-2-70b-chat #platypus solar
do
  for llm in Mistral-7B-Instruct-v0.1
  do    
    for dataset in writing-prompts squad cnn_dailymail_summ eli5_eli5
    do
      constraint=${constarray[$dataset]}
      data_dir=../data/generated_data/$dataset/$llm/$constraint/"$model"$promptmodelconfig
      output_dir=models/$dataset/$llm/$constraint/"$model"$promptmodelconfig/$model_name
      python ../prompt_generation/unravel_csv.py --data_dir $data_dir
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir $output_dir --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel

      python ../prompt_generation/unravel_csv.py --data_dir $data_dir --fudge True
      python train_discriminative_classifier.py --data_dir "$data_dir"_unravel --base_model_name_or_path $base_model --output_dir "$output_dir"-fudge --do_train --overwrite_output_dir --save_total_limit $save_total_limit --finetune_strategy $finetune_strategy --num_train_epochs $num_train_epochs
      rm -rf "$data_dir"_unravel
    done
  done
done