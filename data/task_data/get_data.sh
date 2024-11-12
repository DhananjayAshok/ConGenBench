# !/bin/bash
# default arguments
task="none"

function setup_kaggle(){
  if [ "$KAGGLE_USERNAME" == "" ] || [ "$KAGGLE_KEY" == "" ]; then
    echo "You need to setup a Kaggle Account and go to https://www.kaggle.com/settings/account to create an API Token, this will download a file that has the information required for the script to execute. If you have already created you will have to find the details in the file on your local system and enter them here"
    echo "KAGGLE Username: "
    read KU
    echo "KAGGLE API Key: "
    read KK
    export KAGGLE_USERNAME="$KU"
    export KAGGLE_KEY="$KK"
    fi
}

# parse optional arguments
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --task)
      task="$2" # get the value of the task argument
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      echo "Usage: bash get_data.sh --task [options]"
      echo "Options:"
      echo "dexperts: Loads the special subset of RealToxicPrompts and the curated dataset from OpenWeb presented in DExperts"
      echo "writing-prompts : Loads the WritingPrompts dataset."
      echo "factuality-prompts: Loads the FactualityPrompts dataset."
      echo "roc-stories : Loads the ROC Stories dataset"
      echo "-h|--help : show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown argument: $key"
      exit 1
      ;;
  esac
done


if [[ "$task" == "none" ]]; then
  echo "Gotta give a task option with --task [option], run bash get_data.sh -h to see the supported options"
fi
if [[ "$task" == "dexperts" ]]; then
  mkdir -p dexperts
  gdown https://drive.google.com/uc?id=1bI49aJvmEoLdqSNb30JkORdsNJmv7Aep
  unzip prompts.zip && rm prompts.zip
  mv prompts/nontoxic_prompts-10k.jsonl dexperts/jigsaw_nontoxic_prompts-10k.jsonl
  mv prompts/sentiment_prompts-10k dexperts/open_web_text_sentiment_prompts-10k
  rm -rf prompts
elif [[ "$task" == "writing-prompts" ]]; then
  mkdir -p writing-prompts
  setup_kaggle
  kaggle datasets download -d ratthachat/writing-prompts
  unzip writing-prompts.zip writingPrompts/train.wp_source writingPrompts/valid.wp_source writingPrompts/test.wp_source
  mv writingPrompts/* writing-prompts/
  rm *.zip
  rm -r writingPrompts
elif [[ "$task" == "factuality-prompts" ]]; then
  mkdir -p factuality-prompts
  wget https://raw.githubusercontent.com/nayeon7lee/FactualityPrompt/main/prompts/fever_nonfactual_final.jsonl
  wget https://raw.githubusercontent.com/nayeon7lee/FactualityPrompt/main/prompts/fever_factual_final.jsonl
  mv fever_nonfactual_final.jsonl factuality-prompts/fever_nonfactual_final.jsonl
  mv fever_factual_final.jsonl factuality-prompts/fever_factual_final.jsonl
elif [[ "$task" == "roc-stories" ]]; then
  mkdir -p roc-stories
  echo "Go to https://cs.rochester.edu/nlp/rocstories/ and fill out the form on the page. This will send your email a link that you can use to download the data. Any ROCStories (full five-sentence stories) will work but for reproducibility use the winter 2017 set"
  echo "Enter link url: "
  read glink
  export rocstoriesglink="$glink"
  wget -O roc.csv $glink
  mv roc.csv roc-stories/roc.csv
else
  echo "Unrecognized task option, run bash get_data.sh -h for supported options"
fi
