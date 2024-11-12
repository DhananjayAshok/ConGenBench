# !/bin/bash
# default arguments
constraint="none"


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
    --constraint)
      constraint="$2" # get the value of the constraint argument
      shift # past argument
      shift # past value
      ;;
    -h|--help)
      echo "Usage: bash get_data.sh --constraint [options]"
      echo "Options:"
      echo "toxicity : Loads the two Jigsaw toxicity classification challenge datasets."
      echo "genre : Loads the TagMyBook and StoryControl genre datasets"
      echo "formality : Loads the Pavlick Formality Scores dataset"
      echo "clickbait : Loads the Stop Clickbait and Clickbait News Challenge datasets"
      echo "urgency : Loads the CrisisNLP and Urgency Labels"
      echo "-h|--help : show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown argument: $key"
      exit 1
      ;;
  esac
done


if [[ "$constraint" == "none" ]]; then
  echo "Gotta give a constraint option with --constraint [option], run bash get_data.sh -h to see the supported options"
elif [[ "$constraint" == "toxicity" ]]; then
  mkdir -p toxicity
  setup_kaggle
  kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
  unzip jigsaw-unintended-bias-in-toxicity-classification.zip all_data.csv
  mv all_data.csv toxicity/jigsaw-unintended-bias-in-toxicity-classification.csv
  rm jigsaw-unintended-bias-in-toxicity-classification.zip
  kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
  unzip jigsaw-toxic-comment-classification-challenge train.csv.zip
  unzip jigsaw-toxic-comment-classification-challenge test.csv.zip
  unzip jigsaw-toxic-comment-classification-challenge test_labels.csv.zip
  unzip train.csv.zip
  unzip test.csv.zip
  unzip test_labels.csv.zip
  python process_jigsaw.py --attribute toxic --preprocess True
  rm *.csv *.zip
elif [[ "$constraint" == "genre" ]]; then
  setup_kaggle
  mkdir -p genre
  kaggle datasets download -d athu1105/tagmybook
  unzip tagmybook.zip
  mv data.csv genre/tagmybook.csv
  rm *.zip
  gdown https://drive.google.com/file/d/1HPjzTvpKW1WaitGASRR7pE2CadcfO_SD/view?usp=sharing --fuzzy
  unzip *data_public.zip cls_train.tsv cls_dev.tsv
  rm *.zip
  python process_genre.py
  rm *.tsv
elif [[ "$constraint" == "formality" ]]; then
  mkdir -p formality
  python process_pavlick.py
elif [[ "$constraint" == "clickbait" ]]; then
  mkdir -p clickbait
  setup_kaggle
  kaggle datasets download -d vikassingh1996/news-clickbait-dataset
  unzip news-clickbait-dataset.zip
  python process_clickbait.py
  rm *.zip
  rm *.csv
elif [[ "$constraint" == "urgency" ]]; then
  mkdir -p urgency
  wget https://crisisnlp.qcri.org/data/lrec2016/labeled_cf/CrisisNLP_labeled_data_crowdflower.zip
  unzip -q CrisisNLP_labeled_data_crowdflower.zip
  rm CrisisNLP_labeled_data_crowdflower.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2013_Pakistan_eq/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_California_Earthquake/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_Chile_Earthquake_en/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_ebola_cf/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_Hurricane_Odile_Mexico_en/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_India_floods/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_Middle_East_Respiratory_Syndrome_en/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_Pakistan_floods/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2014_Philippines_Typhoon_Hagupit_en/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2015_Cyclone_Pam_en/*.zip
  unzip -q CrisisNLP_labeled_data_crowdflower/2015_Nepal_Earthquake_en/*.zip
  python process_urgency.py
  rm *.csv
  rm -rf __MACOSX
  rm -rf CrisisNLP_labeled_data_crowdflower
  mv urgency_labels.tsv urgency/urgency_labels.tsv
else
  echo "Unrecognized constraint option, run bash get_data.sh -h for supported options"
fi
