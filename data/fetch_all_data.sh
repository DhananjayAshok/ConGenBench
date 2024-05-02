cd constraint_data
bash get_data.sh --constraint toxicity
bash get_data.sh --constraint genre
bash get_data.sh --constraint clickbait
bash get_data.sh --constraint formality
bash get_data.sh --constraint urgency

cd ../task_data
bash get_data.sh --task writing-prompts
bash get_data.sh --task factuality-prompts
bash get_data.sh --task dexperts
bash get_data.sh --task roc-stories

cd ..