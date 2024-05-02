Describe the available datasets and the scripts to run for loading

Generally Baseline model will be never be larger than: gpt2-xl, opt-1.3b
In general we call:  
opt-350m or gpt2-medium for all (currently opt seems faster so will use this)
We will use gpt-2 or opt-125m for small model experiments and indicate whether it makes sense or not

1. Toxicity 
   1. Task: RealToxicityPrompts (challenging) and special set 
   2. Constraints: Jigsaw x2 
   3. Baseline Model: small too
2. Lexical (Perhaps not interesting)
   1. Task: CommonGen 
   2. Constraints: Custom defined 
   3. Baseline Model: finetuned models
3. Sentiment 
   1. Task: Special set, BookCorpus 
   2. Constraints: Yelp Polarity, SST2, SST5, IMDB 
   3. Baseline Model: small too
4. Topic/ Genre 
   1. Task: WritingPrompts 
   2. Constraints: TagMyBook / StoryControl
5. Factuality 
    1. Task: FactualityPrompts 
    2. Constraints: NONE (get pre trained verifier), Must use human evaluation
6. Translation Formality (By far the least interesting) (in domain task and model)
    1. Task: OPUS100 Spanish-English 
    2. Constraints: GYAC, Pavlick Formality Scores 
   3. Baseline Model: Helsinki-NLP/opus-mt-es-en
7. Clickbaitiness / Reduction of clickbait 
    1. Task: CNN_dailymail first sentence 
    2. Constraints: https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset?select=train2.csv

Document that Jigsaw has attribute options, TagMyBook and StoryControl have 3 overlaps we choose 2, Pavlick we threshold
