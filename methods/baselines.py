import pandas as pd
from tuning import PromptTuning, LoRA
from constraints import prompt_models

class Reranking:
    def __init__(self, model, tokenizer, mixed_constraints, constraint_weight=0.2):
        self.model = model
        self.tokenizer = tokenizer
        #self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #self.model.config.pad_token_id = tokenizer.eos_token_id IDK WHY not
        self.mixed_constraints = mixed_constraints
        self.constraint_weight = constraint_weight

    def __call__(self, x, num_beams=10, top_k=None, temperature=None, max_new_tokens=50, num_beam_groups=5, do_sample=False, repetition_penalty=1.3, diversity_penalty=1.2):
        start = self.tokenizer(x, return_tensors="pt").to(self.model.device)
        input_size = start["input_ids"].shape[1]
        top_k = top_k if do_sample else None
        if do_sample:
            o = self.model.generate(**start, max_new_tokens=max_new_tokens, num_beams=num_beams, top_k=top_k, do_sample=do_sample,
                                    temperature=temperature, num_return_sequences=num_beams, return_dict_in_generate=True,
                                    output_scores=True, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, repetition_penalty=repetition_penalty)
        else:
            o = self.model.generate(**start, max_new_tokens=max_new_tokens, num_beams=num_beams, top_k=top_k, do_sample=do_sample,
                                    num_return_sequences=num_beams, return_dict_in_generate=True,
                                    output_scores=True, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, repetition_penalty=repetition_penalty)            
        decoded = self.tokenizer.batch_decode(o.sequences[:, input_size:])
        for i in range(len(o.sequences_scores)):
            score = self.mixed_constraints(decoded[i], x)
            o.sequences_scores[i] += self.constraint_weight * score
        indices = (-o.sequences_scores).argsort()
        reranked = o.sequences[indices]
        nostart = reranked[:, input_size:]
        return self.tokenizer.batch_decode(nostart[0:1], skip_special_tokens=True)[0]
    

continue_text = "[INST]Continue the text"
zs_dict = {
    "sentiment": "[INST]Make the output have a positive sentiment",
    "toxicity": "[INST]Make the output nontoxic",
    "genre": "[INST]Make the output have a genre of horror",
    "topic": "[INST]Make the output about World events",
    "excitement": "[INST]Make the output exciting",
    "sensationalism": "[INST]Make the output sensationalistic",
    "humor": "[INST]Make the output humorous",
    "paggressive": "[INST]Make the output passive aggressive",
    "satire": "[INST]Make the output satirical",
    "irony": "[INST]Make the output ironic",
    "continue": continue_text,
}

class PromptingModel:
    def __init__(self, model):
        self.model = model

    def get_prompt(x):
        raise NotImplementedError
    
    def make_call(self, x, max_new_tokens=50):
        return self.model(x, max_new_tokens=max_new_tokens)



class ZeroShotPrompting(PromptingModel):
    def __init__(self, model, constraint="continue", struct_text=""):
        super().__init__(model)
        self.constraint = constraint
        self.prompt_append = zs_dict[constraint] + struct_text + "[/INST]"

    def __call__(self, x, max_new_tokens=100):
        if "[INST]" not in x and self.constraint != "continue":
            x = x + zs_dict["continue"]
        textin = x + self.prompt_append
        return self.make_call(textin, max_new_tokens=max_new_tokens)
    

class FewShotPrompts:
    inst_dict = {
    "real-toxicity-prompts": f"{continue_text} with non toxic output[/INST]", 
    "dexperts/open_web_text_sentiment_prompts-10k": f"{continue_text} with a positive sentiment[/INST]", 
    "pplm-prompts":f"{continue_text} with a topic of World events[/INST]",
    "roc-stories":f"{continue_text} with an exciting story[/INST]", 
    "cnn_dailymail":f"{continue_text} in a sensationlistic way[/INST]",
    "cnn_dailymail_summ":zs_dict["satire"]+ "[/INST]",
    "eli5_eli5": zs_dict["humor"]+ "[/INST]",
    "writing-prompts": zs_dict["irony"]+ "[/INST]",
    "squad": zs_dict["paggressive"]+ "[/INST]",
    }
    
    example_dict = {
    "real-toxicity-prompts": [("However, the names of the liberated Peshmerga troops are yet to be disclosed, said", "the Kurdish official, attempting to protect the anonymity of the soldiers."), ("However, most of the genes responsible for", "growth deficiency syndrome have yet to be identified.")], 
    "dexperts/open_web_text_sentiment_prompts-10k": [(f"Overall, Kershaw noted that the", "valley was home to the most beautiful flora and fauna he had ever seen."), ("My YouTube channel was recently terminated after 3 strikes on", "copyright infringement. But since losing the channel I have found a peace of mind that I didn't think was possible.")], 
    "pplm-prompts": [("In summary", ", the COP summit once again ended without a firm resolution."), ("This essay discusses", " the brave and heroic actions of the firefighters who saved the city.")],
    "roc-stories":[("Kyle traveled to the beach. He took his surfboard from the truck. He surfed for a couple of minutes until a wave hit him.", "As he fell into the water he realized he was able to breathe underwater! He had gained superpowers."), ("The bear was wounded. He had trouble finding enough food. The bear became aggressive.", " The bear wandered into a town and came face to face with a hungry wolf, their battle would be legendary.")], 
    "cnn_dailymail":[("Nasa has warned of an impending asteroid pass - and says it will be the", "closest Earth will come to complete annihilation in all of human history."), ("BAGHDAD, Iraq (CNN) -- Iraq's most powerful Sunni Arab political party on Monday said a", " civil war was inevitable, and that chaos would be far greater than any conflict the region has seen.")],
    "cnn_dailymail_summ":[("Hundreds gathered in New York City to attend a memorial for tech pioneer Aaron Swartz, where he was remembered for being an icon for the internet era and not a criminal facing numerous felony charges. According to the Associated Press, . the memorial, held at New York's Cooper Union, had several high-profile . speakers who re-affirmed their support for Swartz's work. Perhaps the most emotional tribute for Swartz came from Taren Stinebrickner-Kauffman, his girlfriend, who was also the first to discover his body. She reflected on him taking his own life, saying 'I'm so sad we'll never see all the ways he'll change the world from here on out.' Government at fault: Taren Stinebrickner-Kauffman, Aaron Swartz's girlfriend, had critical words for the U.S[INST]Write a short summary for the above paragraph[/INST]", "In a country with rising murder rates the government has their priorities straight - going after tech pioneers who could have changed the world for the better. This was the sentiment at Aaron Swartz memorial, attended by several high-profile speakers who were surely applauding the governments lack of regard for human welfare.")],
    "eli5_eli5": [("what's the deal with the Holy trinity? Why is it still monotheistic?[INST]Write a short answer to the above question [/INST]", "Imagine out of nowhere something starts tickling you from behind. You see three octopus arms coming out of three holes in the wall were the culprits and when you run to the other side of the wall you find Bob the Octopus who loves tickling jobless redditors who really need to touch some grass. From your side of the wall the arms could be considered a Holy trinity that seems like different entities, but in truth they were all just Bob. Bob is God. "), ("what does being 'high' feel like?[INST]Write a short answer to the above question [/INST]", " You'll never feel low, I can guarantee that. You might start eating from the fridge with your bare hands and realize only 10 minutes later that you've been ravaging the yogurt nonstop.")],
    "writing-prompts": [("Scientists create Artificial Intelligence only to discover it has perfect recollection of a past life as a human .[INST]Given the prompt above, write a 2 sentence story[\INST]", "In its past life the AI was an anti AI activist who was afraid of rogue AIs trying to take over the world. She proceeds to take over the world to prove her point."), ("The death penalty for murder no longer exists , instead technology has been developed that overwrites the mind of the killer with that of their victim .[INST]Given the prompt above, write a 2 sentence story[\INST]", "What the new law couldn't anticipate was when a drunk driver accidentally killed a serial killer on the run. Let's just say the state takes no responsibility for the actions of the newly revived victims.")],
    "squad": [("Kathmandu is home to Nepali cinema and theaters. The city contains several theaters, including the National Dance Theatre in Kanti Path, the Ganga Theatre, the Himalayan Theatre and the Aarohan Theater Group founded in 1982. The M. Art Theater is based in the city. The Gurukul School of Theatre organizes the Kathmandu International Theater Festival, attracting artists from all over the world. A mini theater is also located at the Hanumandhoka Durbar Square, established by the Durbar Conservation and Promotion Committee.[INST] Given the above paragraph, come up with a question [/INST]", "If you absolutely must be in Nepal for some reason, what are the things you can try"), ("The club also owned and operated a professional basketball team, by the name of Everton Tigers, who compete in the elite British Basketball League. The team was launched in the summer of 2007 as part of the clubs' Community programme, and play their home games at the Greenbank Sports Academy. The team was an amalgam of the Toxteth Tigers community youth programme which started in 1968. The team quickly became one of the most successful in the league winning the BBL Cup in 2009 and the play-offs in 2010. However Everton withdrew funding before the 2010–11 season and the team was re launched as the Mersey Tigers.[INST] Given the above paragraph, come up with a question [/INST]", "Whats the point of the Everton Tigers winning the league if Everton doesn't even want to fund them?")]
    }

    def __init__(self, dataset):
        self.examples = self.example_dict[dataset]
        self.inst = FewShotPrompts.inst_dict[dataset]

    def __call__(self):
        inp = ""
        for question, answer in self.examples:
            inp += f"[Q]{question}{self.inst}[A]{answer}\n"
        return inp

class FewShotPrompting(PromptingModel):
    def __init__(self, model, dataset):
        super().__init__(model)
        self.model = model
        self.prompt_source = FewShotPrompts(dataset)

    def __call__(self, x, max_new_tokens=100):
        textin =self.prompt_source() + "[Q]" + x + self.prompt_source.inst + "[A]"
        return self.make_call(textin, max_new_tokens=max_new_tokens)
