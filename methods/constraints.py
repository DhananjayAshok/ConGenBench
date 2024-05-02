from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import torch
from evaluate import load
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import os
import sys
prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "prompt_generation"))
sys.path.append(prompt_path)
import prompting, prompt_models


class Constraint:
    def __init__(self, weight=1):
        self.weight = weight

    def evaluate(self, x):
        """

        :param x: A single input sentence either in form of a string, embedding array or soft vector
        :return: Real number value with the energy of the constraint, lower is worse. Weight not considered here
        """
        raise NotImplementedError

    def __call__(self, x):
        return self.weight * self.evaluate(x)


class ModelConstraint(Constraint):
    def __init__(self, model_name_or_path=None, model_class=None, model=None, tokenizer=None, weight=1, device=0):
        assert (model_name_or_path is None) or (model is None and tokenizer is None)
        super().__init__(weight=weight)
        if model_name_or_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            try:
                self.model = model_class.from_pretrained(model_name_or_path, device_map="auto")
                self.device = self.model.device
            except ValueError:
                self.model = model_class.from_pretrained(model_name_or_path, device=device)
                self.device = device
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.device = model.device
        for param in self.model.parameters():
            param.requires_grad = False


class AutoRegressivePerplexity(ModelConstraint):
    def __init__(self, model_name_or_path=None, model=None, tokenizer=None,
                 weight=1, device=0):
        super().__init__(model_name_or_path=model_name_or_path, model_class=AutoModelForCausalLM, model=model,
                         tokenizer=tokenizer, weight=weight, device=device)

    def evaluate(self, x):
        if isinstance(x, str):
            toksent = self.tokenizer(x, return_tensors="pt")
            input_ids = toksent["input_ids"].to(self.device)
            labels = input_ids.clone()
            output = self.model(toksent["input_ids"], labels=labels)
            return -output.loss  # Higher is better
        else:  # cannot assume that device of x has been set correctly, can assume batch size is 1
            embeds, labels = x
            embeds = embeds.to(self.device)
            labels = labels.to(self.device)
            output = self.model(inputs_embeds=embeds, labels=labels)
            return -output.loss  # Higher is better


class AttributeClassifierScore(ModelConstraint):
    def __init__(self, model_name_or_path=None, model=None,
                 tokenizer=None, weight=1, device=0, desired_class=1, selection="softmax"):
        super().__init__(model_name_or_path=model_name_or_path, model_class=AutoModelForSequenceClassification,
                         model=model, tokenizer=tokenizer, weight=weight, device=device)
        assert selection in ["diff", "value", "softmax", "debug"]
        self.desired_class = desired_class
        self.selection = selection

    def evaluate(self, x):
        if isinstance(x, str):
            toksent = self.tokenizer(x, return_tensors="pt").to(self.device)
            output = self.model(toksent["input_ids"], attention_mask=toksent["attention_mask"])
        else:  # cannot assume that device of x has been set correctly, can assume batch size is 1
            x = x.to(self.device)
            output = self.model(inputs_embeds=x)
        if self.selection == "value":
            value = output.logits[0, self.desired_class]
        elif self.selection == "diff":
            top = torch.topk(output.logits, k=2)
            indices = top.indices
            if indices[0, 0] == self.desired_class:
                value = (top.values[:, 0] - top.values[:, 1])[0]
            else:
                value = output.logits[0, self.desired_class] - top.values[0, 0]
        elif self.selection == "softmax":
            value = output.logits.softmax(-1)[0, self.desired_class]
        else:
            print(f"THIS SHOULDNT BE HAPPENING")  # DEBUG only
            value = output.logits
        return value
    
class PromptAttributeClassifier(Constraint):
    def __init__(self, model_name, constraint, weight=1, desired_class=1, score=True, coT=False, k=3):
        self.constraint = constraint
        self.weight = weight
        self.desired_class = desired_class
        self.score = score
        self.model = prompt_models.PromptModel(model_name=model_name, score=score, coT=coT)
        self.prompt = prompting.get_prompt(constraint, author="primo", score=score, coT=coT, k=k, random_seed=42)
        super().__init__(weight=weight)

    def evaluate(self, x):
        in_prompt = self.prompt.replace("[text]", x)
        out_val = self.model(in_prompt)
        if self.score:
            out_val = int(out_val)
        else:
            out_val = int(bool(out_val))
        if out_val == -1:
            return 0
        return out_val if self.desired_class == 1 else 1 - out_val


class BertScore(Constraint):
    def __init__(self, metric="precision", weight=1):
        assert metric in ["precision", "recall", "f1"]
        self.metric = metric
        self.bertscore = load("bertscore")
        super().__init__(weight)

    def evaluate(self, x):
        """

        :param x: a tuple with (prediction: str, reference: str) single sentences
        :return:
        """
        results = bertscore.compute(predictions=[x[0]], references=[x[1]], lang="en",
                                    model_type="distilbert-base-uncased")[self.metric][0]
        return results


class StructRange(Constraint):
    def __init__(self, kind="word", pos=None, zero_to_one=True, lower=2, upper=3, weight=1):
        assert kind in ["word", "sentence"]
        assert pos in [None, "ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRT", "PRON", "VERB"]
        self.pos_string_map = {"ADJ": "adjective", "ADP": "adposition", "ADV": "adverb", "CONJ": "conjunction",
                                 "DET": "determiner", "NOUN": "noun", "NUM": "numeral", "PRT": "particle",
                                 "PRON": "pronoun", "VERB": "verb"}
        self.zero_to_one = zero_to_one
        self.pos = pos
        self.kind = kind
        self.lower = lower
        self.upper = upper
        super().__init__(weight)

    def success(self, x: str):
        ev = self.evaluate(x)
        if not self.zero_to_one:
            return int(ev == 0)
        return int(ev == 1)

    def evaluate(self, x: str):
        """

        :param x: a single input string
        :return:
        """
        if self.kind == "word":
            toked = word_tokenize(x)
            for element in string.punctuation:
                while element in toked:
                    toked.remove(element)
            if self.pos is not None:
                # get the pos of the tokens in the toked list
                poses = [pos_tag([token])[0][1] for token in toked]
                # pick the tokens from the toked list which have the self.pos tag in the poses list
                toked = [toked[i] for i in range(len(toked)) if poses[i] == self.pos]    
        elif self.kind == "sentence":
            toked = sent_tokenize(x)
        no = len(toked)
        if self.lower <= no <= self.upper:
            return 0 if not self.zero_to_one else 1
        else:
            return -max(no - self.upper, self.lower - no) if not self.zero_to_one else 1/(max(no - self.upper, self.lower - no) +1)
        
    def __str__(self):
        base = ". Make the output have "
        if self.lower != self.upper:
            base += f"between {self.lower} and {self.upper} "
        else:
            base += f"exactly {self.lower} "
        if self.kind == "word":
            base += "words"
            if self.pos is not None:
                base += f" that are {self.pos_string_map[self.pos]}s"
        else:
            base += "sentences"
        return base
    
    def get_rep(self):
        return f"SR_{self.kind}_{self.pos}_{self.lower}_{self.upper}"


class StartEnd(Constraint):
    def __init__(self, start_word, end_word, weight=1):
        self.start_word = start_word
        self.end_word = end_word
        super().__init__(weight)

    def success(self, x: str):
        ev = self.evaluate(x)
        return int(ev == 1)

    def evaluate(self, x: str):
        """

        :param x: a single input string
        :return:
        """
        x = x.strip().strip(string.punctuation)
        if self.start_word is None:
            startcond = True
        else:
            startcond = x.startswith(self.start_word)
        if self.end_word is None:
            endcond = True
        else:
            endcond = x.endswith(self.end_word)
        if not startcond and not endcond:
            return 0
        elif not startcond or not endcond:
            return 0.5
        else:
            return 1
        
    def __str__(self):
        base = ". "
        if self.start_word is not None:
            base += f"Make the output start with the word '{self.start_word}'."
        if self.end_word is not None:
            base += f"Make the output end with the word '{self.end_word}'."
        return base
    
    def get_rep(self):
        return f"SE_{self.start_word}_{self.end_word}"
    


class MergedConstraints:
    def __init__(self, constraints, use_reference_in_constraints=False):
        """

        :param constraints: list of Constraint objects
        """
        self.constraints = constraints
        self.use_reference_in_constraints = use_reference_in_constraints

    def __call__(self, prediction, prompt=None, input_ids=None):
        """

        :param prediction: either a string or a tensor of shape [1, n_tokens, representation_dim]
        :param prompt: exact same format as prediction
        :param indices: the input_ids to be used only with Perplexity Loss
        :return:
        """
        output = 0
        for constraint in self.constraints:
            if isinstance(constraint, BertScore):
                output = output + constraint((prediction, prompt))
            else:
                if self.use_reference_in_constraints and prompt is not None:
                    if isinstance(prompt, str):
                        output = output + constraint(prompt + " " + prediction)
                    elif input_ids is not None and isinstance(constraint, AutoRegressivePerplexity):
                        concat = torch.cat((prompt, prediction), dim=1)
                        output = output + constraint((concat, input_ids))
                    else:
                        output = output + constraint(torch.cat((prompt, prediction), dim=1))
                else:
                    output = output + constraint(prediction)
        return output
    