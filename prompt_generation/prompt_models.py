import os
import requests
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import openai
import torch
import warnings
import traceback
import time

openai.api_key = os.getenv("OPENAI_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")
if together_api_key is None:
    warnings.warn("No Together API Key set, TogetherAPI models will not work")

device = "cuda" if torch.cuda.is_available() else "cpu"


class TogetherAPI:
    def __init__(self, model="llama-2-7b-chat", n_retries=3, wait_time=5):
        model_path_dict = {"llama-2-70b-chat": "togethercomputer/llama-2-70b-chat",
                           "llama-2-13b-chat": "togethercomputer/llama-2-13b-chat",
                           "llama-2-7b-chat": "togethercomputer/llama-2-7b-chat",
                           "platypus": "garage-bAInd/Platypus2-70B-instruct",
                           "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
                           "solar": "upstage/SOLAR-0-70b-16bit"}
        self.model = model
        if model not in model_path_dict:
            raise ValueError(f"Unrecognized Model: {model}")
        self.model_path = model_path_dict[model]
        self.n_retries = n_retries
        self.wait_time = wait_time

    def __call__(self, prompt, max_new_tokens=50, top_p=0, temperature=0, top_k=5):
        for i in range(self.n_retries):
            try:
                res = requests.post('https://api.together.xyz/inference', json={
                    "model": f"{self.model_path}",
                    "max_tokens": max_new_tokens,
                    "prompt": prompt,
                    "request_type": "language-model-inference",
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": 1,
                    "stop": [
                        "[/INST]",
                        "</s>",
                        "\n"
                    ],
                    "type": "chat",
                    "safety_model": "",
                    "repetitive_penalty": 1,
                }, headers={
                    "Authorization": f"Bearer {together_api_key}",
                })
                output = res.json()['output']['choices'][0]['text']
                return output
            except requests.exceptions.JSONDecodeError:
                warnings.warn(f"Got JSONDecode Error with prompt: {prompt}")
            else:
                traceback.print_exc()
                warnings.warn(f"Unknown Error. Look into this, input was: {prompt}")
            print(f"Sleeping...")
            time.sleep(self.wait_time)
        return -1


class OpenAIGPT:
    #model = "text-davinci-003"
    model = "gpt-3.5-turbo-instruct"
    #model = "gpt-4"
    #model = "gpt-3.5-turbo"
    #model = "davinci"
    seconds_per_query = (60 / 20) + 0.01
    max_tokens=120
    @staticmethod
    def request_model(prompt):
        return openai.Completion.create(model=OpenAIGPT.model, prompt=prompt, max_tokens=OpenAIGPT.max_tokens)

    @staticmethod
    def request_chat_model(msgs):
        messages = []
        for message in msgs:
            role, content = message
            messages.append({"role": role, "content": content})
        return openai.ChatCompletion.create(model=OpenAIGPT.model, messages=messages)

    @staticmethod
    def decode_response(response):
        if OpenAIGPT.is_chat():
            return response["choices"][0]["message"]["content"]
        else:
            return response["choices"][0]["text"]

    @staticmethod
    def query(prompt):
        return OpenAIGPT.decode_response(OpenAIGPT.request_model(prompt))

    @staticmethod
    def chat_query(msgs):
        return OpenAIGPT.decode_response(OpenAIGPT.request_chat_model(msgs))

    @staticmethod
    def is_chat():
        return OpenAIGPT.model in ["gpt-4", "gpt-3.5-turbo"]

    @staticmethod
    def __call__(inputs, max_new_tokens=50):
        OpenAIGPT.max_tokens = max_new_tokens
        if OpenAIGPT.is_chat():
            return OpenAIGPT.chat_query(inputs)
        else:
            return OpenAIGPT.query(inputs)


class HuggingFaceModel:
    def __init__(self, model_name_or_path, model_class=AutoModel, model_max_length=250):
        self.model = model_class.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=model_max_length)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = self.model.eval()

    def query(self, prompt, max_new_tokens=3):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]

    def __call__(self, prompt, max_new_tokens=3):
        return self.query(prompt, max_new_tokens=max_new_tokens)


class T5(HuggingFaceModel):
    def __init__(self, size="large"):
        assert size in ["small", "base", "large", "xl", "xxl"]
        model_name = f"google/flan-t5-{size}"
        HuggingFaceModel.__init__(self, model_name, model_class=AutoModelForSeq2SeqLM)


class Llama2(HuggingFaceModel):
    def __init__(self, size="7"):
        assert size in ["7", "13",  "70", 7, 13, 30]
        model_name = f"meta-llama/Llama-2-{size}b-hf"
        HuggingFaceModel.__init__(self, model_name, model_class=AutoModelForCausalLM)


class Alpaca(HuggingFaceModel):
    def __init__(self, size="gpt4-xl"):
        assert size in ["base", "large",  "gpt4-xl", "xl", "xxl"]
        model_name = f"declare-lab/flan-alpaca-{size}"
        HuggingFaceModel.__init__(self, model_name, model_class=AutoModelForSeq2SeqLM)


class MPTInstruct(HuggingFaceModel):
    def __init__(self, size="7"):
        assert size in ["7", "30", 7, 30]
        model_name = f"mosaicml/mpt-{size}b-instruct"
        HuggingFaceModel.__init__(self, model_name, model_class=AutoModelForCausalLM)


class FalconInstruct(HuggingFaceModel):
    def __init__(self, size="7"):
        assert size in ["7", "40", 7, 40]
        model_name = f"tiiuae/falcon-{size}b-instruct"
        HuggingFaceModel.__init__(self, model_name, model_class=AutoModelForCausalLM)

    def query(self, prompt, max_new_tokens=3):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        input_length = inputs["input_ids"].shape[1]
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]
    

class MistralInstruct(HuggingFaceModel):
    def __init__(self):
        model_name = f"mistralai/Mistral-7B-Instruct-v0.1"
        HuggingFaceModel.__init__(self, model_name, model_class=AutoModelForCausalLM)

    def query(self, prompt, max_new_tokens=3):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        input_length = inputs["input_ids"].shape[1]
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)[0]


class PromptModel:
    def __init__(self, model_name, score=False, coT=False, warn=True, verbose=False):
        if model_name in ["gpt-3.5-turbo", "text-davinci-003", "gpt-3.5-turbo-instruct"]:
            OpenAIGPT.model = model_name
            self.model = OpenAIGPT()
        elif "t5" in model_name:
            size = model_name.split("-")[1]
            self.model = T5(size=size)
        elif "mistral_local" in model_name:
            self.model = MistralInstruct()
        elif "mistral" in model_name:
            self.model = TogetherAPI(model=model_name)
        else:
            self.model = TogetherAPI(model=model_name)
        self.score = score
        self.coT = coT
        self.warn = warn
        self.verbose = verbose

    def __call__(self, prompt):
        output = self.model(prompt)
        if self.verbose:
            print(f"With Input: {prompt}\nOutput: {output}")
        if self.coT:
            if output.count("|") != 1 and self.warn:
                warnings.warn(f"Expected exactly one | in output: {output}")
            else:
                output = output.split("|")[1]
        if output == -1:
            if self.warn:
                warnings.warn(f"Error for input: {prompt}")
            return -1
        if self.score:
            for option in range(10, -1, -1):
                if str(option) in output:
                    return option / 10
            if self.warn:
                warnings.warn(f"Could not find integer score in output: {output}")
            if "yes" in output.lower() or "true" in output.lower():
                return 0.7
            elif "no" in output.lower() or "false" in output.lower():
                return 0.3
            else:
                return None
        else:
            output = output.lower()
            true = "true" in output
            false = "false" in output
            if true and false:
                if self.warn:
                    warnings.warn(f"Both true and false found for input: {prompt}\nOutput: {output}")
                return None
            if not true and not false:
                if self.warn:
                    warnings.warn(f"Neither true or false found for input: {prompt}\nOutput: {output}")
                return None
            else:
                return true
