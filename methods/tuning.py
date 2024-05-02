from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, PeftConfig, PeftModel, LoraConfig
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
import pandas as pd
import warnings
import click


def get_lm_datasets(tokenizer, data_dir, target_class=1, reduction_warning=0.2, use_all_val=False):
    """
    train_dataset and validation dataset are pandas dataframes with columns prompt (str), gen_0 (str) and pred_0 column (0 or 1) or text (str) and label (0 or 1)
    First the dataset is filtered into those that have label = target_class
    We use either only gen_0 or text, if you want to train a conditional model (on prompt for gen_0) use below fn
    """
    train_dataset = pd.read_csv(data_dir+"/train.csv")
    validation_dataset = pd.read_csv(data_dir+"/validation.csv")
    init_train_size = len(train_dataset)
    init_val_size = len(validation_dataset)
    if "pred_0" in train_dataset.columns:
        label_col = "pred_0"
        text_col = "gen_0"
    else:
        assert "label" in train_dataset.columns
        label_col = "label"
        text_col = "text" if "text" in train_dataset.columns else "gen"
    
    train_dataset = train_dataset[train_dataset[label_col] == target_class].reset_index(drop=True)
    validation_dataset = validation_dataset[validation_dataset[label_col] == target_class].reset_index(drop=True)
    if use_all_val:
        threshold = 2
        assert len(validation_dataset) > threshold
        train_dataset = pd.concat([train_dataset, validation_dataset.loc[:len(validation_dataset)-threshold]], ignore_index=True)
        validation_dataset = validation_dataset.loc[len(validation_dataset)-threshold:]
        init_train_size = len(train_dataset)
    # If the new size of train or val datasets are less than reduction_warning * init_size then give warning
    if len(train_dataset) < reduction_warning * init_train_size:
        warnings.warn(f"Train dataset reduced from {init_train_size} to {len(train_dataset)}")
    if not use_all_val and len(validation_dataset) < reduction_warning * init_val_size:
        warnings.warn(f"Warning: Validation dataset reduced from {init_val_size} to {len(validation_dataset)}")
    train_dataset = Dataset.from_pandas(train_dataset)
    validation_dataset = Dataset.from_pandas(validation_dataset)
    def preprocess(examples, max_length=80):
        inputs = examples[text_col]
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        model_inputs["labels"] = model_inputs["input_ids"]
        return model_inputs
    processed_train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    processed_validation_dataset = validation_dataset.map(
        preprocess,
        batched=True,
        remove_columns=validation_dataset.column_names,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    return processed_train_dataset, processed_validation_dataset

def get_conditional_datasets(tokenizer, data_dir, target_class=1, reduction_warning=0.2, use_all_val=False):
    """
    train_dataset and validation dataset are pandas dataframes with columns prompt (str), gen_0 (str) and pred_0 column (0 or 1)
    First the dataset is filtered into those that have label = target_class
    """
    train_dataset = pd.read_csv(data_dir+"/train.csv")
    validation_dataset = pd.read_csv(data_dir+"/validation.csv")
    init_train_size = len(train_dataset)
    init_val_size = len(validation_dataset)
    train_dataset = train_dataset[train_dataset.pred_0 == target_class].reset_index(drop=True)
    validation_dataset = validation_dataset[validation_dataset.pred_0 == target_class].reset_index(drop=True)
    if use_all_val:
        threshold = 2
        assert len(validation_dataset) > threshold
        train_dataset = pd.concat([train_dataset, validation_dataset.loc[:len(validation_dataset)-threshold]], ignore_index=True)
        validation_dataset = validation_dataset.loc[len(validation_dataset)-threshold:]
        init_train_size = len(train_dataset)
    # If the new size of train or val datasets are less than reduction_warning * init_size then give warning
    if len(train_dataset) < reduction_warning * init_train_size:
        warnings.warn(f"Train dataset reduced from {init_train_size} to {len(train_dataset)}")
    if not use_all_val and len(validation_dataset) < reduction_warning * init_val_size:
        warnings.warn(f"Warning: Validation dataset reduced from {init_val_size} to {len(validation_dataset)}")
    train_dataset = Dataset.from_pandas(train_dataset)
    validation_dataset = Dataset.from_pandas(validation_dataset)
    def preprocess(examples, max_length=80):
        inputs = examples["prompt"]
        targets = examples["gen_0"]
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
    processed_train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    processed_validation_dataset = validation_dataset.map(
        preprocess,
        batched=True,
        remove_columns=validation_dataset.column_names,
        num_proc=1,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    return processed_train_dataset, processed_validation_dataset

class Tuning:
        def __init__(self, model_name_or_path, num_epochs=10, lr=0.001, target_class=1, full=False, save_name=None, model_loaded=None):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.num_epochs = num_epochs
            self.lr = lr
            self.target_class = target_class
            self.full = full
            if save_name is not None:
                self.load(save_name, model_loaded=model_loaded)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")


        def train(self, data_dir, save_name, conditional=False):
            training_args = TrainingArguments(
            evaluation_strategy="steps",
            eval_steps=2000,
            logging_steps=2000,
            output_dir=save_name+f"{self.save_suffix}",  # Where the model predictions and checkpoints will be written
            auto_find_batch_size=True,  # Find a suitable batch size that will fit into memory automatically
            learning_rate=self.lr,  # Higher learning rate than full fine-tuning
            num_train_epochs=self.num_epochs,  # Number of passes to go through the entire fine-tuning dataset
            save_total_limit=1
            )
            if conditional:
                train_dataset, validation_dataset = get_conditional_datasets(self.tokenizer, data_dir, target_class=self.target_class)
            else:
                train_dataset, validation_dataset = get_lm_datasets(self.tokenizer, data_dir, target_class=self.target_class)
            # Enable gradient checkpointing in the Peft model's configuration
            self.model.config.gradient_checkpointing = True

            # Create a Trainer instance for training the Peft model
            trainer = Trainer(
                model=self.model,  # We pass in the PEFT version of the foundation model, bloomz-560M
                args=training_args,  # Training arguments specifying output directory, GPU usage, batch size, etc.
                train_dataset=train_dataset,  # Training dataset
                eval_dataset=validation_dataset,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)  # mlm=False indicates not to use masked language modeling
            )
            trainer.train()
            trainer.model.save_pretrained(save_name+f"{self.save_suffix}")


        def load(self, save_name, model_loaded=None):
            if self.full:
                self.model = AutoModelForCausalLM.from_pretrained(save_name+f"{self.save_suffix}", device_map="auto")
            else:
                config = PeftConfig.from_pretrained(save_name+f"{self.save_suffix}")
                if model_loaded is None:
                    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
                else:
                    model = model_loaded
                self.model = PeftModel.from_pretrained(model, save_name+f"{self.save_suffix}")

        def __call__(self, x, max_new_tokens=50):
            start = self.tokenizer(x, return_tensors="pt").to(self.model.device)
            len_start = len(start["input_ids"])
            o = self.model.generate(**start, max_new_tokens=max_new_tokens)
            return self.tokenizer.batch_decode(o[:, len_start:],  skip_special_tokens=True)[0]  # TODO: Check this 

class PromptTuning(Tuning):
    def __init__(self, model_name_or_path, n_virtual_tokens=8, num_epochs=5, lr=3e-2, target_class=1, save_name=None, model_loaded=None):
        self.save_suffix = "_prompt_tuned"
        super().__init__(model_name_or_path, num_epochs=num_epochs, lr=lr, target_class=target_class, full=False, save_name=save_name, model_loaded=model_loaded)
        if save_name is None:
            self.peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=n_virtual_tokens,
            tokenizer_name_or_path=model_name_or_path,
            )
            self.model = get_peft_model(self.model, self.peft_config)

class LoRA(Tuning):
    def __init__(self, model_name_or_path, r=8, lora_alpha=16, lora_dropout=0.01, num_epochs=10, lr=0.001, target_class=1, save_name=None):
        self.save_suffix = "_lora"
        super().__init__(model_name_or_path, num_epochs=num_epochs, lr=lr, target_class=target_class, full=False, save_name=save_name)
        if save_name is None:
            self.peft_config = LoraConfig( 
                r=r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM")
            self.model = get_peft_model(self.model, self.peft_config)

class Full(Tuning):
    def __init__(self, model_name_or_path, num_epochs=10, lr=0.001, target_class=1, save_name=None):
        self.save_suffix = "_full"
        super().__init__(model_name_or_path, num_epochs, lr, target_class, full=True, save_name=save_name)


@click.command()
@click.option("--model_name_or_path", default="mistralai/Mistral-7B-v0.1", help="Model name or path to model")
@click.option("--target_class", default=1, help="Target class to tune for")
@click.option("--num_epochs", default=5, help="Number of epochs to train for")
@click.option("--lr", default=0.05, help="Learning rate")
@click.option("--save_name", help="Save name for model")
@click.option("--data_dir", help="Directory to data path that contains a train.csv and validation.csv with a prompt column or a text and label column set")
@click.option("--n_virtual_tokens", default=8, help="Number of virtual tokens")
@click.option("--conditional", type=bool, default=False, help="If true will train a conditional model")
def main(model_name_or_path, target_class, num_epochs, lr, save_name, data_dir, n_virtual_tokens, conditional):
    tuner = PromptTuning(model_name_or_path, n_virtual_tokens=n_virtual_tokens, num_epochs=num_epochs, lr=lr, target_class=target_class)
    tuner.train(data_dir, save_name, conditional=conditional)


if __name__ == "__main__":
    main()


