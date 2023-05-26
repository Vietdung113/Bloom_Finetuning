import pandas as pd

import torch
import transformers
from transformers import BloomTokenizerFast, BloomForCausalLM, TrainingArguments
from datasets import Dataset, load_dataset
from utils import ModifiedTrainer, tokenise_data, data_collator
from utils import ModelArguments, DataArguments


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args, data_args = parser.parse_args_into_dataclasses()

    print(model_args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_name = model_args.model_name_or_path

    tokeniser = BloomTokenizerFast.from_pretrained(
        f"{model_name}", add_prefix_space=True
    )
    model = BloomForCausalLM.from_pretrained(f"{model_name}").to(device)

    dataset = load_dataset('csv', data_files={'train': data_args.train_data})

    input_ids = tokenise_data(dataset, tokeniser)

    model.gradient_checkpointing_enable()
    model.is_parallelizable = True
    model.model_parallel = True

    # Training hyper parameter
    training_args = TrainingArguments(
        "output",
        fp16=True,
        gradient_accumulation_steps= 1,
        per_device_train_batch_size = 1,
        learning_rate = 2e-5,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy='epoch',
        metric_for_best_model='loss',
    )

    # train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=input_ids,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()