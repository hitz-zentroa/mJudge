import copy
import json

import torch
from torch.utils.data import Dataset

system_prompt_en="You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
system_prompt_es="Eres un evaluador imparcial encargado de proporcionar comentarios claros y objetivos basados en criterios específicos, asegurando que cada evaluación refleje los estándares absolutos establecidos para el desempeño."
system_prompt_eu="Ebaluatzaile inpartziala zara, irizpide zehatzetan oinarrituta iritzi argiak eta objektiboak emateko ardura duena, ebaluazio bakoitzak errendimendurako ezarritako estandar absolutuak islatzen dituela bermatuz."

class FeedbackDataset(Dataset):
    def __init__(self, dataset_config, tokenizer,train_proportion=1.0, split="train"):
        if split == "train":
            self.file=dataset_config.train_split
            self.ann = json.load(open(dataset_config.train_split))
            train_length=int(len(self.ann)*train_proportion)
            self.ann = self.ann[:train_length]
        else:
            self.file=dataset_config.test_split
            self.ann = json.load(open(dataset_config.test_split))
        self.tokenizer = tokenizer
        self.lang = dataset_config.lang

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        if self.lang.split== "mono_es":
            system_prompt = system_prompt_es
        elif self.lang.split == "mono_eu":
            system_prompt = system_prompt_eu
        else:
            system_prompt = system_prompt_en

        ann = self.ann[index]
        prompt = system_prompt + " " + ann["instruction"]
        example = prompt + ann["output"]
        text=example
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "name":self.file,
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
