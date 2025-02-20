import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
from datasets import load_dataset

#https://huggingface.co/datasets/meta-math/MetaMathQA - potentially will use
#https://huggingface.co/datasets/allenai/math_qa - multiple choice
# data = load_dataset("allenai/math_qa", split="train[:5000]", trust_remote_code=True)
data = load_dataset("allenai/math_qa", split="train")
data = data.train_test_split(test_size=0.2)

#https://arxiv.org/abs/2309.12284

answers_dict = {
    "a":0,
    "b":1,
    "c":2,
    "d":3,
    "e":4
}


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def preprocess_function(examples):
    examples["label"] = answers_dict[examples["correct"]]
    temp = []
    options_array = examples["options"] + " "
    start_index = 0
    end_index = 0 
    for i in range(len(options_array)):
        if options_array[i] == ")":
            start_index = i
        elif options_array[i] == "," or i == len(options_array) - 1:
            end_index = i
            if i == len(options_array) - 1:
                temp.append(options_array[start_index + 2:end_index])
            else:
                temp.append(options_array[start_index + 2:end_index - 1])
    examples["answer choices"] = temp
    examples["answer_0"] = examples["answer choices"][0] 
    examples["answer_1"] = examples["answer choices"][1]
    examples["answer_2"] = examples["answer choices"][2]
    examples["answer_3"] = examples["answer choices"][3]
    examples["answer_4"] = examples["answer choices"][4]
    examples["filler"] = ""
    return examples
data = data.map(preprocess_function)
print(data["train"][0])

answers = ["answer_0", "answer_1", "answer_2", "answer_3", "answer_4"]

def combine(examples):
    first_sentences = [[question] * 5 for question in examples["Problem"]]
    question_headers = examples["filler"]
    second_sentences = [
        [f"{header} {examples[answer][i]}" for answer in answers] for i, header in enumerate(question_headers)
    ]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}

tokenized_math = data.map(combine, batched=True)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import os


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
    
import evaluate
import numpy as np
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
training_args = TrainingArguments(
    output_dir="total_tuned_math_mc",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_math["train"],
    eval_dataset=tokenized_math["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,

)

trainer.train()























# def combine(examples):
#     reasoning = examples["Problem"] + " " + examples["Rationale"]
#     if len(reasoning) > 50:
#         while reasoning[-1].isdigit() == False:
#             reasoning = reasoning[:-1]
#         examples["reasoning"] = reasoning
#     return examples

# data = data.map(preprocess_function)
# data = data.map(combine)

# def preprocess_function(examples):
#     examples["full_response"] = examples["query"] + examples["response"]
#     return tokenizer(examples["full_response"])
# data = data.map(preprocess_function, remove_columns=("query","response","original_question"))
# print(data["train"][0])

# from transformers import DataCollatorForLanguageModeling


# tokenizer.add_special_tokens({'pad_token' : '[PAD]'})
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# model = AutoModelForCausalLM.from_pretrained("google-bert/bert-base-uncased")
# training_args = TrainingArguments(
#     output_dir="please-work",
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     weight_decay=0.01,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=data["train"],
#     eval_dataset=data["test"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# trainer.train()