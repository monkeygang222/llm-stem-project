from datasets import load_dataset

data = load_dataset("inverse-scaling/redefine-math", split="train")
data = data.train_test_split(test_size=0.2)
print(data["train"][0])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
def add_labels(examples):
    examples["label"] = examples["answer_index"]
    return examples

data = data.map(add_labels)
print(data["train"][0])
def preprocess_function(examples):
    
    question_answers = []
    for i in range(len(examples["prompt"])):
        temp1 = examples["prompt"][i]
        temp2 = examples["prompt"][i]
        temp1 += examples["classes"][i][0]
        temp2 += examples["classes"][i][1]
        question_answers.append(temp1)
        question_answers.append(temp2)
    
    #print(question_answers)
    tokenized_examples = tokenizer(question_answers, truncation=True)
    return {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}

tokenized_math = data.map(preprocess_function, batched=True)
print("finished with actual mapping")
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

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
    output_dir="my_awesome_math_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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

prompt = data['test'][0]["prompt"]
candidate1 = data["test"][0]["classes"][0]
candidate2 = data["test"][0]["classes"][1]
options = [candidate1, candidate2]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggingface/my_awesome_math_model")
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)

from transformers import AutoModelForMultipleChoice

model = AutoModelForMultipleChoice.from_pretrained("huggingface/my_awesome_math_model")
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits
predicted_class = logits.argmax().item()
print(prompt)
print(candidate1)
print(candidate2)
print("The answer chosen by the model is: " + options[predicted_class])