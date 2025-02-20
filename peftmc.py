import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
from datasets import load_dataset

#https://huggingface.co/datasets/meta-math/MetaMathQA - potentially will use
#https://huggingface.co/datasets/allenai/math_qa - multiple choice
# data = load_dataset("allenai/math_qa", split="train[:5000]", trust_remote_code=True)
data = load_dataset("allenai/math_qa", split="train[:5000]")
data = data.train_test_split(test_size=0.2)

#https://arxiv.org/abs/2309.12284

answers_dict = {
    "a":0,
    "b":1,
    "c":2,
    "d":3,
    "e":4
}

def setupdata(examples):
    
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
    examples["question"] = examples["Problem"]
    examples["choices"] = temp
    examples["answer"] = answers_dict[examples["correct"]]
    return examples
data = data.map(setupdata, remove_columns=["Problem", "options", "annotated_formula", "linear_formula", "category", "Rationale", "correct"])

print(data["train"][0])


from transformers import AutoTokenizer

model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda:0")

def preprocess_function(examples):

    question = examples["question"]
    choices = examples["choices"]
    answer_idx = examples["answer"]
    
    # Format each choice with the question
    inputs = [tokenizer(f"[CLS] Question: {question} [SEP] Answer: {choice} [SEP]", padding="max_length", truncation=True) for choice in choices]
    
    # Tokenize the inputs
    tokenized_inputs = {key: [dic[key] for dic in inputs] for key in inputs[0]}
    # Store the correct answer as a label
    tokenized_inputs["labels"] = answer_idx  
    return tokenized_inputs

dataset = data.map(preprocess_function, batched=True)

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5).to(device)

 
print(dataset["train"][0]["labels"])
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1,
    task_type="SEQ_CLS"  
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    output_dir="mcq_model",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["train"],
)
trainer.train()





# New question and choices for inference
question = "What is the result of 5 + 3?"
choices = ["7", "8", "9", "10"]

inputs = [tokenizer(f"[CLS] Question: {question} [SEP] Answer: {choice} [SEP]", padding="max_length", truncation=True, return_tensors="pt") for choice in choices]

input_ids = torch.cat([input["input_ids"].to(device) for input in inputs], dim=0)
attention_mask = torch.cat([input["attention_mask"].to(device) for input in inputs], dim=0)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

logits = outputs.logits
predicted_class_id = torch.argmax(logits, dim=-1).item()

predicted_answer = choices[predicted_class_id]

print(f"Predicted answer: {predicted_answer}")