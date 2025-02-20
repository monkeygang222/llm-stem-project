# from datasets import load_dataset
# data = load_dataset("allenai/math_qa", split="validation")

# answers_dict = {
#     "a":0,
#     "b":1,
#     "c":2,
#     "d":3,
#     "e":4
# }

# def setupdata(examples):
    
#     temp = []
#     options_array = examples["options"] + " "
#     start_index = 0
#     end_index = 0 
#     for i in range(len(options_array)):
#         if options_array[i] == ")":
#             start_index = i
#         elif options_array[i] == "," or i == len(options_array) - 1:
#             end_index = i
#             if i == len(options_array) - 1:
#                 temp.append(options_array[start_index + 2:end_index])
#             else:
#                 temp.append(options_array[start_index + 2:end_index - 1])
#     examples["question"] = examples["Problem"]
#     examples["choices"] = temp
#     examples["answer"] = answers_dict[examples["correct"]]
#     return examples
# data = data.map(setupdata, remove_columns=["Problem", "options", "annotated_formula", "linear_formula", "category", "Rationale", "correct"])
# print(data[0])
# import torch

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from peft import PeftModel

# # num_choices = 5
# base_model_name = "google-bert/bert-base-uncased"
# peft_model_path = "mcq_model/checkpoint-375"  





# def predict(question, choices, correct_idx):

#     tokenized_inputs = tokenizer(
#         [f"Question: {question} Answer: {choice}" for choice in choices],  
#         return_tensors="pt", padding=True, truncation=True
#     )

#     with torch.no_grad():
#         outputs = model(**tokenized_inputs)

#     best_choice_idx = torch.argmax(outputs.logits).item()
#     best_answer = choices[best_choice_idx]

#     is_correct = best_choice_idx == correct_idx

#     return best_answer, is_correct

# # Example prediction

# for i in range(100):
#     question = data[i]["question"]
#     choices = data[i]["choices"]
#     correct = data[i]["answer"]
#     print(question)
#     print(choices)
#     print(predict(question, choices, correct))

# import torch

# # New question and choices for inference
# question = "What is the result of 5 + 3?"
# choices = ["7", "8", "9", "10"]

# inputs = [tokenizer(f"[CLS] Question: {question} [SEP] Answer: {choice} [SEP]", padding="max_length", truncation=True, return_tensors="pt") for choice in choices]

# input_ids = torch.cat([input["input_ids"] for input in inputs], dim=0)
# attention_mask = torch.cat([input["attention_mask"] for input in inputs], dim=0)

# with torch.no_grad():
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# logits = outputs.logits
# predicted_class_id = torch.argmax(logits, dim=-1).item()

# predicted_answer = choices[predicted_class_id]

# print(f"Predicted answer: {predicted_answer}")
# import torch

# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# New question and choices for inference
from datasets import load_dataset
data = load_dataset("allenai/math_qa", split="validation")

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

counter = 0
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=5).to(device)


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
for i in range(1000):
    question = data[i]["question"]
    choices = data[i]["choices"]

    inputs = [tokenizer(f"[CLS] Question: {question} [SEP] Answer: {choice} [SEP]", padding="max_length", truncation=True, return_tensors="pt") for choice in choices]

    input_ids = torch.cat([input["input_ids"].to(device) for input in inputs], dim=0)
    attention_mask = torch.cat([input["attention_mask"].to(device) for input in inputs], dim=0)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1)
    if predicted_class_id.numel() == 1: 
        predicted_class_id = predicted_class_id.item()
    else: 
        predicted_class_id = predicted_class_id.tolist()
    if predicted_class_id[0] == data[i]["answer"]:
        counter += 1
    predicted_answer = choices[predicted_class_id] if isinstance(predicted_class_id, int) else [choices[idx] for idx in predicted_class_id]
print(counter)


