from transformers import DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
from datasets import load_dataset
import numpy as np
import evaluate


# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
# # encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
# # batch_sentences = [
# #     "But what about second breakfast?",
# #     "Don't think he knows about second breakfast, Pip.",
# #     "What about elevensies?",
# # ]
# # encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
# # print(encoded_inputs)
# # for input in encoded_inputs["input_ids"]:
# #     print(tokenizer.decode(input))

# dataset = load_dataset("yelp_review_full")
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# smaller_train_datasets = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# smaller_test_datasets = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels = 5)

# training_args = TrainingArguments(output_dir="test_trainer")
# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references = labels)

# trainer = Trainer(
#     model = model,
#     args = training_args,
#     train_dataset=smaller_train_datasets,
#     eval_dataset=smaller_test_datasets,
#     compute_metrics=compute_metrics
# )
# # trainer.train()

# classifier = pipeline(task="sentiment-analysis")
# result = classifier("I love minors")
# print(result)
# training_args = TrainingArguments(output_dir="test_trainer")
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
# model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
# #otherdataset = load_dataset("AI-MO/NuminaMath-CoT")

# print(dataset["test"][0])
# #encoded_input = tokenizer(dataset["test"]["problem"][0])
# batch_sentences = []

# small_data = dataset["test"].shuffle(seed=42).select(range(200))
# for i in range(len(small_data["question"])):
#     batch_sentences.append(small_data["question"][i])
# print(small_data["question"][0])
# print(small_data["choices"][0])
# print(small_data["answer"][0])
# encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
#print(encoded_inputs)

# def preprocess_function(examples):
#     questions = []
#     for i, question in enumerate(examples["question"]):
#         print(question)
#         for j in range(4):
#             temp_list = []
            
#             print(examples["choices"][i][j])
#             question += str(examples["choices"][i][j])
#             temp_list.append(question)
#         questions.append(temp_list)

# preprocess_function(small_data)

dataset = load_dataset("openai/gsm8k", "main")

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = "google-t5/t5-small")

rouge = evaluate.load("rouge")
prefix = "summarize: "
def preprocess_function(examples):
    inputs = [prefix + question for question in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["answer"], max_length = 1024, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def tokenize_function(examples):
    return tokenizer(examples["question"], padding="max_length", truncation=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="huggingface/my_awesome_model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print("Whats good")
trainer.train()

text = "summarize: George has 18 apples and $30. He can buy more apples for $5 per apple. How many apple pies can he make if each pie needs 4 apples and he spends all of his money on apples?"

answerer = pipeline("summarization", model="huggingface/my_awesome_model")
print(answerer(text))