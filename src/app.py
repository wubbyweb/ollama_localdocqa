from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return tokenizer, model

def answer_question(question, context, tokenizer, model):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

if __name__ == "__main__":
    load_dotenv()
    
    login(os.environ['HF_WRITE_TOKEN'])
    # Load model
    model_path = "meta-llama/Llama-2-7b-hf"  # Path to your OLLAMA LLM model
    tokenizer, model = load_model(model_path)

    # Load document
    document_path = "data/documents/document1.txt"  # Path to your document
    with open(document_path, "r", encoding="utf-8") as file:
        context = file.read()

    # Ask question
    question = "What is the main topic of the document?"
    answer = answer_question(question, context, tokenizer, model)
    print("Question:", question)
    print("Answer:", answer)
