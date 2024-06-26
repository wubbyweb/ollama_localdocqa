# Create main project directory
New-Item -ItemType Directory -Path MyProject
cd MyProject

# Create directories
New-Item -ItemType Directory -Path data\documents -Force
New-Item -ItemType Directory -Path models\ollama_model -Force
New-Item -ItemType Directory -Path src -Force

# Create Python program file
@"
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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
    # Load model
    model_path = "models/ollama_model"  # Path to your OLLAMA LLM model
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
"@ | Out-File -FilePath src\main.py -Force

# Create README.md file
@"
# MyProject

This is a project for utilizing OLLAMA LLMs to perform document question answering locally from your desktop.

## Usage

1. **Data**: Place your text documents in the `data/documents` directory.
2. **Model**: Download the OLLAMA LLM model files and place them in the `models/ollama_model` directory.
3. **Run**: Execute the `src/main.py` script to perform document question answering.

Feel free to modify the code and structure according to your requirements.

## Dependencies

- [Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
"@ | Out-File -FilePath README.md -Force

# Provide instructions
Write-Output "Project files and README.md created successfully!"
