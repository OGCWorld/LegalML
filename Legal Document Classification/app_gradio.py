import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained LegalBERT model
MODEL_PATH = r"C:\Users\cfranklin2019\OneDrive - Florida Atlantic University\Documents\GitHub\LegalML\Legal Document Classification\Notebooks\legalbert_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict legal text classification
def classify_legal_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return f"Predicted Legal Category: {predicted_class}"

# Gradio UI
interface = gr.Interface(
    fn=classify_legal_text,
    inputs="text",
    outputs="text",
    title="LegalBERT Text Classifier",
    description="Enter a legal sentence to classify its category using a fine-tuned LegalBERT model."
)

# Launch app
interface.launch()
