import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Model Setup ---
model_path = "./toxic_multilabel_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_toxicity(text):
    if not text.strip():
        return "⚠️ Empty Input", "None", {}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
    all_scores = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    
    is_toxic = probs[0] >= 0.4
    status = "🚨 TOXIC CONTENT" if is_toxic else "✅ CLEAN CONTENT"
    
    triggered_categories = [label_names[i].replace('_', ' ').upper() for i in range(1, 6) if probs[i] >= 0.5]
    
    if is_toxic:
        category_display = " | ".join(triggered_categories) if triggered_categories else "GENERAL TOXICITY"
    else:
        category_display = "SAFE"

    return status, category_display, all_scores

# --- Custom Modern Styling ---
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
    font-family: 'Inter', sans-serif;
}
#title-box { text-align: center; margin-bottom: 20px; }
#title-box h1 { color: #0369a1; font-weight: 800; }
.input-card { 
    background: white; 
    padding: 20px; 
    border-radius: 15px; 
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); 
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="sky")) as demo:
    with gr.Column(elem_id="title-box"):
        gr.Markdown("# 🛡️ Sentinel AI")
        gr.Markdown("### Advanced Multi-Label Toxicity Guard")

    with gr.Row(elem_classes="input-card"):
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="✍️ Analyze Comment", 
                placeholder="Type your text here to check for harmful content...",
                lines=8
            )
            with gr.Row():
                analyze_btn = gr.Button("Analyze Toxicity 🚀", variant="primary")
                # FIX: ClearButton now targets all specific components
                clear_btn = gr.ClearButton(value="Clear All")
            
        with gr.Column(scale=1):
            output_status = gr.Textbox(label="Final Verdict", interactive=False)
            output_cat = gr.Textbox(label="Detected Categories", interactive=False)
            output_labels = gr.Label(label="Category Confidence Scores")

    # Wire logic
    analyze_btn.click(
        fn=predict_toxicity, 
        inputs=input_text, 
        outputs=[output_status, output_cat, output_labels]
    )
    
    # The fix for your clear button issue:
    clear_btn.add([input_text, output_status, output_cat, output_labels])

if __name__ == "__main__":
    demo.launch()