from transformers import pipeline
import gradio as gr

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Define a function to use in the Gradio app
def analyze_sentiment(text):
    result = classifier(text)[0]
    return f"Label: {result['label']}, Confidence: {round(result['score'], 2)}"

# Build Gradio UI
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter your sentence..."),
    outputs="text",
    title="🧠 Sentiment Analyzer",
    description="Built using Hugging Face Transformers and Gradio"
)

iface.launch()
