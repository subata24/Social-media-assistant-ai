import gradio as gr
from transformers import pipeline

# ✅ Free text generation pipeline
generator = pipeline("text-generation", model="facebook/opt-iml-1.3b")

def generate_captions(topic, tone):
    if not topic:
        return "Please enter a topic.", ""
    
    # Generate captions
    prompt = f"Write 3 {tone} Instagram captions about: {topic}"
    captions = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

    # Generate hashtags
    prompt_hashtags = f"Suggest 10 trending hashtags for Instagram post about: {topic}"
    hashtags = generator(prompt_hashtags, max_length=50, num_return_sequences=1)[0]['generated_text']

    return captions, hashtags

# Gradio Interface
title = "📱 AI Social Media Assistant (Free)"
description = "Generate Instagram captions and hashtags for your posts using free AI models."

iface = gr.Interface(
    fn=generate_captions,
    inputs=[
        gr.Textbox(label="Enter Topic", placeholder="e.g., Summer fashion trends"),
        gr.Dropdown(["Professional", "Fun", "Inspirational", "Gen-Z"], label="Tone")
    ],
    outputs=[
        gr.Textbox(label="Generated Captions"),
        gr.Textbox(label="Suggested Hashtags")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()