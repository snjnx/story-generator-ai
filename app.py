import gradio as gr
from transformers import pipeline
import re

# Load the LaMini-Flan model
generator = pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-783M")

# Helper to truncate mid-sentence cutoffs
def truncate_to_last_sentence(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for i in range(len(sentences) - 1, -1, -1):
        if sentences[i].strip().endswith(('.', '?', '!')):
            return ' '.join(sentences[:i + 1])
    return text.strip()

# Story generation logic
def generate_story(prompt, genre, word_limit):
    tokens = int(word_limit * 2.2)  # Raise token estimate aggressively

    instruction = (
        f"You are a skilled author. Write a full {genre.lower()} story starting with:\n"
        f"\"{prompt.strip()}\"\n\n"
        f"The story must be complete, with a proper beginning, middle, and a satisfying ending. "
        f"Make sure it is roughly {word_limit} words and do not repeat phrases or ramble. End naturally."
    )

    result = generator(
        instruction,
        max_new_tokens=tokens,
        temperature=0.95,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2
    )[0]["generated_text"]

    return truncate_to_last_sentence(result)

# Gradio interface
gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(label="Story Prompt", placeholder="e.g. Mia found herself in ancient India."),
        gr.Dropdown(label="Genre", choices=["Fantasy", "Crime Thriller", "Romantic", "Sci-Fi", "Tragic", "Young Adult", "Children"]),
        gr.Slider(label="Word Limit", minimum=100, maximum=600, step=50, value=350)
    ],
    outputs=gr.Textbox(label="Generated Story"),
    title="AI Story Generator (LaMini-Flan)",
    description="A compact AI storyteller that generates story ideas. Paste a prompt, choose a genre, and get a plot to ponder over."
).launch(share=True)
