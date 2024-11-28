import torch
from transformers import pipeline
import gradio as gr

# Initialize the speech-to-text pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30,
)

# Define a function to process the audio file and return the transcribed text
def transcribe_audio(audio):
    prediction = pipe(audio)["text"]
    return prediction

# Create a Gradio interface
interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),  # Removed 'source' as it's not needed anymore
    outputs="text",
    title="Speech-to-Text Transcription",
    description="Upload an audio file, and this app will transcribe it using OpenAI Whisper."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
