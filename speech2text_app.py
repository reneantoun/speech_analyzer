import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    
    # Transcribe the audio file and return the result
    result = pipe(audio_file)["text"]
    return result

# Set up Gradio interface
audio_input = gr.Audio(type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription App",
    description="Upload an audio file, and this app will transcribe it."
)

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
