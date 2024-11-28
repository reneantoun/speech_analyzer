import torch
import gradio as gr
from langchain.llms import HuggingFaceHub  # Use HuggingFace as an example
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

#######------------- LLM-------------####
# Initiate LLM instance
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

LLAMA3_model = Model(
    model_id='meta-llama/llama-3-8b-instruct',
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(LLAMA3_model)

#######------------- Prompt Template-------------####
# This template is structured based on LLAMA3
temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""
# Create the PromptTemplate
pt = PromptTemplate(
    input_variables=["context"],
    template=temp,
)

# Create the LLMChain
prompt_to_LLAMA3 = LLMChain(llm=llm, prompt=pt)

#######------------- Speech2Text-------------####
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    
    # Transcribe the audio file and return the result
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    
    # Run the LLMChain to merge transcript text with the template and send it to the LLM
    result = prompt_to_LLAMA3.run(transcript_txt)
    return result

#######------------- Gradio-------------####
# Define Gradio inputs and outputs
audio_input = gr.Audio(type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface
iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Speech Analyzer App",
    description="Upload an audio file, and the app will transcribe it and analyze key points using the LLM.",
)

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
