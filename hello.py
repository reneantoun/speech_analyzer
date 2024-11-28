import gradio as gr

# Function to greet the user
def greet(name):
    return "Hello " + name + "!"

# Create the Gradio interface
demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# Launch the Gradio app
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

