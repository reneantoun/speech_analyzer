from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# IBM Watson Machine Learning credentials
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

# Parameters for the LLM
params = {
    GenParams.MAX_NEW_TOKENS: 800,  # Maximum tokens for generation
    GenParams.TEMPERATURE: 0.1,    # Creativity parameter (lower = deterministic, higher = random)
}

# Set up the Llama 3 model
LLAMA3_model = Model(
    model_id='meta-llama/llama-3-8b-instruct',  # Llama 3 model ID
    credentials=my_credentials,
    params=params,
    project_id="skills-network",  # Project ID provided in the instructions
)

# Create an instance of the LLM
llm = WatsonxLLM(LLAMA3_model)

# Generate a response
response = llm("How to read a book effectively?")

# Print the response
print(response)
