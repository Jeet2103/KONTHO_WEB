from flask import Flask, request, jsonify
import numpy as np
# import pickle
import joblib
# from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load environment variables from .env file
groq_api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

# Initialize the Groq Llama model using LangChain's ChatGroq
# groq_model = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)
llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=70, temperature= 0.2)

# Create the prompt template for sentence completion
chat_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI assiatant that takes in input incomplete and incoherent sentences and replies with gramatically and 
    semantically correct and complete sentence.Give just the sentence and nothing else.\nincomplete sentence :  {incomplete_sentence}
    \ncomplete sentence :""
    """
)

# Create the chain
llm_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt
)

# Load the trained machine learning model (assuming the model is saved as 'INITIAL_MODEL_1.pkl')
model = joblib.load('INITIAL_MODEL_1.pkl')

@app.route('/', methods=['GET'])
def welcome():
    return "Welcome to the prediction service!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json(force=True)
    
    if 'input_data' not in data:
        return jsonify({"error": "No input data provided!"}), 400
    
    input_data = data['input_data']

    try:
        # Convert input data to NumPy array for prediction
        input_array = np.array(input_data).reshape(1, -1)  # Reshape to ensure correct input format
        
        # Predict the class
        predicted_class = model.predict(input_array)

        # Create a mapping of the class predictions to the desired strings
        class_mapping = {
            0: "Hello",
            1: "Sorry",
            2: "Thank you"
        }

        # Get the first prediction (assuming one input at a time)
        predicted_output = predicted_class[0]

        # Get the corresponding string from the class_mapping
        response_string = class_mapping.get(predicted_output, "Unknown")
        
        return jsonify({'message': response_string})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/complete', methods=['GET'])
def complete_sentence():
    # Get the incomplete sentence from query parameters
    incomplete_sentence = request.args.get('sentence')

    if not incomplete_sentence:
        return jsonify({"error": "Please provide an incomplete sentence!"}), 400

    # Call the LLM chain to complete the sentence
    try:
        result = llm_chain.run(incomplete_sentence)
        return jsonify({"completed_sentence": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
