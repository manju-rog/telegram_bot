# agent_manager.py

import os
import pandas as pd
from dotenv import load_dotenv
import logging
import json 
# --- Core Dependencies ---
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
import google.generativeai as genai

# Import our existing ML model predictor
import ml_manager

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# --- Tool 1: The ML Price Predictor (REBUILT FOR RELIABILITY) ---
def predict_house_price_tool(natural_language_query_with_features: str) -> str:
    """
    Predicts the price of a hypothetical house.
    Use this for "what if" scenarios when the user provides specific feature values.
    The input should be a single string containing all the features mentioned by the user.
    Example: 'Predict the price for a house with MedInc 8, HouseAge 5, AveRooms 7, AveBedrms 1, Population 300, AveOccup 2, Latitude 34, and Longitude -118.'
    """
    prompt = f"""
    From the following user query, extract the 8 required features for a house price prediction.
    The required features are: {ml_manager.FEATURE_NAMES}.
    If a feature is not mentioned in the query, assign it a value of `None`.
    Respond with ONLY a JSON object containing the 8 features.

    User Query: "{natural_language_query_with_features}"

    JSON Response:
    """
    try:
        # First LLM call to extract features into a structured format
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = llm.generate_content(prompt)
        # Clean up potential markdown
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        features = json.loads(cleaned_response)

        logger.info(f"Extracted features for prediction: {features}")

        # Fill any missing (None) values with the mean from our base data
        _, df_base = ml_manager.get_model_and_data()
        for key, val in features.items():
            if val is None:
                features[key] = df_base[key].mean()
        
        # Call our robust ML prediction function
        prediction = ml_manager.predict_price(features)
        return f"Based on the provided details, the estimated price is ${prediction:,.2f}."

    except Exception as e:
        logger.error(f"Failed to extract features or predict price: {e}")
        return "I couldn't quite understand the features provided. Could you please list them more clearly? For example: 'MedInc is 3.5, HouseAge is 15, ...'"

price_prediction_tool = FunctionTool.from_defaults(
    fn=predict_house_price_tool,
    name="HousePricePredictor",
    description="Predicts the price of a hypothetical house given its features in a single text string."
)


# --- Tool 2: The Pandas Data Analyst (REBUILT FOR ACCURACY) ---
try:
    df = pd.read_csv("housing_initial.csv")
    # THE PRICE COLUMN IS MedHouseVal, NOT SalePrice or Price
    VALID_COLUMNS = ml_manager.FEATURE_NAMES + ['MedHouseVal']
    df = df[VALID_COLUMNS]
    DF_HEAD_STR = str(df.head())
except FileNotFoundError:
    logger.error("data/housing_initial.csv not found! Please run prepare_data.py.")
    df = pd.DataFrame()
    VALID_COLUMNS = []
    DF_HEAD_STR = "Error: Data file not found."

def pandas_data_analyst_tool(query: str) -> str:
    """
    Answers factual questions about the California housing dataset by executing Python Pandas code.
    Use this for questions involving aggregations (average, max, min), counting, sorting, or filtering data.
    """
    # ** THE CRUCIAL FIX IS HERE: We explicitly list the valid column names in the prompt **
    prompt = f"""
    You are an expert Python Pandas programmer.
    Your task is to convert a natural language query into a single, executable line of Pandas code that operates on a dataframe named `df`.
    The dataframe `df` has the following columns: {VALID_COLUMNS}
    This is the head of the dataframe:
    {DF_HEAD_STR}

    RULES:
    1. The code MUST be a single line.
    2. You MUST only use the columns listed above. Do not make up column names like 'Price'. The price column is 'MedHouseVal'.
    3. Only output the raw Python code. Do not add explanations or markdown.

    Natural Language Query: "{query}"
    
    Pandas Code:
    """
    
    try:
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = llm.generate_content(prompt)
        pandas_code = response.text.strip().replace('`', '')

        logger.info(f"Generated Pandas Code: {pandas_code}")
        
        result = eval(pandas_code, {"df": df, "pd": pd})
        
        logger.info(f"Execution result: {result}")
        return str(result)

    except Exception as e:
        logger.error(f"Error executing pandas code for query '{query}': {e}")
        return "I'm sorry, I encountered an error trying to analyze the data. It's possible the question was too complex or referred to data I don't have. Please try rephrasing."

pandas_query_tool = FunctionTool.from_defaults(
    fn=pandas_data_analyst_tool,
    name="HousingDataQueryTool",
    description="Answers factual questions about the California housing dataset."
)


# --- The Hybrid Agent (Unchanged, but now controls better tools) ---
def get_chat_agent():
    llm = Gemini(model="models/gemini-1.5-flash-latest")
    
    tools = [
        price_prediction_tool,
        pandas_query_tool,
    ]
    
    system_prompt = (
        "You are an expert housing assistant. You have two tools at your disposal:\n"
        "1. HousePricePredictor: Use this for 'what-if' scenarios or when a user asks to predict a price for a house with specific features. Gather all features in a single call.\n"
        "2. HousingDataQueryTool: Use this to answer factual questions about the dataset, such as counting, averaging, or finding specific rows.\n"
        "Analyze the user's query and choose the most appropriate tool. If you don't have enough information for a tool, ask the user for it."
    )
    
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, system_prompt=system_prompt)
    return agent

chat_agent = get_chat_agent()

def handle_user_query(user_input, chat_id):
    try:
        response = chat_agent.chat(user_input)
        return str(response)
    except Exception as e:
        logger.error(f"Error in agent chat: {e}")
        return "An error occurred while processing your request. Please try again."