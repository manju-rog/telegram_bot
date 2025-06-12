# house_price_logic.py

import os
import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Import our specialized modules
import ml_manager
import data_analyst_tool

# --- Configuration ---
load_dotenv()
logger = logging.getLogger(__name__)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the features the user can select from
SELECTABLE_FEATURES = ml_manager.FEATURE_NAMES

# --- Session Management ---
def load_session(chat_id):
    session_file = f"session_{chat_id}.json"
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            return json.load(f)
    # Default new session state
    return {"flow": None, "selected_features": [], "feature_values": {}}

def save_session(chat_id, session_data):
    with open(f"session_{chat_id}.json", 'w') as f:
        json.dump(session_data, f, indent=4)

def reset_session(chat_id):
    if os.path.exists(f"session_{chat_id}.json"):
        os.remove(f"session_{chat_id}.json")
    logger.info(f"Session reset for chat_id {chat_id}")

# --- Smart Router (To Fix the Data Query Bug) ---
def smart_router_llm(user_input):
    """
    Decides if the user is asking a factual data question or something else.
    This is used when the user is not in an active flow.
    """
    prompt = f"""
    Analyze the user's query and determine its category.
    The categories are:
    1. "data_query": The user is asking a factual question about a dataset (e.g., 'how many', 'what is the average', 'show me the top 5').
    2. "other": The user is having a general conversation, asking about predictions, or saying something else.

    User Query: "{user_input}"

    Respond with ONLY one word: "data_query" or "other".
    """
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = llm.generate_content(prompt)
    return response.text.strip()

# --- Feature Parser for "Single Go" Input ---
def parse_feature_values_from_text(text_input: str, features_to_find: list):
    """
    Uses an LLM to parse multiple feature values from a single user message.
    """
    prompt = f"""
    From the user's text, extract the numerical values for the following features: {features_to_find}.
    
    User's Text: "{text_input}"

    Respond with ONLY a JSON object where keys are the feature names and values are the numbers found.
    If a value is not found for a feature, omit it from the JSON.
    """
    try:
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(cleaned_response)
    except Exception as e:
        logger.error(f"Error parsing feature values: {e}")
        return {}