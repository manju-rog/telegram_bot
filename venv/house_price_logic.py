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


# --- Session Management (Unchanged) ---
def load_session(chat_id):
    session_file = f"session_{chat_id}.json"
    if os.path.exists(session_file):
        with open(session_file, 'r') as f: return json.load(f)
    return {"features": {}}

def save_session(chat_id, session_data):
    with open(f"session_{chat_id}.json", 'w') as f: json.dump(session_data, f, indent=4)

def reset_session(chat_id):
    if os.path.exists(f"session_{chat_id}.json"): os.remove(f"session_{chat_id}.json")
    return "Memory wiped! Let's start building a new house profile."

# --- Core Logic (Unchanged) ---
def feature_extractor_llm(user_input, existing_features):
    prompt = f"""
    You are an expert feature extractor for a house price prediction model.
    Your task is to analyze the user's query and extract any of the following features: {ml_manager.FEATURE_NAMES}.

    **CRITICAL TASK:** You must also determine the user's intent.
    - If the query seems to describe a **new house from scratch** (e.g., "what's the price of a house with..."), include `"reset": true`.
    - If the query seems to be **adding a detail** to an existing house (e.g., "now make it 10 years old"), include `"reset": false`.

    Current known features: {json.dumps(existing_features)}
    User query: "{user_input}"

    Respond with ONLY a JSON object containing two keys:
    1. "reset": a boolean (true or false).
    2. "features": a JSON object of the features you found in the LATEST query.
    """
    try:
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = llm.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(cleaned_response)
    except Exception as e:
        logger.error(f"Error in feature extractor: {e}")
        return {"reset": False, "features": {}}

def smart_router_llm(user_input):
    prompt = f"""
    Analyze the user's query and determine its category.
    The categories are:
    1. "data_query": The user is asking a factual question about a dataset.
    2. "prediction_query": The user is providing details about a house for a price prediction.
    3. "other": The user is having a general conversation or asking something unrelated.

    User Query: "{user_input}"

    Respond with ONLY one word: "data_query", "prediction_query", or "other".
    """
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = llm.generate_content(prompt)
    return response.text.strip()

# --- NEW: Dynamic Response Generator for "other" intent ---
def generate_dynamic_redirection(user_input):
    """
    Creates a varied, engaging response to guide the user back on topic.
    """
    prompt = f"""
    You are a helpful and slightly witty AI assistant named Properlytics Bot.
    The user has said something that is off-topic. Your goal is to create a short, friendly response that acknowledges their message and then smoothly redirects them to your main functions.

    Your main functions are:
    1. Answering factual questions about a housing dataset (e.g., "What's the average house age?").
    2. Building a house profile to predict its price (e.g., "A house with 3 bedrooms.").

    User's off-topic message: "{user_input}"

    Generate a creative and engaging response. Here are some examples for inspiration:
    - "As an AI focused on real estate, I'm doing great crunching numbers! But I'd be even better if we could talk about a property. Do you have one in mind?"
    - "That's a bit outside my floor plan! I'm an expert on house prices, though. Got any property questions for me?"
    - "I appreciate you asking! My world is all about square footage and location, location, location. How can I help you with a housing query?"
    - "While I can't chat about that, I can tell you the median house value in our dataset. Interested?"

    Create a new, similar response now.
    """
    try:
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating dynamic redirection: {e}")
        # A reliable fallback if the LLM fails
        return "I can help with two things: answering factual questions about the housing dataset, or predicting a price for a house you describe. What would you like to do?"

# --- Main Handler: The Ultimate Combination ---
def handle_user_message(user_input, chat_id):
    if user_input.lower().strip() == '/reset':
        return reset_session(chat_id)

    intent = smart_router_llm(user_input)
    logger.info(f"User intent detected: {intent}")

    if intent == "data_query":
        return data_analyst_tool.query_housing_data(user_input)
    
    elif intent == "prediction_query":
        session = load_session(chat_id)
        
        extraction_result = feature_extractor_llm(user_input, session['features'])
        
        if extraction_result.get("reset", False):
            logger.info("Resetting features based on user query.")
            session['features'] = {}
            
        new_features = extraction_result.get("features", {})
        if new_features:
            session['features'].update(new_features)
            save_session(chat_id, session)

        if session['features']:
            prediction = ml_manager.predict_price(session['features'])
            
            feature_summary = "\n".join([f"- {key}: `{value}`" for key, value in session['features'].items()])
            response_text = (
                f"**Updated Profile & Estimate:**\n"
                f"{feature_summary}\n\n"
                f"Estimated Price: **${prediction:,.2f}**"
            )
            return response_text
        else:
            return "I couldn't find any specific house features in your message. Could you try describing the house again?"

    else: # intent == "other"
        # --- THE NEW DYNAMIC RESPONSE LOGIC ---
        return generate_dynamic_redirection(user_input)