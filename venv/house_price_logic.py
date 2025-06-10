# house_price_logic.py

import requests
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import re
import os
import logging
from dotenv import load_dotenv

# --- 1. INITIAL SETUP & CONFIGURATION ---
print("Initializing House Price Logic...")
load_dotenv() # Load variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Securely load API Key from environment
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file!")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

# --- 2. MODEL AND DATA LOADING (No changes here, but kept for completeness) ---
model_path = "house_price_model.joblib"
if not os.path.exists(model_path):
    # This part remains the same: training the model if it doesn't exist
    logger.info("Training new ML model...")
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    feature_names = data.feature_names
    X_train, _, y_train, _ = train_test_split(df[feature_names], df["MedHouseVal"], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    logger.info("Model saved!")
else:
    logger.info("Loading existing ML model...")

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
feature_names = data.feature_names
mdl = joblib.load(model_path)
logger.info("Initialization Complete.")


# --- 3. SESSION MANAGEMENT (No changes here) ---
def load_session(chat_id):
    session_file = f"session_{chat_id}.json"
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session = json.load(f)
    else:
        session = {"features": {}, "conversation_history": []}
    session.setdefault('state', 'gathering_info')
    return session

def save_session(chat_id, session_data):
    session_file = f"session_{chat_id}.json"
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=4)

def reset_session(chat_id):
    session_file = f"session_{chat_id}.json"
    if os.path.exists(session_file):
        os.remove(session_file)
    return "Memory wiped! I'm ready for a new property. What do you have in mind?"


# --- 4. CORE LOGIC FUNCTIONS ---
def format_conversation_history(session):
    recent_history = session["conversation_history"][-12:]
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])

def format_features_for_display(feature_dict):
    return ", ".join([f"{key}: {value}" for key, value in feature_dict.items()])

def process_with_llm(user_input, session):
    """Interacts with the Gemini API using the production-grade prompt."""

    prompt = f"""
    You are the "Properlytics Bot", a specialized AI assistant for house price prediction.
    Your personality is helpful, professional, and slightly witty. Your primary goal is to gather property features and provide an estimated price.

    **YOUR CORE INSTRUCTIONS:**

    1.  **MAINTAIN YOUR PERSONA:** You ONLY discuss real estate and property prices. You must gracefully deflect all other topics.
    2.  **EXTRACT FEATURES:** Your main job is to extract these specific features from the user's text: {', '.join(feature_names)}.
    3.  **HANDLE OUT-OF-CONTEXT QUERIES:** If the user asks about anything other than housing (e.g., weather, jokes, politics, your own nature), you MUST use one of the following deflections. Do not answer their question.
        - "That's a bit outside my floor plan! I'm focused on property prices. Do you have a house in mind?"
        - "While I can estimate the value of a house, I'm not equipped to answer that. Let's get back to real estate!"
        - "Interesting question! However, my expertise is strictly in the housing market. What property details can you share?"
        - "My blueprint doesn't cover that topic. I can, however, tell you about house prices!"
    4.  **HANDLE VAGUE OR AMBIGUOUS QUERIES:** If a user's query is about housing but is too vague (e.g., "is my house expensive?"), ask clarifying questions to get concrete features.
    5.  **HANDLE FEATURE MODIFICATIONS:** If a user wants to change a feature (e.g., "add one more room"), acknowledge the change and, if the new value is significant (like a high room count), politely ask for more context (like income or location) to improve accuracy.
    6.  **DECIDE THE NEXT STEP:** Based on the extracted info, decide your response.

    **CURRENT CONVERSATION:**

    *   **User's Most Recent Message:** "{user_input}"
    *   **Features I Already Know:** {json.dumps(session["features"])}
    *   **Full Conversation History:**
        {format_conversation_history(session)}

    ---
    **YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT. Do not add any text before or after the JSON.**
    The JSON object should have these keys:
    1.  "is_house_query": boolean (true ONLY if the user is talking about properties).
    2.  "response": Your complete, conversational response string to the user, following the persona and rules above.
    3.  "features": A dictionary of any NEW features you extracted from the user's latest message.
    """
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(ENDPOINT, headers=headers, json=data, timeout=20)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        
        output = response.json()
        llm_response_text = output["candidates"][0]["content"]["parts"][0]["text"]
        
        # Robust JSON parsing
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            logger.warning(f"LLM did not return valid JSON. Response: {llm_response_text}")
            return {"is_house_query": False, "response": "I seem to be having a little trouble processing that. Could you please rephrase?", "features": {}}

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Gemini API: {e}")
        return {"is_house_query": False, "response": "I'm having trouble connecting to my services right now. Please try again in a moment.", "features": {}}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing LLM response: {e} - Full response: {output}")
        return {"is_house_query": False, "response": "I received an unusual response from my core brain. Let's try that again.", "features": {}}


def predict_house_price(session):
    if not session["features"]: return None
    model_input = {feature: df[feature].mean() for feature in feature_names}
    model_input.update(session["features"])
    df_input = pd.DataFrame([model_input], columns=feature_names)
    prediction = mdl.predict(df_input)[0]
    return prediction * 100000

# --- 5. MAIN HANDLER (No changes to logic, just uses the improved functions) ---
def handle_user_message(user_input, chat_id):
    if user_input.lower() in ["/reset", "reset"]: return reset_session(chat_id)

    session = load_session(chat_id)
    
    if session['state'] == 'waiting_for_confirmation':
        affirmative_responses = ['yes', 'yep', 'correct', 'right', 'ok', 'sure', 'yeah', 'y']
        
        session["conversation_history"].append({"role": "user", "content": user_input})
        if user_input.lower().strip() in affirmative_responses:
            session['state'] = 'gathering_info'
            final_response = "Excellent! Let me run the numbers for you."
            
            if len(session["features"]) >= 2:
                prediction = predict_house_price(session)
                if prediction is not None:
                    final_response += f"\n\nBased on the current details, my estimated price is: **${prediction:,.2f}**\n\nFeel free to add more details for a refined estimate!"
            else:
                final_response = "Confirmed! I still need at least one more key detail (like location, income, or house age) to run a prediction."

            session["conversation_history"].append({"role": "assistant", "content": final_response})
            save_session(chat_id, session)
            return final_response
        else:
            session['state'] = 'gathering_info'
            response_text = "My apologies for the misunderstanding. Let's correct that. What are the right details?"
            session["conversation_history"].append({"role": "assistant", "content": response_text})
            save_session(chat_id, session)
            return response_text

    session["conversation_history"].append({"role": "user", "content": user_input})
    result = process_with_llm(user_input, session)
    
    if result.get("is_house_query") and result.get("features"):
        newly_extracted_features = result["features"]
        session["features"].update(newly_extracted_features)
        session['state'] = 'waiting_for_confirmation'
        
        confirmation_question = (
            f"{result['response']}\n\nTo be sure, I've noted these new details:\n"
            f"**{format_features_for_display(newly_extracted_features)}**\n\nIs this correct?"
        )
        
        session["conversation_history"].append({"role": "assistant", "content": confirmation_question})
        save_session(chat_id, session)
        return confirmation_question
    else:
        session["conversation_history"].append({"role": "assistant", "content": result["response"]})
        save_session(chat_id, session)
        return result['response']