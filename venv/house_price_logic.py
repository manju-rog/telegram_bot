# house_price_logic.py

import requests
import json
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import re
import os

# --- 1. INITIAL SETUP ---
print("Initializing House Price Logic...")

# IMPORTANT: Paste your actual Gemini API Key here.
API_KEY = "AIzaSyB7G-ePwKXFwzl1wswUT3eTYlU9-Kzr3mQ"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

# Load or train the machine learning model
model_path = "house_price_model.joblib"
if not os.path.exists(model_path):
    print("Training new model...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    feature_names = data.feature_names
    target_name = "MedHouseVal"
    X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df[target_name], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Model saved!")
else:
    print("Loading existing model...")

# Load necessary data into memory
data = fetch_california_housing(as_frame=True)
df = data.frame
feature_names = data.feature_names
mdl = joblib.load(model_path)
print("Initialization Complete.")


# --- 2. SESSION MANAGEMENT ---
def load_session(chat_id):
    """Loads a session for a specific user and ensures state exists."""
    session_file = f"session_{chat_id}.json"
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session = json.load(f)
    else:
        session = {"features": {}, "conversation_history": []}
    
    session.setdefault('state', 'gathering_info')
    return session

def save_session(chat_id, session_data):
    """Saves a session for a specific user."""
    session_file = f"session_{chat_id}.json"
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=4)

def reset_session(chat_id):
    """Resets a session for a specific user."""
    session_file = f"session_{chat_id}.json"
    if os.path.exists(session_file):
        os.remove(session_file)
    return "All features reset. Let's start over!"


# --- 3. CORE LOGIC FUNCTIONS ---
def format_conversation_history(session):
    recent_history = session["conversation_history"][-10:]
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_history])

def format_features_for_display(feature_dict):
    """Turns a dictionary of features into a readable string."""
    return ", ".join([f"{key}: {value}" for key, value in feature_dict.items()])

def process_with_llm(user_input, session):
    """Interacts with the Gemini API using the improved, context-aware prompt."""
    prompt = f"""
You are an expert house price prediction assistant. Your goal is to gather information from the user to predict a house price.

**Your Instructions:**

1.  **Extract Features:** From the user's query, extract any of the following features: {', '.join(feature_names)}.
2.  **Handle Ambiguity and Context:** If the user provides a value that could be unusual on its own (like a high number of rooms or a very low house age), your conversational response MUST ask for more clarifying information (like income or location) to ensure the prediction is accurate. This is crucial because a house with many rooms is priced very differently in a high-income vs. a low-income area.
3.  **Conversational Response:** Generate a helpful, conversational response based on the user's input.

**Example Scenarios:**

*   **User Query:** "I need a house with 3 bedrooms and 5 rooms."
    *   **Good `response`:** "Okay, a 3-bedroom, 5-room house. I've noted that."
    *   **Good `features`:** {{"AveBedrms": 3.0, "AveRooms": 5.0}}
*   **User Query:** "Now add one more room" (assuming the bot knows there are already 5 rooms)
    *   **Good `response`:** "Okay, I've updated the property to have 6 rooms. A larger house like this can vary a lot in price. To help me be more accurate, could you also tell me the median income for the area or its location?"
    *   **Good `features`:** {{"AveRooms": 6.0}}
*   **User Query:** "what's the weather today"
    *   **Good `response`:** "I can only help with house price predictions. Do you have a property in mind?"
    *   **Good `features`:** {{}}

**Current Conversation:**

*   **User's Most Recent Message:** "{user_input}"
*   **Features Already Known:** {json.dumps(session["features"])}
*   **Previous Chat History:**
{format_conversation_history(session)}

---
**Your Output MUST be a single JSON object with the following keys:**
1.  "is_house_query": boolean
2.  "response": Your conversational response as a string.
3.  "features": A dictionary of any house features you extracted.
"""
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        output = response.json()
        llm_response_text = output["candidates"][0]["content"]["parts"][0]["text"]
        
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return {"is_house_query": False, "response": "I'm having trouble understanding. Can you tell me about the house?", "features": {}}
    except Exception as e:
        print(f"Error processing with LLM: {e}")
        return {"is_house_query": False, "response": "Sorry, I encountered an error. Let's try again.", "features": {}}

def predict_house_price(session):
    """Predicts house price based on current session features."""
    if not session["features"]:
        return None
    model_input = {feature: df[feature].mean() for feature in feature_names}
    model_input.update(session["features"])
    df_input = pd.DataFrame([model_input], columns=feature_names)
    prediction = mdl.predict(df_input)[0]
    return prediction * 100000


# --- 4. MAIN HANDLER with State Machine ---
def handle_user_message(user_input, chat_id):
    """Processes a user's message with state management for confirmation."""
    if user_input.lower() in ["exit", "quit", "bye"]:
        return "Thank you for using the House Price Prediction Assistant. Goodbye!"
    if user_input.lower() == "reset":
        return reset_session(chat_id)

    session = load_session(chat_id)
    
    if session['state'] == 'waiting_for_confirmation':
        affirmative_responses = ['yes', 'yep', 'correct', 'right', 'ok', 'sure', 'yeah', 'y']
        
        if user_input.lower().strip() in affirmative_responses:
            session['state'] = 'gathering_info'
            session["conversation_history"].append({"role": "user", "content": user_input})
            
            final_response = "Great! Let me calculate the price for you."
            
            if len(session["features"]) >= 2:
                prediction = predict_house_price(session)
                if prediction is not None:
                    final_response += f"\n\nBased on the details provided, the estimated house price is: **${prediction:,.2f}**"
                else:
                    final_response = "I have the confirmed details, but something went wrong with the prediction."
            else:
                final_response = "Confirmed! I still need at least one more feature to predict the price. What else can you tell me?"

            session["conversation_history"].append({"role": "assistant", "content": "Confirmed. (Proceeding with prediction...)"})
            save_session(chat_id, session)
            return final_response
            
        else:
            session['state'] = 'gathering_info'
            session["conversation_history"].append({"role": "user", "content": user_input})
            response_text = "My apologies! Let's try that again. Please provide the correct details for the property."
            session["conversation_history"].append({"role": "assistant", "content": response_text})
            save_session(chat_id, session)
            return response_text

    session["conversation_history"].append({"role": "user", "content": user_input})
    result = process_with_llm(user_input, session)
    
    if result.get("features"):
        newly_extracted_features = result["features"]
        session["features"].update(newly_extracted_features)
        session['state'] = 'waiting_for_confirmation'
        
        confirmation_question = (
            f"{result['response']}\n\nJust to confirm, I've understood the following new details:\n"
            f"**{format_features_for_display(newly_extracted_features)}**\n\n"
            f"Is this correct? (yes/no)"
        )
        
        session["conversation_history"].append({"role": "assistant", "content": confirmation_question})
        save_session(chat_id, session)
        return confirmation_question
    
    else:
        session["conversation_history"].append({"role": "assistant", "content": result["response"]})
        save_session(chat_id, session)
        return result['response']