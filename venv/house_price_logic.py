import requests
import json
import re
import os
import logging
from dotenv import load_dotenv
import ml_manager # Import our new ML manager

# --- Configuration ---
load_dotenv()
logger = logging.getLogger(__name__)
API_KEY = os.getenv("GEMINI_API_KEY")
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

# --- Session Management (Unchanged) ---
def load_session(chat_id):
    session_file = f"session_{chat_id}.json"
    if os.path.exists(session_file):
        with open(session_file, 'r') as f: session = json.load(f)
    else:
        session = {"features": {}, "conversation_history": []}
    session.setdefault('state', 'gathering_info')
    return session

def save_session(chat_id, session_data):
    with open(f"session_{chat_id}.json", 'w') as f: json.dump(session_data, f, indent=4)

def reset_session(chat_id):
    if os.path.exists(f"session_{chat_id}.json"): os.remove(f"session_{chat_id}.json")
    return "Memory wiped! I'm ready for a new property. What do you have in mind?"

# --- Core Logic ---
def format_features_for_display(feature_dict):
    if not feature_dict: return "No features specified yet."
    return "\n".join([f"- {key}: `{value}`" for key, value in feature_dict.items()])

def process_with_llm(user_input, session):
    """The same production-grade LLM prompt from before."""
    prompt = f"""
    You are the "Properlytics Bot", a specialized AI assistant for house price prediction. Your personality is helpful, professional, and slightly witty.

    **YOUR CORE INSTRUCTIONS:**

    1.  **MAINTAIN YOUR PERSONA:** You ONLY discuss real estate. Gracefully deflect all other topics using varied, polite responses.
    2.  **EXTRACT & MODIFY FEATURES:** Your main job is to extract or modify these features: {', '.join(ml_manager.FEATURE_NAMES)}. If a user says "change rooms to 6", update the AveRooms feature.
    3.  **HANDLE OUT-OF-CONTEXT QUERIES:** If the user asks about anything other than housing, you MUST use a deflection like: "That's a bit outside my floor plan! Let's get back to real estate."
    4.  **HANDLE VAGUE QUERIES:** If a query is about housing but too vague (e.g., "is my house expensive?"), ask clarifying questions to get concrete features.

    **CURRENT CONVERSATION:**

    *   **User's Most Recent Message:** "{user_input}"
    *   **Features I Already Know:** {json.dumps(session.get("features", {}))}

    ---
    **YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT with these keys:**
    1.  "is_house_query": boolean
    2.  "response": Your conversational response.
    3.  "features": A dictionary of any NEW or MODIFIED features you extracted.
    """
    # ... (The rest of the LLM call logic remains the same as the previous version)
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(ENDPOINT, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        output = response.json()
        llm_response_text = output["candidates"][0]["content"]["parts"][0]["text"]
        json_match = re.search(r'\{.*\}', llm_response_text, re.DOTALL)
        if json_match: return json.loads(json_match.group(0))
        else:
            logger.warning(f"LLM did not return valid JSON. Response: {llm_response_text}")
            return {"is_house_query": False, "response": "I seem to be having a little trouble processing that. Could you please rephrase?", "features": {}}
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Gemini API: {e}")
        return {"is_house_query": False, "response": "I'm having trouble connecting to my services right now. Please try again in a moment.", "features": {}}
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return {"is_house_query": False, "response": "I received an unusual response. Let's try that again.", "features": {}}

# --- Main Handler (Completely Refactored for Better UX) ---
def handle_user_message(user_input, chat_id):
    session = load_session(chat_id)
    session["conversation_history"].append({"role": "user", "content": user_input})

    # Handle explicit commands first
    if user_input.lower().strip() == '/reset':
        return reset_session(chat_id)
    
    if user_input.lower().strip() == '/predict':
        if len(session['features']) < 2:
            return "I need at least two features to make a prediction. What else can you tell me about the property?"
        
        prediction = ml_manager.predict_price(session['features'])
        return f"Based on the current features, my estimated price is: **${prediction:,.2f}**"

    # Process natural language with the LLM
    result = process_with_llm(user_input, session)
    session["conversation_history"].append({"role": "assistant", "content": result['response']})

    # If the LLM didn't understand or it was off-topic, just return its response.
    if not result.get("is_house_query"):
        save_session(chat_id, session)
        return result['response']

    # Update features if any were extracted/modified
    if result.get("features"):
        session['features'].update(result['features'])

    # Build a comprehensive response with a summary
    current_features_summary = format_features_for_display(session['features'])
    
    response_text = f"{result['response']}\n\n**Current Property Details:**\n{current_features_summary}"

    # Guide the user on what to do next
    if len(session['features']) >= 2:
        response_text += "\n\nFeel free to add more details, modify existing ones, or type `/predict` to get an estimate."
    else:
        response_text += "\n\nI still need a bit more information to make a good prediction."
        
    save_session(chat_id, session)
    return response_text