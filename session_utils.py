import json
import os

session_file = "house_assistant_session.json"

def load_session():
    """Load the session data or create a new session."""
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            session = json.load(f)
        print("Loaded existing session with features:", list(session["features"].keys()))
    else:
        session = {
            "features": {},
            "conversation_history": []
        }
        print("Started new session")
    return session

def save_session(session):
    """Save the session data to file."""
    with open(session_file, 'w') as f:
        json.dump(session, f)
    print("Session saved")
