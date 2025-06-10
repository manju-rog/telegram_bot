# test_session_utils.py

from session_utils import load_session, save_session

session = load_session()
session["conversation_history"].append({"role": "user", "content": "I want a house with 2 bedrooms"})
save_session(session)
