import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import subprocess
import json
import os
import tempfile
import traceback

# Enable detailed logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('telegram_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Your Telegram Bot Token (get from @BotFather)
BOT_TOKEN = "7281474695:AAH-Pgb5kufmMTbwEI9OSt3nhys5nuPU5bk"

class HousePriceBot:
    def __init__(self):
        self.user_sessions = {}  # Store session data per user
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        user_id = update.effective_user.id
        welcome_message = """
🏠 Welcome to the House Price Prediction Assistant!

I can help you predict house prices based on various features like:
- Location (Latitude/Longitude)
- House age
- Number of bedrooms/rooms
- Income level in the area
- Population density

Just tell me about a property you're interested in, and I'll help estimate its price!

Commands:
/start - Start the bot
/reset - Clear all your current property details
/help - Show this help message
        """
        await update.message.reply_text(welcome_message)
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        help_text = """
🏠 House Price Prediction Bot Help

How to use:
1. Simply describe a property in natural language
2. I'll extract relevant features automatically
3. Once I have enough information, I'll predict the price

Examples:
- "I'm looking at a 5-year-old house in San Francisco with 3 bedrooms"
- "What's the price of a house in LA with latitude 34.05, longitude -118.24?"
- "A property with median income $80k, 2.5 average rooms"

Commands:
/reset - Clear your current session and start over
/help - Show this help
        """
        await update.message.reply_text(help_text)
        
    async def reset_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset user's session."""
        user_id = update.effective_user.id
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        await update.message.reply_text("🔄 Your session has been reset! Tell me about a new property.")
        
    def call_house_assistant(self, user_input, user_id):
        """Call your existing house assistant program."""
        logger.info(f"Processing request for user {user_id}: {user_input}")
        
        try:
            # Create a temporary session file for this user
            session_file = f"telegram_session_{user_id}.json"
            logger.debug(f"Using session file: {session_file}")
            
            # Load existing session for this user if it exists
            if user_id in self.user_sessions:
                logger.debug(f"Loading existing session for user {user_id}")
                with open(session_file, 'w') as f:
                    json.dump(self.user_sessions[user_id], f)
            else:
                logger.debug(f"No existing session for user {user_id}")
            
            # Create a modified version of your script that takes input as argument
            # and returns JSON response
            logger.debug("Creating temporary script...")
            script_content = f'''
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
import sys
import traceback

print("DEBUG: Script started", file=sys.stderr)

try:
    # Your existing model loading code
    model_path = "house_price_model.joblib"
    print(f"DEBUG: Looking for model at {{model_path}}", file=sys.stderr)
    
    if not os.path.exists(model_path):
        print("DEBUG: Training new model...", file=sys.stderr)
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        feature_names = data.feature_names
        target_name = "MedHouseVal"
        X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df[target_name], test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print("DEBUG: Model trained and saved", file=sys.stderr)
    else:
        print("DEBUG: Loading existing model...", file=sys.stderr)
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        feature_names = data.feature_names

    mdl = joblib.load(model_path)
    print("DEBUG: Model loaded successfully", file=sys.stderr)

    API_KEY = "AIzaSyA9PnAzCJxM7cqxVs5-QM_t5CJCknmxzks"
    ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={{API_KEY}}"

    session_file = "{session_file}"
    print(f"DEBUG: Session file: {{session_file}}", file=sys.stderr)

    # Load session
    if os.path.exists(session_file):
        print("DEBUG: Loading existing session", file=sys.stderr)
        with open(session_file, 'r') as f:
            session = json.load(f)
    else:
        print("DEBUG: Creating new session", file=sys.stderr)
        session = {{"features": {{}}, "conversation_history": []}}

    def process_user_input(user_input):
        print(f"DEBUG: Processing input: {{user_input}}", file=sys.stderr)
        session["conversation_history"].append({{"role": "user", "content": user_input}})
        
        prompt = f\"\"\"
You are a helpful house price prediction assistant. Your job is to:

1) Determine if the user's query is related to house prices or properties
2) If it IS about house prices/properties, extract the following features:
   - MedInc: Median income in block group (number)
   - HouseAge: Median house age in block group (number, in years)
   - AveRooms: Average rooms per household (number)
   - AveBedrms: Average bedrooms per household (number)
   - Population: Population per block group (number)
   - AveOccup: Average occupants per household (number)
   - Latitude: Geographic coordinate (number)
   - Longitude: Geographic coordinate (number)

3) If it is NOT about house prices, respond conversationally with a brief, friendly message that redirects to house prices.

Here's the most recent user query: "{{user_input}}"

Current features I already know: {{json.dumps(session["features"], indent=2)}}

Respond with a JSON object that has:
1. "is_house_query": true/false - whether this is a house-related query
2. "response": your conversational response to the user
3. "features": a dictionary of any house features you extracted (empty if none or not a housing query)
\"\"\"

        headers = {{"Content-Type": "application/json"}}
        data = {{
            "contents": [
                {{
                    "parts": [
                        {{
                            "text": prompt
                        }}
                    ]
                }}
            ]
        }}

        try:
            print("DEBUG: Making API request...", file=sys.stderr)
            response = requests.post(ENDPOINT, headers=headers, json=data, timeout=30)
            print(f"DEBUG: API response status: {{response.status_code}}", file=sys.stderr)
            response.raise_for_status()
            output = response.json()
            llm_response = output["candidates"][0]["content"]["parts"][0]["text"]
            print(f"DEBUG: LLM response: {{llm_response[:200]}}...", file=sys.stderr)
            
            json_match = re.search(r'{{.*}}', llm_response, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group(0))
                print(f"DEBUG: Parsed result: {{result}}", file=sys.stderr)
                if "features" in result and isinstance(result["features"], dict):
                    session["features"].update(result["features"])
                session["conversation_history"].append({{"role": "assistant", "content": result["response"]}})
                return result
            else:
                print("DEBUG: No JSON found in response", file=sys.stderr)
                return {{
                    "is_house_query": False,
                    "response": "I'm having trouble understanding. Can you tell me about the house you're interested in?",
                    "features": {{}}
                }}
        except Exception as e:
            print(f"DEBUG: API Error: {{str(e)}}", file=sys.stderr)
            print(f"DEBUG: API Error traceback: {{traceback.format_exc()}}", file=sys.stderr)
            return {{
                "is_house_query": False,
                "response": f"API Error: {{str(e)}}",
                "features": {{}}
            }}

    def predict_house_price():
        print("DEBUG: Predicting house price...", file=sys.stderr)
        if not session["features"]:
            print("DEBUG: No features available", file=sys.stderr)
            return None
        
        model_input = {{}}
        for feature in feature_names:
            if feature in session["features"]:
                model_input[feature] = session["features"][feature]
            else:
                model_input[feature] = df[feature].mean()
        
        print(f"DEBUG: Model input: {{model_input}}", file=sys.stderr)
        df_input = pd.DataFrame([model_input], columns=feature_names)
        prediction = mdl.predict(df_input)[0]
        print(f"DEBUG: Raw prediction: {{prediction}}", file=sys.stderr)
        return prediction * 100000

    # Process the input
    user_input = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"DEBUG: User input from argv: {{user_input}}", file=sys.stderr)
    
    result = process_user_input(user_input)
    print(f"DEBUG: Process result: {{result}}", file=sys.stderr)

    # Add prediction if applicable
    if result["is_house_query"] and len(session["features"]) >= 2:
        print("DEBUG: Adding prediction...", file=sys.stderr)
        prediction = predict_house_price()
        if prediction is not None:
            result["prediction"] = prediction

    # Save session
    print("DEBUG: Saving session...", file=sys.stderr)
    with open(session_file, 'w') as f:
        json.dump(session, f)

    # Output result as JSON
    result["current_features"] = session["features"]
    print(json.dumps(result))
    
except Exception as e:
    print(f"DEBUG: Script error: {{str(e)}}", file=sys.stderr)
    print(f"DEBUG: Script traceback: {{traceback.format_exc()}}", file=sys.stderr)
    print(json.dumps({{
        "is_house_query": False,
        "response": f"Script Error: {{str(e)}}",
        "features": {{}}
    }}))
'''
            
            # Write the script to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
                temp_script.write(script_content)
                temp_script_path = temp_script.name
            
            try:
                # Run the script with user input
                result = subprocess.run(
                    ['python', temp_script_path, user_input],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    response_data = json.loads(result.stdout.strip())
                    
                    # Update user session
                    if os.path.exists(session_file):
                        with open(session_file, 'r') as f:
                            self.user_sessions[user_id] = json.load(f)
                        os.remove(session_file)  # Clean up
                    
                    return response_data
                else:
                    return {
                        "is_house_query": False,
                        "response": "Sorry, I encountered an error processing your request.",
                        "features": {}
                    }
                    
            finally:
                # Clean up temporary script
                if os.path.exists(temp_script_path):
                    os.remove(temp_script_path)
                    
        except Exception as e:
            logger.error(f"Error calling house assistant: {e}")
            return {
                "is_house_query": False,
                "response": "Sorry, I'm having technical difficulties. Please try again.",
                "features": {}
            }
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages."""
        user_id = update.effective_user.id
        user_input = update.message.text
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Call your house assistant
        result = self.call_house_assistant(user_input, user_id)
        
        # Format response
        response_text = result["response"]
        
        # Add prediction if available
        if "prediction" in result:
            response_text += f"\n\n💰 **Estimated Price: ${result['prediction']:,.2f}**"
        
        # Add current features if any
        if result.get("current_features"):
            features_text = "\n\n📋 **Current Property Details:**\n"
            for feature, value in result["current_features"].items():
                features_text += f"• {feature}: {value}\n"
            response_text += features_text
        
        # Add guidance if needed
        if result["is_house_query"] and len(result.get("current_features", {})) < 2:
            response_text += "\n\n💡 *Tip: Provide more details for a more accurate prediction!*"
        
        await update.message.reply_text(response_text, parse_mode='Markdown')

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Create bot instance
    bot = HousePriceBot()
    
    # Register handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("reset", bot.reset_session))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()