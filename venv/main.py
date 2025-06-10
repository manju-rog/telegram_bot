from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from session_utils import load_session, save_session
from gemini_utils import extract_features_from_text
from bot_utils import predict_price

# Load the session
session = load_session()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am your House Price Bot. Tell me about a property!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    # Add user query to conversation history
    session["conversation_history"].append({"role": "user", "content": user_text})

    # Extract features from user input using Gemini API
    features = extract_features_from_text(user_text)
    if features:  # If house features are found
        session["features"].update(features)  # Update session with new features
        price = predict_price(features)  # Predict the price using model
        await update.message.reply_text(f"🏠 Estimated Price: ₹{price:,.2f}")
    else:  # If no features found, request more info
        await update.message.reply_text("🔍 I couldn't find enough information. Can you describe the property again?")

    # Save session after processing the message
    save_session(session)

def main():
    app = Application.builder().token("7281474695:AAH-Pgb5kufmMTbwEI9OSt3nhys5nuPU5bk").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot started and listening for messages...")
    app.run_polling()

if __name__ == "__main__":
    main()
