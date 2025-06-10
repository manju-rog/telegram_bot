# telegram_bot.py

import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# Import the main handler function from your logic file
import house_price_logic

# --- Telegram Bot Configuration ---
# IMPORTANT: Paste your Telegram Bot Token here.
TELEGRAM_TOKEN = "7281474695:AAH-Pgb5kufmMTbwEI9OSt3nhys5nuPU5bk"

# Enable logging to see errors
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Bot Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    welcome_message = (
        "Welcome to the House Price Prediction Assistant! 🤖\n\n"
        "You can ask me to estimate house prices. For example:\n"
        "- 'What's the price of a 10 year old house with 4 rooms in California?'\n"
        "- 'The population is 1500.'\n\n"
        "Type `/reset` to clear all information and start a new prediction."
    )
    await update.message.reply_text(welcome_message)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /reset command."""
    chat_id = update.effective_chat.id
    response_text = house_price_logic.reset_session(chat_id)
    await update.message.reply_text(response_text)

# --- Main Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all non-command text messages from the user."""
    user_input = update.message.text
    chat_id = update.effective_chat.id
    
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    logger.info(f"Received message from chat_id {chat_id}: '{user_input}'")
    
    # Get the response from your backend logic
    response_text = house_price_logic.handle_user_message(user_input, chat_id)
    
    # Send the final response back to the user in Telegram
    # Use parse_mode='Markdown' to make the price bold and format text
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

# --- Main Bot Execution ---
def main() -> None:
    """Start the bot."""
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running... Press Ctrl-C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()