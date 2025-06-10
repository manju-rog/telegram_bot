# telegram_bot.py

import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv

# Import the main handler function from your logic file
import house_price_logic

# --- Configuration and Logging ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file!")

# Set up more professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Bot Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message."""
    welcome_message = (
        "Hello! I'm the **Properlytics Bot** 🤖\n\n"
        "I can help you estimate house prices based on the California Housing dataset. "
        "Just give me some details to get started, like:\n"
        "- 'A house that is 15 years old with 6 rooms.'\n"
        "- 'The median income in the area is 8.5.'\n\n"
        "Type `/reset` anytime to start over."
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /reset command."""
    chat_id = update.effective_chat.id
    response_text = house_price_logic.reset_session(chat_id)
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

# --- Main Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all non-command text messages."""
    if not update.message or not update.message.text:
        return

    user_input = update.message.text
    chat_id = update.effective_chat.id
    
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    logger.info(f"Received message from chat_id {chat_id}")
    
    # Get the response from the backend logic
    response_text = house_price_logic.handle_user_message(user_input, chat_id)
    
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

# --- Main Bot Execution ---
def main() -> None:
    """Start the bot."""
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.run_polling()

if __name__ == "__main__":
    main()