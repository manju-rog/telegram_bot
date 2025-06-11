# telegram_bot.py

import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv

# Import our final, ultimate logic handler
import house_price_logic

# --- Configuration and Logging ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN must be set in .env file!")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = (
        "Hello! I am the ultimate Hybrid Housing Bot. I can help you in two ways:\n\n"
        "1. **Ask me factual questions:** 'What is the average house age?' or 'Show me the 5 cheapest houses'.\n\n"
        "2. **Build a house profile:** 'A house with 3 bedrooms and 5 rooms'. I'll give you a price estimate that updates with every detail you add!\n\n"
        "Type `/reset` to clear the house profile and start over."
    )
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response_text = house_price_logic.reset_session(update.effective_chat.id)
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

# --- Main Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    
    user_input = update.message.text
    chat_id = update.effective_chat.id
    
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    logger.info(f"Received query from chat_id {chat_id}: '{user_input}'")
    
    response_text = house_price_logic.handle_user_message(user_input, chat_id)
    
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

# --- Main Bot Execution ---
def main() -> None:
    logger.info("Starting Ultimate Hybrid Bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.run_polling()

if __name__ == "__main__":
    main()