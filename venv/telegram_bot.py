import logging
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv

import house_price_logic
import ml_manager

# --- Configuration and Logging ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")
if not all([TELEGRAM_TOKEN, ADMIN_CHAT_ID]):
    raise ValueError("TELEGRAM_TOKEN and ADMIN_CHAT_ID must be set in .env file!")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = "Hello! I'm **Properlytics Bot** 🤖\nI can estimate house prices. Give me details like 'A 15 year old house with 6 rooms', then type `/predict` when you're ready!"
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response_text = house_price_logic.reset_session(update.effective_chat.id)
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response_text = house_price_logic.handle_user_message("/predict", update.effective_chat.id)
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin-only command to retrain the model."""
    user_id = str(update.effective_chat.id)
    if user_id != ADMIN_CHAT_ID:
        await update.message.reply_text("Sorry, this command is for admins only.")
        return
    
    await update.message.reply_text("Starting model retraining process... This may take a moment.")
    result = ml_manager.retrain_model("housing_new_data.csv")
    await update.message.reply_text(result)

# --- Main Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    response_text = house_price_logic.handle_user_message(update.message.text, update.effective_chat.id)
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

# --- Main Bot Execution ---
def main() -> None:
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("retrain", retrain)) # New admin command
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))  
    application.run_polling()

if __name__ == "__main__":
    main()