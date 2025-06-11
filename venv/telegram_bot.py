# telegram_bot.py

import logging
import os
import uuid
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv
from pydub import AudioSegment
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# --- NEW: Import Vosk for free speech-to-text ---
from vosk import Model, KaldiRecognizer

# Import our existing backend logic
import house_price_logic
import ml_manager

# --- Configuration and Initialization ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID")

if not all([TELEGRAM_TOKEN, ADMIN_CHAT_ID]):
    raise ValueError("TELEGRAM_TOKEN and ADMIN_CHAT_ID must be set in .env file!")

# --- NEW: Initialize the Vosk model ---
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError("Vosk model not found. Please run download_vosk_model.py first.")
vosk_model = Model(VOSK_MODEL_PATH)

# Create media directory if it doesn't exist
os.makedirs("generated_media", exist_ok=True)

# Logging setup (unchanged)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


# --- Multimodal Helper Functions (Unchanged, they are already free) ---

async def text_to_speech(text, file_path):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(file_path)
        return True
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        return False

def create_prediction_image(price_text):
    try:
        img_path = f"generated_media/{uuid.uuid4()}.png"
        font_path = "font.ttf"
        width, height = 800, 400
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        for i in range(height):
            r, g, b = int(25 + (i/height)*45), int(25 + (i/height)*45), int(50 + (i/height)*100)
            draw.line([(0, i), (width, i)], fill=(r,g,b))
            
        title_font = ImageFont.truetype(font_path, 40)
        price_font = ImageFont.truetype(font_path, 72)
        draw.text((50, 50), "Estimated Property Value:", font=title_font, fill=(255, 255, 255))
        draw.text((width/2, height/2), price_text, font=price_font, fill=(220, 220, 220), anchor="mm")
        
        img.save(img_path)
        return img_path
    except Exception as e:
        logger.error(f"Error creating prediction image: {e}")
        return None

# --- Core Logic Processing Function (Unchanged) ---
async def process_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE, user_input_text: str):
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    response_text = house_price_logic.handle_user_message(user_input_text, chat_id)
    
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

    audio_file_path = f"generated_media/{uuid.uuid4()}.mp3"
    if await text_to_speech(response_text.replace('*', ''), audio_file_path):
        with open(audio_file_path, 'rb') as audio_file:
            await context.bot.send_voice(chat_id=chat_id, voice=audio_file)
        os.remove(audio_file_path)

    if "my estimated price is:" in response_text.lower():
        price_text = response_text.split("$")[-1].strip().replace("**", "")
        img_path = create_prediction_image(f"${price_text}")
        if img_path:
            with open(img_path, 'rb') as img_file:
                await context.bot.send_photo(chat_id=chat_id, photo=img_file)
            os.remove(img_path)

# --- Telegram Handlers ---

# --- NEW: Voice Handler using Vosk ---
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles voice messages using the free, local Vosk library."""
    chat_id = update.effective_chat.id
    ogg_path = f"generated_media/{uuid.uuid4()}.ogg"
    
    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        await voice_file.download_to_drive(ogg_path)
        
        # Convert OGG to a WAV format that Vosk understands
        audio = AudioSegment.from_ogg(ogg_path)
        # Set to 16kHz sample rate, 16-bit, mono channel
        audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)

        # Transcribe with Vosk
        recognizer = KaldiRecognizer(vosk_model, 16000)
        recognizer.AcceptWaveform(audio.raw_data)
        result = json.loads(recognizer.FinalResult())
        transcribed_text = result['text']
        
        if not transcribed_text:
            await update.message.reply_text("I heard you, but I couldn't make out any words. Could you please speak a bit more clearly?")
            return

        logger.info(f"Vosk transcribed voice from chat_id {chat_id}: '{transcribed_text}'")
        await update.message.reply_text(f"Heard you! Processing: *\"{transcribed_text}\"*", parse_mode=ParseMode.MARKDOWN)
        
        await process_user_input(update, context, transcribed_text)

    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        await update.message.reply_text("Sorry, I had trouble understanding your voice message. Please try again or type your request.")
    finally:
        # Clean up local files
        if os.path.exists(ogg_path):
            os.remove(ogg_path)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await process_user_input(update, context, update.message.text)

# (Other command handlers like start, reset, predict, retrain are unchanged)
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = "Hello! I'm **Properlytics Bot** 🤖\nI can estimate house prices. You can type or send me a voice message with details like 'A 15 year old house with 6 rooms'. Then use `/predict` when you're ready!"
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.MARKDOWN)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response_text = house_price_logic.reset_session(update.effective_chat.id)
    await update.message.reply_text(response_text, parse_mode=ParseMode.MARKDOWN)

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await process_user_input(update, context, "/predict")

async def retrain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_chat.id)
    if user_id != ADMIN_CHAT_ID:
        await update.message.reply_text("Sorry, this command is for admins only.")
        return
    await update.message.reply_text("Starting model retraining process... This may take a moment.")
    result = ml_manager.retrain_model("data/housing_new_data.csv")
    await update.message.reply_text(result)

# --- Main Bot Execution ---
def main() -> None:
    logger.info("Starting multimodal bot with FREE tools...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("retrain", retrain))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    application.run_polling()

if __name__ == "__main__":
    main()