# telegram_bot.py

import logging
import os
import uuid
import json
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from dotenv import load_dotenv

from pydub import AudioSegment
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from vosk import Model, KaldiRecognizer

import house_price_logic
import data_analyst_tool
import ml_manager

# --- Configuration ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Resource Initialization ---
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError("Vosk model not found. Please download it manually.")
vosk_model = Model(VOSK_MODEL_PATH)
os.makedirs("generated_media", exist_ok=True)


# --- Multimodal Generation Functions ---
async def text_to_speech(text, file_path):
    try:
        clean_text = text.replace('*', '').replace('`', '').replace('✅', 'Feature').replace('💰', '').replace('🏠', '')
        tts = gTTS(text=clean_text, lang='en', slow=False)
        tts.save(file_path)
        return True
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        return False

# --- FIXED: Image Generation with Absolute Path ---
def create_prediction_image(price_text):
    try:
        img_path = f"generated_media/{uuid.uuid4()}.png"
        
        # --- ROBUST FONT PATH ---
        # Get the directory where the script is running
        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(script_dir, "font.ttf")
        
        if not os.path.exists(font_path):
            logger.error(f"Font file not found at {font_path}!")
            return None

        width, height = 800, 400
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        for i in range(height):
            r, g, b = int(25+(i/height)*45), int(25+(i/height)*45), int(50+(i/height)*100)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
            
        title_font = ImageFont.truetype(font_path, 40)
        price_font = ImageFont.truetype(font_path, 72)
        draw.text((50, 50), "Estimated Property Value:", font=title_font, fill=(255, 255, 255))
        draw.text((width/2, height/2), price_text, font=price_font, fill=(220, 220, 220), anchor="mm")
        
        img.save(img_path)
        return img_path
    except Exception as e:
        logger.error(f"Error creating prediction image: {e}")
        return None

# --- UI Generation (Unchanged) ---
def get_main_menu_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔮 Predict a House Price", callback_data='start_prediction_flow')],
                                 [InlineKeyboardButton("📊 Ask a Data Question", callback_data='start_data_query_flow')]])
def get_feature_selection_keyboard(selected_features: list):
    keyboard = []
    for feature in house_price_logic.SELECTABLE_FEATURES:
        text = f"✅ {feature}" if feature in selected_features else f"🔲 {feature}"
        keyboard.append([InlineKeyboardButton(text, callback_data=f'select_feature_{feature}')])
    if selected_features:
        keyboard.append([InlineKeyboardButton("➡️ Confirm Selections", callback_data='confirm_selections')])
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data='cancel')])
    return InlineKeyboardMarkup(keyboard)
def get_confirmation_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("✅ Yes, Predict Now!", callback_data='confirm_values_yes')],
                                 [InlineKeyboardButton("✏️ No, Re-enter Values", callback_data='confirm_values_no')],
                                 [InlineKeyboardButton("❌ Cancel", callback_data='cancel')]])


# --- Command & Callback Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    house_price_logic.reset_session(update.effective_chat.id)
    await update.message.reply_text("Welcome to the **Properlytics Bot**! 🤖\n\nPlease choose an option:", reply_markup=get_main_menu_keyboard(), parse_mode=ParseMode.MARKDOWN)

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # This entire function is unchanged from the previous version. It handles the button-driven UI flow.
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    callback_data = query.data
    session = house_price_logic.load_session(chat_id)
    if callback_data == 'start_prediction_flow':
        session.update({"flow": 'feature_selection', "selected_features": [], "feature_values": {}})
        house_price_logic.save_session(chat_id, session)
        await query.edit_message_text(text="👇 **Step 1: Select Features**\nTap the features you want to provide values for.", reply_markup=get_feature_selection_keyboard([]))
    elif callback_data == 'start_data_query_flow':
        session['flow'] = 'data_query'
        house_price_logic.save_session(chat_id, session)
        await query.edit_message_text(text="📈 Okay, I'm ready. What is your factual question about the housing dataset?")
    elif callback_data.startswith('select_feature_'):
        feature = callback_data.split('_', 2)[-1]
        if feature in session['selected_features']: session['selected_features'].remove(feature)
        else: session['selected_features'].append(feature)
        house_price_logic.save_session(chat_id, session)
        await query.edit_message_text(text="👇 **Step 1: Select Features**\nTap the features you want to provide values for.", reply_markup=get_feature_selection_keyboard(session['selected_features']))
    elif callback_data == 'confirm_selections':
        if session.get('selected_features'):
            session['flow'] = 'awaiting_feature_values'
            house_price_logic.save_session(chat_id, session)
            feature_list_str = ", ".join(f"`{f}`" for f in session['selected_features'])
            await query.edit_message_text(f"✍️ **Step 2: Provide Values**\nPlease type or send a voice message with the values for:\n\n{feature_list_str}", parse_mode=ParseMode.MARKDOWN)
        else:
            await context.bot.answer_callback_query(query.id, "Please select at least one feature first.", show_alert=True)
    elif callback_data == 'confirm_values_yes':
        prediction = ml_manager.predict_price(session['feature_values'])
        summary_text = "\n".join([f"✅ *{key}:* `{value}`" for key, value in session['feature_values'].items()])
        final_text = f"**Prediction Complete!** 🔮\n\n{summary_text}\n\nEstimated Price: **${prediction:,.2f}**"
        await query.edit_message_text(text=final_text, parse_mode=ParseMode.MARKDOWN)
        await send_multimodal_summary(chat_id, context, final_text, f"${prediction:,.2f}")
        house_price_logic.reset_session(chat_id)
    elif callback_data == 'confirm_values_no':
        session['flow'] = 'awaiting_feature_values'
        house_price_logic.save_session(chat_id, session)
        feature_list_str = ", ".join(f"`{f}`" for f in session['selected_features'])
        await query.edit_message_text(f"✍️ No problem. Let's try again.\nPlease type or send a voice message with the values for:\n\n{feature_list_str}", parse_mode=ParseMode.MARKDOWN)
    elif callback_data == 'cancel':
        house_price_logic.reset_session(chat_id)
        await query.edit_message_text(text="Action cancelled. What would you like to do next?", reply_markup=get_main_menu_keyboard())

# --- Text & Voice Input Handling ---
async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await process_input(update, context, update.message.text)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ogg_path = f"generated_media/{uuid.uuid4()}.ogg"
    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        await voice_file.download_to_drive(ogg_path)
        audio = AudioSegment.from_ogg(ogg_path).set_frame_rate(16000).set_sample_width(2).set_channels(1)
        recognizer = KaldiRecognizer(vosk_model, 16000)
        recognizer.AcceptWaveform(audio.raw_data)
        result = json.loads(recognizer.FinalResult())
        transcribed_text = result['text']
        if not transcribed_text:
            await update.message.reply_text("I heard you, but couldn't make out any words. Please try again.")
            return
        await update.message.reply_text(f"Heard you! Processing: *\"{transcribed_text}\"*", parse_mode=ParseMode.MARKDOWN)
        await process_input(update, context, transcribed_text)
    except Exception as e:
        logger.error(f"Error handling voice message: {e}")
        await update.message.reply_text("Sorry, I had trouble with that voice message. Please try typing instead.")
    finally:
        if os.path.exists(ogg_path): os.remove(ogg_path)

# --- NEW: Central Input Processor with Shortcut Logic ---
async def process_input(update: Update, context: ContextTypes.DEFAULT_TYPE, text_input: str):
    """Central processing function for both text and transcribed voice."""
    chat_id = update.effective_chat.id
    session = house_price_logic.load_session(chat_id)

    # If user is in the wizard flow, process their value submission
    if session.get("flow") == "awaiting_feature_values":
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        parsed_values = house_price_logic.parse_feature_values_from_text(text_input, session['selected_features'])
        if not parsed_values or len(parsed_values) != len(session['selected_features']):
            await update.message.reply_text("I had trouble understanding all the values. Please try phrasing them differently (e.g., 'Rooms is 5, Age is 12').")
            return
        session['feature_values'] = parsed_values
        session['flow'] = 'awaiting_confirmation'
        house_price_logic.save_session(chat_id, session)
        summary_text = "\n".join([f"🔹 *{key}:* `{value}`" for key, value in parsed_values.items()])
        await update.message.reply_text(f"👍 **Step 3: Confirm Values**\n\nI've understood the following:\n{summary_text}\n\nIs this correct?", reply_markup=get_confirmation_keyboard(), parse_mode=ParseMode.MARKDOWN)
    
    # If not in a flow, use the smart router
    else:
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        intent = house_price_logic.smart_router_llm(text_input)
        
        if intent == "data_query":
            response = data_analyst_tool.query_housing_data(text_input)
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
            await update.message.reply_text("What would you like to do next?", reply_markup=get_main_menu_keyboard())

        elif intent == "prediction_query":
            # --- THIS IS THE NEW SHORTCUT PATH ---
            logger.info("Direct prediction query detected. Bypassing wizard.")
            features = house_price_logic.parse_feature_values_from_text(text_input, house_price_logic.SELECTABLE_FEATURES)
            if not features:
                await update.message.reply_text("I think you're asking for a price, but I couldn't extract any specific features. Could you be more descriptive?")
                return
            
            prediction = ml_manager.predict_price(features)
            summary_text = "\n".join([f"✅ *{key}:* `{value}`" for key, value in features.items()])
            final_text = f"**Direct Prediction Complete!** 🔮\n\nBased on your query, I found these details:\n{summary_text}\n\nEstimated Price: **${prediction:,.2f}**"
            await update.message.reply_text(final_text, parse_mode=ParseMode.MARKDOWN)
            await send_multimodal_summary(chat_id, context, final_text, f"${prediction:,.2f}")

        else: # "other" intent
            await update.message.reply_text("I'm ready for your command! Please use the menu to get started.", reply_markup=get_main_menu_keyboard())

# --- Final Multimodal Summary Function ---
async def send_multimodal_summary(chat_id, context, text_summary, price_str):
    """Sends the TTS audio and prediction image after a successful prediction."""
    audio_path = f"generated_media/{uuid.uuid4()}.mp3"
    if await text_to_speech(text_summary, audio_path):
        with open(audio_path, 'rb') as audio_file:
            await context.bot.send_voice(chat_id=chat_id, voice=audio_file)
        os.remove(audio_path)
    
    img_path = create_prediction_image(price_str)
    if img_path:
        with open(img_path, 'rb') as img_file:
            await context.bot.send_photo(chat_id=chat_id, photo=img_file)
        os.remove(img_path)

# --- Main Bot Execution ---
def main() -> None:
    logger.info("Starting Final Interactive Multimodal Bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.run_polling()

if __name__ == "__main__":
    main()