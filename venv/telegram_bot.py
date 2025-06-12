# telegram_bot.py

import logging
import os
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.constants import ParseMode
from dotenv import load_dotenv

import house_price_logic
import data_analyst_tool
import ml_manager

# --- Configuration ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- UI Generation Functions ---
def get_main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔮 Predict a House Price", callback_data='start_prediction_flow')],
        [InlineKeyboardButton("📊 Ask a Data Question", callback_data='start_data_query_flow')],
    ]
    return InlineKeyboardMarkup(keyboard)

def get_feature_selection_keyboard(selected_features: list):
    """Generates a dynamic keyboard with checkmarks for selected features."""
    keyboard = []
    all_features = house_price_logic.SELECTABLE_FEATURES
    for feature in all_features:
        # Add a checkmark if the feature is already selected
        text = f"✅ {feature}" if feature in selected_features else f"🔲 {feature}"
        keyboard.append([InlineKeyboardButton(text, callback_data=f'select_feature_{feature}')])
    
    # Add control buttons
    if selected_features: # Only show confirm button if at least one feature is selected
        keyboard.append([InlineKeyboardButton("➡️ Confirm Selections", callback_data='confirm_selections')])
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data='cancel')])
    return InlineKeyboardMarkup(keyboard)

def get_confirmation_keyboard():
    keyboard = [
        [InlineKeyboardButton("✅ Yes, Predict Now!", callback_data='confirm_values_yes')],
        [InlineKeyboardButton("✏️ No, Re-enter Values", callback_data='confirm_values_no')],
        [InlineKeyboardButton("❌ Cancel", callback_data='cancel')]
    ]
    return InlineKeyboardMarkup(keyboard)


# --- Command & Callback Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the main menu."""
    house_price_logic.reset_session(update.effective_chat.id)
    await update.message.reply_text(
        "Welcome to the **Properlytics Bot**! 🤖\n\nI'm your AI real estate assistant. Please choose an option:",
        reply_markup=get_main_menu_keyboard(),
        parse_mode=ParseMode.MARKDOWN
    )

async def button_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles ALL button presses and drives the UI state."""
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    callback_data = query.data
    session = house_price_logic.load_session(chat_id)

    # --- Main Menu Flows ---
    if callback_data == 'start_prediction_flow':
        session['flow'] = 'feature_selection'
        session['selected_features'] = []
        session['feature_values'] = {}
        house_price_logic.save_session(chat_id, session)
        await query.edit_message_text(
            text="👇 **Step 1: Select Features**\nTap the features you want to provide values for.",
            reply_markup=get_feature_selection_keyboard(session['selected_features'])
        )

    elif callback_data == 'start_data_query_flow':
        session['flow'] = 'data_query'
        house_price_logic.save_session(chat_id, session)
        await query.edit_message_text("📈 Okay, I'm ready. What is your factual question about the housing dataset?")

    # --- Feature Selection Flow ---
    elif callback_data.startswith('select_feature_'):
        feature = callback_data.split('_', 2)[-1]
        if feature in session['selected_features']:
            session['selected_features'].remove(feature)
        else:
            session['selected_features'].append(feature)
        house_price_logic.save_session(chat_id, session)
        await query.edit_message_text(
            text="👇 **Step 1: Select Features**\nTap the features you want to provide values for.",
            reply_markup=get_feature_selection_keyboard(session['selected_features'])
        )

    # --- Value Entry Flow ---
    elif callback_data == 'confirm_selections':
        if session.get('selected_features'):
            session['flow'] = 'awaiting_feature_values'
            house_price_logic.save_session(chat_id, session)
            feature_list_str = ", ".join(session['selected_features'])
            await query.edit_message_text(f"✍️ **Step 2: Provide Values**\nPlease type the values for the following features in a single message:\n\n`{feature_list_str}`")
        else:
            await context.bot.answer_callback_query(query.id, "Please select at least one feature first.", show_alert=True)
            
    # --- Confirmation Flow ---
    elif callback_data == 'confirm_values_yes':
        prediction = ml_manager.predict_price(session['feature_values'])
        summary_text = "\n".join([f"✅ *{key}:* `{value}`" for key, value in session['feature_values'].items()])
        await query.edit_message_text(
            text=f"**Prediction Complete!** 🔮\n\n{summary_text}\n\nEstimated Price: **${prediction:,.2f}**",
            parse_mode=ParseMode.MARKDOWN
        )
        house_price_logic.reset_session(chat_id) # End of flow

    elif callback_data == 'confirm_values_no':
        # Send user back to value entry step
        session['flow'] = 'awaiting_feature_values'
        house_price_logic.save_session(chat_id, session)
        feature_list_str = ", ".join(session['selected_features'])
        await query.edit_message_text(f"✍️ No problem. Let's try again.\nPlease type the values for:\n\n`{feature_list_str}`")
        
    # --- Cancel Flow ---
    elif callback_data == 'cancel':
        house_price_logic.reset_session(chat_id)
        await query.edit_message_text(
            "Action cancelled. What would you like to do next?",
            reply_markup=get_main_menu_keyboard()
        )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles text input, routing it based on the current session 'flow'."""
    chat_id = update.effective_chat.id
    user_text = update.message.text
    session = house_price_logic.load_session(chat_id)

    if session.get("flow") == "awaiting_feature_values":
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        parsed_values = house_price_logic.parse_feature_values_from_text(user_text, session['selected_features'])
        
        if not parsed_values:
            await update.message.reply_text("I had trouble understanding those values. Please try phrasing them differently (e.g., 'Rooms is 5, Age is 12').")
            return
            
        session['feature_values'] = parsed_values
        session['flow'] = 'awaiting_confirmation'
        house_price_logic.save_session(chat_id, session)
        
        summary_text = "\n".join([f"🔹 *{key}:* `{value}`" for key, value in parsed_values.items()])
        await update.message.reply_text(
            f"👍 **Step 3: Confirm Values**\n\nI've understood the following:\n{summary_text}\n\nIs this correct?",
            reply_markup=get_confirmation_keyboard(),
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        # --- FIX FOR DATA QUERY ---
        # If not in a specific flow, use the smart router for any text message
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        intent = house_price_logic.smart_router_llm(user_text)
        if intent == "data_query":
            response = data_analyst_tool.query_housing_data(user_text)
            await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(
                "I'm ready for your command! Please use the menu to get started.",
                reply_markup=get_main_menu_keyboard()
            )

# --- Main Bot Execution ---
def main() -> None:
    logger.info("Starting Interactive Chat App Bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    application.run_polling()

if __name__ == "__main__":
    main()