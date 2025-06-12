# Properlytics Bot: The Ultimate AI House price prediction Assistant 🤖🏠

Welcome to Properlytics Bot, a sophisticated, multimodal Telegram bot that serves as an intelligent real estate assistant. This project demonstrates a complete, end-to-end implementation of a modern AI chatbot, evolving from a simple prediction script to a state-driven, interactive "chat app" with multiple advanced capabilities.

This bot is the culmination of a guided development process, showcasing how to build a robust, user-friendly, and powerful AI product.

## ✨ Key Features: A Futuristic Chat Experience

This is not just a chatbot; it's a feature-rich application built inside Telegram.

*   **Interactive UI with Inline Buttons:** Forget command-line interfaces. The bot is driven by a sleek, button-based UI that guides the user through its functions.
*   **Dynamic Message Editing:** The bot provides a seamless "app-like" experience by editing its messages in-place, creating dynamic forms and live-updating profiles without cluttering the chat.
*   **Multi-Select Prediction Wizard:** Users can select exactly which housing features they want to provide from a checklist, giving them full control over the prediction process.
*   **"Single Go" Data Entry:** After selecting features, the bot intelligently prompts for all values in a single, easy-to-understand step.
*   **Dual-Capability "Hybrid" AI:** The bot seamlessly integrates two powerful AI functionalities:
    1.  **ML Price Prediction:** Uses a trained `RandomForestRegressor` model to estimate the price of hypothetical houses based on user-provided features.
    2.  **Natural Language Data Querying:** Uses a Gemini-powered "Data Analyst" to answer factual questions about the underlying housing dataset by converting natural language into executable Pandas code.
*   **Admin-Only Model Retraining:** Includes a secure `/retrain` command for an admin to incrementally update the ML model with new data, demonstrating a core MLOps concept.
*   **Professional Architecture:** Built with a clean separation of concerns:
    *   `telegram_bot.py`: Manages the entire UI and user interaction state.
    *   `house_price_logic.py`: A lean session manager and feature parser.
    *   `data_analyst_tool.py`: A powerful tool to query the dataset.
    *   `ml_manager.py`: Handles all ML model loading, prediction, and retraining.

## 🚀 Getting Started

Follow these steps to get your own instance of Properlytics Bot running.

### 1. Prerequisites

*   Python 3.9+
*   A Telegram account and the Telegram app.
*   A Google AI (Gemini) API Key.

### 2. Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/your-username/properlytics-bot.git
cd properlytics-bot
