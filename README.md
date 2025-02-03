# 🤖 Telegram AI Bot

A smart AI-powered Telegram bot that supports **text chat, image generation, voice recognition, and pronunciation checking** using OpenAI API and Whisper.

## ✨ Features:
- 📝 **Text Chat**: Chat with an AI assistant in natural language.
- 🎙 **Pronunciation Check**: Get feedback on your English pronunciation.
- 🖼 **Image Generation**: Generate images using DALL·E based on user prompts.
- 🗣 **Conversation Mode**: Engage in a conversation for English learning practice.
- 📊 **Analytics**: Track user activity and interactions.

## 🛠 Tech Stack:
- **Python** (asyncio, aiogram, OpenAI API)
- **Speech Recognition** (Whisper, pydub)
- **Text-to-Speech** (gTTS)
- **Data Analytics** (JSON logging, user tracking)

---

## 🚀 Installation Guide

### **1️⃣ Clone the Repository**
sh
git clone https://github.com/yourusername/telegram-ai-bot.git
cd telegram-ai-bot

2️⃣ Install Dependencies
pip install -r requirements.txt


3️⃣ Set Up Environment Variables
Create a .env file in the root directory and add the following:
TELEGRAM_BOT_TOKEN=your_telegram_token
OPENAI_API_KEY=your_openai_api_key
OWNER_ID=your_telegram_id

4️⃣ Run the Bot
python bot/main.py

⚠️ Important Notes
Keep your .env file private (it contains API keys).
The bot requires OpenAI API access for chat and image generation.
Ensure ffmpeg is installed for speech recognition.

🎯 Future Enhancements
✅ Improve voice-to-text accuracy
✅ Add grammar correction in conversation mode
🚀 Expand analytics dashboard with visualization

Author: Rafael Dzhabrailov Contact: 📩 Email: rafaelrafael8879@gmail.com 
🔗 LinkedIn: www.linkedin.com/in/rafael-dzhabrailov-756716330 
GitHub: https://github.com/raf8879 
Feel free to reach out with questions, suggestions, or feedback!


📜 License
This project is licensed under the MIT License. Feel free to use and modify it!