import asyncio
import logging
import json
from datetime import datetime, timedelta
import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import openai
from pydub import AudioSegment
from gtts import gTTS
from aiogram.types import FSInputFile
from dotenv import load_dotenv
import os
from difflib import SequenceMatcher
import subprocess
import tempfile
import string
import whisper
from phonemizer import phonemize
from difflib import SequenceMatcher
import Levenshtein
import editdistance
from collections import Counter
import functools
import warnings
from aiogram.types import BotCommand, BotCommandScopeDefault, BotCommandScopeChat



async def set_bot_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="contact", description="Связаться с владельцем")
    ]
    await bot.set_my_commands(commands, scope=BotCommandScopeDefault())

    OWNER_ID = your_telegram_id  # Ваш Telegram ID
    private_commands = commands + [
        BotCommand(command="analytics_1", description="Базовая аналитика"),
        BotCommand(command="analytics_2", description="Детализированная аналитика"),
    ]
    await bot.set_my_commands(private_commands, scope=BotCommandScopeChat(chat_id=OWNER_ID))

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
load_dotenv()  # Загружаем переменные окружения из файла .env

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка токенов из переменных окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)


@dp.message(Command("contact"))
async def send_contact(message: Message):
    await message.answer("Связаться со мной можно, перейдя по ссылке: [Нажмите здесь](https://t.me/raf8879)", parse_mode="Markdown")


class ChatStates(StatesGroup):
    waiting_for_input = State()
    choosing_role = State()
    setting_custom_role = State()
    generating_image = State()
    waiting_for_voice = State()
    choosing_difficulty = State()
    choosing_topic = State()
    waiting_for_mode = State()
    conversation_mode = State()
    pronunciation_mode = State()
    waiting_for_reference_text = State()
    image_generation_options = State()


# Темы для выбора
topics = ["Job Interview", "Weather", "Travel", "About Myself"]

# Уровни сложности
difficulty_levels = {
    "A1": "simple and short sentences with basic vocabulary.",
    "A2": "slightly longer sentences with more varied vocabulary.",
    "B1": "complex sentences with intermediate vocabulary and grammar.",
    "B2": "long and detailed sentences with upper-intermediate vocabulary.",
    "C1": "advanced sentences with idioms and professional terms.",
    "C2": "highly advanced sentences with academic or professional phrasing.",
}

# Предопределенные роли для тестового чата
predefined_roles = {
    "ESL Tutor": "You are an ESL tutor helping students learn English.",
    "Math Teacher": "You are a math teacher assisting with math problems.",
    "Psychologist": "You are a psychologist providing support and guidance on anxiety.",
}

# Словарь для хранения времени последней активности пользователей
user_last_activity = {}

# Файл для хранения аналитических данных
ANALYTICS_FILE = 'analytics.json'

# Функция для загрузки аналитических данных из файла
def load_analytics():
    if os.path.exists(ANALYTICS_FILE):
        with open(ANALYTICS_FILE, 'r') as f:
            analytics_data = json.load(f)
            # Преобразуем множество пользователей из списка
            analytics_data['total_users'] = set(analytics_data['total_users'])

    else:
        analytics_data = {
            'total_users': set(),
            'total_messages': 0,
        }
    return analytics_data

# Функция для сохранения аналитических данных в файл
def save_analytics(analytics_data):
    # Преобразуем множество пользователей в список для сохранения в JSON
    analytics_data_copy = analytics_data.copy()
    analytics_data_copy['total_users'] = list(analytics_data['total_users'])
    with open(ANALYTICS_FILE, 'w') as f:
        json.dump(analytics_data_copy, f)



# Загружаем аналитические данные при запуске бота
analytics_data = load_analytics()


async def clear_user_context():
    while True:
        current_time = datetime.now()
        for user_id in list(user_last_activity.keys()):
            last_active = user_last_activity[user_id]
            if current_time - last_active > timedelta(minutes=15):
                # Используем метод storage.clear_state() для удаления данных состояния
                await storage.clear_state(chat_id=user_id, user_id=user_id)
                logger.info(f"Контекст пользователя {user_id} был автоматически очищен из-за неактивности.")
                del user_last_activity[user_id]
        await asyncio.sleep(60)  # Проверяем каждую минуту

@dp.message(Command('start'))
async def cmd_start(message: Message, state: FSMContext):
    global analytics_data
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    user_id = message.from_user.id

    # Проверяем, является ли пользователь новым
    if user_id not in analytics_data['total_users']:
        analytics_data['total_users'].add(user_id)
        save_analytics(analytics_data)

    # Создаем клавиатуру
    kb_builder = ReplyKeyboardBuilder()
    kb_builder.button(text='📝 Text chat')
    kb_builder.button(text='🖼 Image generation')
    kb_builder.button(text='🗣 Conversation Mode')
    kb_builder.button(text='🎙 Pronunciation Check')  # Новая кнопка
    kb_builder.adjust(2, 2)  # Первая строка - 2 кнопки, вторая - 1 кнопка

    keyboard = kb_builder.as_markup(resize_keyboard=True)

    await message.answer("Welcome! Select a mode of operation:", reply_markup=keyboard)
    await state.clear()


# Меню выбора сложности
@dp.message(F.text == '🎙 Pronunciation Check')
async def pronunciation_menu(message: Message, state: FSMContext):
    update_user_activity(message.from_user.id, message.from_user.full_name, feature='pronunciation_check')
    user_last_activity[message.from_user.id] = datetime.now()

    # Создаем клавиатуру для выбора уровня сложности
    kb_builder = ReplyKeyboardBuilder()
    for level in difficulty_levels.keys():
        kb_builder.button(text=level)
    kb_builder.button(text='🔙 Back')

    # Указываем количество кнопок в строке (2 кнопки на строку)
    kb_builder.adjust(2)

    keyboard = kb_builder.as_markup(resize_keyboard=True)

    await message.answer("Choose your difficulty level:", reply_markup=keyboard)
    await state.set_state(ChatStates.choosing_difficulty)


# Выбор темы после сложности
@dp.message(ChatStates.choosing_difficulty, F.text.in_(difficulty_levels.keys()))
async def choose_difficulty(message: Message, state: FSMContext):
    user_last_activity[message.from_user.id] = datetime.now()

    level = message.text
    await state.update_data(level=level)

    # Темы для практики
    topics = ["Job Interview", "Travel", "About Myself", "Weather"]

    # Создаем клавиатуру
    kb_builder = ReplyKeyboardBuilder()
    for topic in topics:
        kb_builder.button(text=topic)
    kb_builder.button(text='🔙 Back')

    # Указываем количество кнопок в строке (2 кнопки на строку)
    kb_builder.adjust(2)

    keyboard = kb_builder.as_markup(resize_keyboard=True)

    await message.answer("Choose a topic for practice:", reply_markup=keyboard)
    await state.set_state(ChatStates.choosing_topic)
import random
@dp.message(ChatStates.choosing_topic, F.text.in_(topics))
async def generate_practice_sentence(message: Message, state: FSMContext):
    user_data = await state.get_data()
    level = user_data.get("level", "A1")
    topic = message.text
    previous_sentences = user_data.get("previous_sentences", [])

    adjectives = ["interesting", "engaging", "thought-provoking", "funny", "unusual"]
    chosen_adjective = random.choice(adjectives)

    previous_sentences_text = ' | '.join(previous_sentences) if previous_sentences else "None"

    prompt = (
        f"Generate 5 unique and {chosen_adjective} sentences about '{topic}' for English learners at level {level}. "
        f"Each sentence should be {difficulty_levels[level]} "
        f"and different from each other and from these sentences: {previous_sentences_text}. "
        f"Provide the sentences in a numbered list."
    )

    await message.answer(f"Generating a practice sentence for level {level} and topic '{topic}'...")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant creating sentences for English practice."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.9,
            presence_penalty=0.6,
            frequency_penalty=0.5,
        )
        # Обработка ответа
        sentences = response['choices'][0]['message']['content'].strip().split('\n')
        sentences = [s.strip().lstrip('12345. ') for s in sentences if s.strip()]
        practice_sentence = random.choice(sentences)

        # Обновляем список предыдущих предложений
        previous_sentences.append(practice_sentence)
        await state.update_data(previous_sentences=previous_sentences)

        await state.update_data(reference_text=practice_sentence)
        await message.answer(
            f"Here is your practice sentence:\n\n'{practice_sentence}'\n\n"
            "Please send a voice message reading this sentence."
        )
        await state.set_state(ChatStates.waiting_for_voice)
    except Exception as e:
        logging.error(f"Error generating practice sentence: {e}")
        await message.answer("An error occurred. Please try again.")

model = whisper.load_model("base", device="cpu", download_root="./models")






def generate_retry_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Try Again")],
            [KeyboardButton(text="🆕 New Sentence")],
            [KeyboardButton(text="Back")]
        ],
        resize_keyboard=True,  # Устанавливаем размер клавиатуры
        one_time_keyboard=True  # Клавиатура исчезнет после нажатия
    )
    return keyboard


@dp.message(F.text == "🔄 Try Again")
async def try_again(message: Message, state: FSMContext):
    # Получаем текст, который нужно повторить
    user_data = await state.get_data()
    reference_text = user_data.get("reference_text", "No sentence found.")
    # Отправляем текст для повторения
    await message.answer(f"Please repeat the sentence:\n'{reference_text}'")


@dp.message(F.text == "🆕 New Sentence")
async def new_sentence(message: Message, state: FSMContext):
    await pronunciation_menu(message, state)


@dp.message(F.text == "Back")
async def back_to_main_menu(message: Message, state: FSMContext):
    await cmd_start(message, state)


@dp.message(F.text == '📝 Text chat')
async def test_chat_menu(message: Message, state: FSMContext):
    update_user_activity(message.from_user.id, message.from_user.full_name, feature='text_chat')
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    # Создаем клавиатуру
    kb_builder = ReplyKeyboardBuilder()
    for role in predefined_roles.keys():
        kb_builder.button(text=role)
    kb_builder.button(text='🆕 Own role')
    kb_builder.button(text='🔙 Back')
    kb_builder.adjust(2, 2)  # По одной кнопке в строке

    keyboard = kb_builder.as_markup(resize_keyboard=True)

    await message.answer("Choose a role or set your own:", reply_markup=keyboard)
    await state.set_state(ChatStates.choosing_role)


@dp.message(ChatStates.choosing_role, F.text.in_(list(predefined_roles.keys())))
async def set_role(message: Message, state: FSMContext):
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    role = predefined_roles[message.text]
    await state.update_data(role=role, messages=[])
    await message.answer(f"Role set: '{message.text}'. You can start communicating.")
    await state.set_state(ChatStates.waiting_for_input)


@dp.message(ChatStates.choosing_role, F.text == '🆕 Own role')
async def custom_role_prompt(message: Message, state: FSMContext):
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    await message.answer("Please describe the role that ChatGPT should take on:")
    await state.set_state(ChatStates.setting_custom_role)


@dp.message(ChatStates.setting_custom_role)
async def set_custom_role(message: Message, state: FSMContext):
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    role = message.text.strip()
    await state.update_data(role=role, messages=[])
    await message.answer("Custom role set. You can start communicating.")
    await state.set_state(ChatStates.waiting_for_input)


@dp.message(F.text == "🗣 Conversation Mode")
async def conversation_mode(message: Message, state: FSMContext):
    update_user_activity(message.from_user.id, message.from_user.full_name, feature='conversation_mode')
    user_last_activity[message.from_user.id] = datetime.now()
    await message.answer(
        "You have entered Conversation mode for practicing English.\n"
        "The bot will help you with grammar, pronunciation, vocabulary, and communication.\n"
        "If you want to clear the history, enter the /clear command.\n"
        "Start the dialogue!"
    )
    await state.update_data(role="You are a friendly conversational partner helping the user practice English.", messages=[])
    await state.set_state(ChatStates.conversation_mode)


@dp.message(F.voice, ChatStates.conversation_mode)
async def handle_voice_message(message: Message, state: FSMContext):
    user_last_activity[message.from_user.id] = datetime.now()

    try:
        # Получаем файл голосового сообщения
        voice_file = await bot.get_file(message.voice.file_id)
        file_path = voice_file.file_path

        # Скачиваем и обрабатываем файл во временной директории
        with tempfile.TemporaryDirectory() as tmpdir:
            ogg_path = os.path.join(tmpdir, f"{message.voice.file_id}.ogg")
            wav_path = os.path.join(tmpdir, f"{message.voice.file_id}.wav")
            await bot.download_file(file_path, destination=ogg_path)

            # Конвертируем .ogg в .wav
            audio = AudioSegment.from_file(ogg_path)
            audio.export(wav_path, format="wav")

            # Распознаём речь с помощью Whisper
            transcription = model.transcribe(wav_path, language="en")
            user_text = transcription["text"].strip()
            logger.info(f"Recognized text: {user_text}")
            await message.answer(f"You said: {user_text}")

            # Работа с ChatGPT
            user_data = await state.get_data()
            role = user_data.get('role', "You are a friendly and knowledgeable English tutor. "
                                         "You help the user practice English by providing guidance on grammar, pronunciation, vocabulary, and general conversation. "
                                         "Always be encouraging, patient, and provide detailed explanations if the user asks for help. "
                                         "Make sure to correct mistakes politely and suggest improvements where necessary.")
            messages = user_data.get('messages', [])

            # Добавляем текст пользователя в контекст
            messages.append({"role": "user", "content": user_text})
            if len(messages) > 20:
                messages = messages[-20:]

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": role},
                        *messages,
                    ],
                    max_tokens=150,
                    temperature=0.7,
                )
                bot_reply = response['choices'][0]['message']['content']
            except Exception as e:
                logger.error(f"Error when accessing the API: {e}")
                await message.answer("An error occurred while processing the request. Please try again.")
                return

            # Добавляем ответ бота в историю
            messages.append({"role": "assistant", "content": bot_reply})
            await state.update_data(messages=messages)

            # Генерация голосового ответа
            try:
                # Используем gTTS для преобразования текста в речь
                tts = gTTS(text=bot_reply, lang="en")
                response_voice_path = os.path.join(tmpdir,
                                                   f"{message.from_user.id}_{datetime.now().timestamp()}_response.ogg")
                tts.save(response_voice_path)

                # Отправляем голосовой ответ
                voice = FSInputFile(response_voice_path)
                await message.answer_voice(voice)
            except Exception as e:
                logger.error(f"Error generating voice response: {e}")
                await message.answer("An error occurred while generating the voice response. Please try again.")

    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        await message.answer("An error occurred while processing your voice message. Please try again.")


@dp.message(F.text == '🖼 Image generation')
async def image_generation_prompt(message: Message, state: FSMContext):
    update_user_activity(message.from_user.id, message.from_user.full_name, feature='image_generation')
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    await message.answer("Please describe the image you want to generate:")
    await state.set_state(ChatStates.generating_image)


@dp.message(F.text == '🔙 Back')
async def back_to_main_menu(message: Message, state: FSMContext):
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    await cmd_start(message, state)


@dp.message(Command('clear'))
async def clear_context(message: Message, state: FSMContext):
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    await state.update_data(messages=[])
    await message.answer("The context has been cleared.")


@dp.message(ChatStates.waiting_for_input)
async def chat_with_gpt(message: Message, state: FSMContext):
    global analytics_data
    user_last_activity[message.from_user.id] = datetime.now()

    user_id = message.from_user.id
    analytics_data['total_messages'] += 1
    save_analytics(analytics_data)

    user_data = await state.get_data()
    role = user_data.get('role', 'You are a helpful assistant.')
    messages = user_data.get('messages', [])

    # Проверяем, что сообщение содержит текст
    if message.text and message.text.strip():  # Проверяем, что message.text не None и не пустой
        messages.append({"role": "user", "content": message.text.strip()})

    # Ограничиваем историю последних 20 сообщений
    messages = [msg for msg in messages if msg.get("content")]  # Удаляем пустые сообщения
    if len(messages) > 20:
        messages = messages[-20:]

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": role},
                *messages
            ],
            max_tokens=150,
            temperature=0.7,
        ))
        reply = response['choices'][0]['message']['content'].strip()

        # Добавляем ответ ассистента в историю
        messages.append({"role": "assistant", "content": reply})

        # Сохраняем обновленную историю
        await state.update_data(messages=messages)

        await message.answer(reply)
    except Exception as e:
        await message.answer("An error occurred when accessing the AI.")
        logger.exception("Error when accessing the API")



def get_image_generation_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔄 Generate Image Again")],
            [KeyboardButton(text="🔙 Back")]
        ],
        resize_keyboard=True,  # Устанавливаем размер клавиатуры
        one_time_keyboard=True  # Клавиатура исчезнет после нажатия
    )
    return keyboard


@dp.message(F.text == "🔄 Generate Image Again")
async def generate_image(message: Message, state: FSMContext):
    await image_generation_prompt(message, state)


@dp.message(ChatStates.generating_image)
async def generate_image(message: Message, state: FSMContext):
    global analytics_data
    # Обновляем время последней активности пользователя
    user_last_activity[message.from_user.id] = datetime.now()

    user_id = message.from_user.id
    # Увеличиваем общее количество сообщений
    analytics_data['total_messages'] += 1
    save_analytics(analytics_data)

    prompt = message.text.strip()
    await message.answer("Generating image...")
    try:
        # Используем синхронный метод в отдельном потоке
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: openai.Image.create(
            prompt=prompt,
            model="dall-e-3",
            n=1,
            size="1024x1024",
            quality="standard",
        ))
        image_url = response['data'][0]['url']
    except Exception as e:
        await message.answer("An error occurred while generating the image.")
        logger.exception("An error occurred while generating the image.")
        return

    await message.answer_photo(image_url)
    await state.clear()
    await message.answer(
        "What would you like to do next?",
        reply_markup=get_image_generation_keyboard()
    )
    # Устанавливаем состояние для ожидания выбора пользователя
    await state.set_state(ChatStates.image_generation_options)


# Команда для владельца бота для просмотра аналитики
@dp.message(Command('analytics_1'))
async def show_analytics(message: Message):
    global analytics_data
    OWNER_USER_ID = 5226052650  # Замените на ваш реальный Telegram User ID

    if message.from_user.id != OWNER_USER_ID:
        await message.answer("You do not have permission to execute this command.")
        return

    total_users = len(analytics_data['total_users'])
    total_messages = analytics_data.get('total_messages', 0)

    analytics_message = (
        f"📊 Bot usage statistics:\n"
        f"👥 Total number of users: {total_users}\n"
        f"💬 Total number of messages: {total_messages}"
    )

    await message.answer(analytics_message)


# Файл для детализированной аналитики
DETAILED_ANALYTICS_FILE = 'detailed_analytics.json'

# Функция для загрузки детализированной аналитики
def load_detailed_analytics():
    if os.path.exists(DETAILED_ANALYTICS_FILE):
        with open(DETAILED_ANALYTICS_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
            'users': {},  # Структура: {user_id: {"name": str, "last_active": str}}
            'daily_stats': {},
            'feature_usage': {  # Для подсчета использования функций
                'text_chat': 0,
                'image_generation': 0,
                'pronunciation_check': 0,
                'conversation_mode': 0
            }
        }


def save_detailed_analytics(data):
    for date_key, stats in data['daily_stats'].items():
        stats['users'] = list(stats['users'])  # Преобразуем множество в список
    with open(DETAILED_ANALYTICS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Загружаем аналитику при запуске
detailed_analytics = load_detailed_analytics()

# Обновление данных аналитики при активности пользователя
def update_user_activity(user_id, user_name, feature=None):
    current_time = datetime.now()
    detailed_analytics['users'][user_id] = {
        "name": user_name,
        "last_active": current_time.isoformat()
    }

    # Обновляем ежедневную статистику
    date_key = current_time.strftime('%Y-%m-%d')
    if date_key not in detailed_analytics['daily_stats']:
        detailed_analytics['daily_stats'][date_key] = {'messages': 0, 'users': set()}
    detailed_analytics['daily_stats'][date_key]['messages'] += 1
    for date_key, stats in detailed_analytics['daily_stats'].items():
        stats['users'] = set(stats['users'])
    detailed_analytics['daily_stats'][date_key]['users'].add(user_id)

    # Увеличиваем счетчик использования функции, если она указана
    if feature:
        if feature not in detailed_analytics['feature_usage']:
            detailed_analytics['feature_usage'][feature] = 0
        detailed_analytics['feature_usage'][feature] += 1

    # Сохраняем изменения
    save_detailed_analytics(detailed_analytics)
# Функция для получения аналитики
#
def get_detailed_analytics():
    current_time = datetime.now()
    total_users = len(detailed_analytics['users'])
    user_names = list(set([details["name"] for details in detailed_analytics['users'].values()]))

    # Преобразуем идентификаторы пользователей в строки для единообразия
    users_data = {str(user_id): details for user_id, details in detailed_analytics['users'].items()}

    # Пользователи за последний час
    last_hour_users = [
        user_id for user_id, details in users_data.items()
        if current_time - datetime.fromisoformat(details["last_active"]) <= timedelta(hours=1)
    ]

    # Пользователи за день
    date_key = current_time.strftime('%Y-%m-%d')
    today_stats = detailed_analytics['daily_stats'].get(date_key, {})
    today_users = len(today_stats.get('users', set()))
    today_messages = today_stats.get('messages', 0)

    # Пользователи за неделю
    week_users = {
        user_id for user_id, details in users_data.items()
        if current_time - datetime.fromisoformat(details["last_active"]) <= timedelta(weeks=1)
    }
    week_messages = sum(
        stats.get('messages', 0)
        for date, stats in detailed_analytics['daily_stats'].items()
        if current_time - datetime.strptime(date, '%Y-%m-%d') <= timedelta(weeks=1)
    )

    # Пользователи за месяц
    month_users = {
        user_id for user_id, details in users_data.items()
        if current_time - datetime.fromisoformat(details["last_active"]) <= timedelta(days=30)
    }
    month_messages = sum(
        stats.get('messages', 0)
        for date, stats in detailed_analytics['daily_stats'].items()
        if current_time - datetime.strptime(date, '%Y-%m-%d') <= timedelta(days=30)
    )

    # Среднее количество сообщений на пользователя
    avg_messages_per_user = today_messages / today_users if today_users > 0 else 0

    # Самый активный пользователь за день
    user_message_count = {}
    for user_id in today_stats.get('users', []):
        user_id = str(user_id)  # Приводим к строке
        user_message_count[user_id] = user_message_count.get(user_id, 0) + 1
    most_active_user = max(user_message_count, key=user_message_count.get, default="None")
    most_active_user_messages = user_message_count.get(most_active_user, 0)

    return {
        "total_users": total_users,
        "user_names": user_names,
        "last_hour_users": len(set(last_hour_users)),
        "today_users": today_users,
        "week_users": len(set(week_users)),
        "month_users": len(set(month_users)),
        "feature_usage": detailed_analytics.get('feature_usage', {})
    }


# Обновленная команда /analytics_2
@dp.message(Command('analytics_2'))
async def show_detailed_analytics(message: Message):
    OWNER_USER_ID = 5226052650  # Замените на ваш реальный Telegram User ID

    if message.from_user.id != OWNER_USER_ID:
        await message.answer("You do not have permission to execute this command.")
        return

    analytics = get_detailed_analytics()

    analytics_message = (
        f"📊 **Bot Analytics:**\n"
        f"👥 Total Users: {analytics['total_users']}\n"
        f"📋 Names of Users: {', '.join(analytics['user_names'])}\n"
        f"🕒 Active Users (Last Hour): {analytics['last_hour_users']}\n"
        f"📆 Active Users (Today): {analytics['today_users']}\n"
        f"📅 Active Users (This Week): {analytics['week_users']}\n"
        f"📈 Active Users (This Month): {analytics['month_users']}\n\n"
        f"📌 **Feature Usage:**\n"
        f"📝 Text Chat: {analytics['feature_usage'].get('text_chat', 0)}\n"
        f"🖼 Image Generation: {analytics['feature_usage'].get('image_generation', 0)}\n"
        f"🎙 Pronunciation Check: {analytics['feature_usage'].get('pronunciation_check', 0)}\n"
        f"🗣 Conversation Mode: {analytics['feature_usage'].get('conversation_mode', 0)}\n"
    )

    await message.answer(analytics_message)


# Пример вызова обновления активности при получении сообщения
@dp.message()
async def track_activity(message: Message):
    update_user_activity(message.from_user.id, message.from_user.full_name)


async def main():
    # Создаём задачу для функции clear_user_context() внутри цикла событий
    await set_bot_commands(bot)
    asyncio.create_task(clear_user_context())
    await dp.start_polling(bot, skip_updates=True)

if __name__ == '__main__':
    asyncio.run(main())






