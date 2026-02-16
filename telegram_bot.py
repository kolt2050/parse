"""
Telegram Bot for YouTube AI Search
Listens for /халява command and returns free AI-related videos
Schedules daily search at 9:00 MSK
"""

import os
import asyncio
import logging
import base64
import httpx
import json
import time
import concurrent.futures
from threading import Lock
from datetime import time as dt_time, datetime
import pytz
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Import search functionality from app
from youtubesearchpython import CustomSearch, VideoUploadDateFilter
from youtubesearchpython.handlers.componenthandler import ComponentHandler

# Load environment variables
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
PROXY_URL = os.getenv('PROXY_URL')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Monkey-patching to fix TypeError in youtubesearchpython
original_getVideoComponent = ComponentHandler._getVideoComponent
def patched_getVideoComponent(self, element: dict, shelfTitle: str = None) -> dict:
    video = element.get('videoRenderer')
    if not video:
        return original_getVideoComponent(self, element, shelfTitle)
    try:
        component = original_getVideoComponent(self, element, shelfTitle)
        return component
    except TypeError:
        def safe_get(source, path):
            val = source
            for key in path:
                if val is None: return None
                if isinstance(key, int):
                    val = val[key] if len(val) > key else None
                else:
                    val = val.get(key)
            return val
        component = {
            'type': 'video',
            'id': safe_get(video, ['videoId']),
            'title': safe_get(video, ['title', 'runs', 0, 'text']),
            'publishedTime': safe_get(video, ['publishedTimeText', 'simpleText']),
            'channel': {
                'name': safe_get(video, ['ownerText', 'runs', 0, 'text']),
                'id': safe_get(video, ['ownerText', 'runs', 0, 'navigationEndpoint', 'browseEndpoint', 'browseId']) or 'unknown',
            },
            'link': 'https://www.youtube.com/watch?v=' + (safe_get(video, ['videoId']) or ''),
        }
        return component
ComponentHandler._getVideoComponent = patched_getVideoComponent

# AI Keywords list
IT_KEYWORDS = [
    'ai', 'machine learning', 'deep learning', 'large language model',
    'gpt', 'chatgpt', 'openai', 'sora', 'dall-e', 'dalle',
    'claude', 'anthropic', 'opus', 'sonnet', 'haiku',
    'gemini', 'deepmind', 'llama ai', 'mistral',
    'grok', 'xai', 'deepseek', 'qwen',
    'stable diffusion', 'midjourney', 'runway', 'kling',
    'hugging face', 'huggingface', 'cohere', 'perplexity',
    'generative', 'copilot', 'suno', 'udio'
]

# Terms to preserve (not translate)
PRESERVE_TERMS = IT_KEYWORDS + [
    'api', 'sdk', 'github', 'python', 'javascript', 'typescript',
    'tutorial', 'course', 'youtube', 'google', 'microsoft', 'meta',
    'nvidia', 'amd', 'intel', 'cuda', 'pytorch', 'tensorflow',
    'vscode', 'vs code', 'notebook', 'jupyter', 'colab',
    'token', 'tokens', 'gpu', 'cpu', 'vram', 'ram',
    'whisper', 'flux', 'lora', 'comfyui', 'automatic1111', 'webui'
]

# Max age for videos (in minutes)
MAX_VIDEO_AGE_MINUTES = 24 * 60  # 24 hours


def translate_title(title):
    """Translate title to Russian, preserving AI-specific terms"""
    if not title: return None
    try:
        from deep_translator import GoogleTranslator
        import re
        import os as os_lib
        
        # Temporarily remove proxy for translation to avoid ProxyError
        old_http_proxy = os_lib.environ.get('HTTP_PROXY')
        old_https_proxy = os_lib.environ.get('HTTPS_PROXY')
        if 'HTTP_PROXY' in os_lib.environ: del os_lib.environ['HTTP_PROXY']
        if 'HTTPS_PROXY' in os_lib.environ: del os_lib.environ['HTTPS_PROXY']
        
        try:
            # Mark terms to preserve with placeholders
            preserved = {}
            text = title
            
            for i, term in enumerate(sorted(PRESERVE_TERMS, key=len, reverse=True)):
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                matches = pattern.findall(text)
                for match in matches:
                    placeholder = f"__TERM{i}_{matches.index(match)}__"
                    preserved[placeholder] = match
                    text = text.replace(match, placeholder, 1)
            
            # Translate
            translated = GoogleTranslator(source='en', target='ru').translate(text)
            
            # Restore preserved terms
            for placeholder, original in preserved.items():
                translated = translated.replace(placeholder, original)
            
            return translated
        finally:
            # Restore proxies
            if old_http_proxy: os_lib.environ['HTTP_PROXY'] = old_http_proxy
            if old_https_proxy: os_lib.environ['HTTPS_PROXY'] = old_https_proxy
            
    except Exception as e:
        logging.warning(f"Translation error: {e}")
        return None


async def get_base64_image(client, url):
    """Download image and return as base64 data URI"""
    if not url:
        return None
    try:
        response = await client.get(url)
        if response.status_code == 200:
            content_type = response.headers.get('content-type', 'image/jpeg')
            img_base64 = base64.b64encode(response.content).decode('utf-8')
            return f"data:{content_type};base64,{img_base64}"
    except Exception as e:
        logging.warning(f"Error fetching image {url}: {e}")
    return None


def parse_published_time(time_str):
    """Convert YouTube published time string to minutes for sorting and filtering"""
    if not time_str:
        return 999999
    
    time_str = str(time_str).lower()
    
    # Handle non-numerical recent cases
    if any(x in time_str for x in ['yesterday', 'вчера']):
        return 1440
    if any(x in time_str for x in ['just now', 'только что']):
        return 1
        
    try:
        import re
        nums = re.findall(r'\d+', time_str)
        if nums:
            n = int(nums[0])
            if any(x in time_str for x in ['minute', 'minut', 'мин']): return n
            if any(x in time_str for x in ['hour', 'chas', 'час']): return n * 60
            if any(x in time_str for x in ['day', 'den', 'дня', 'дне', 'дней']): return n * 1440
            if any(x in time_str for x in ['second', 'sekund', 'сек']): return 1
            if any(x in time_str for x in ['week', 'nedel', 'нед']): return n * 10080
            if any(x in time_str for x in ['month', 'mesyac', 'мес']): return n * 43200
        return 999999
    except:
        return 999999


def search_free_ai_videos(lang='en'):
    if PROXY_URL:
        os.environ['HTTP_PROXY'] = PROXY_URL
        os.environ['HTTPS_PROXY'] = PROXY_URL
    
    import re
    all_videos = []
    seen_ids = set()
    lock = Lock()
    
    def search_keyword(keyword):
        if lang == 'en':
            search_query = f"{keyword} free"
        else:
            search_query = f"{keyword} бесплатно"
            
        local_results = []
        try:
            videos_search = CustomSearch(search_query, VideoUploadDateFilter.thisWeek, limit=20)
            results = videos_search.result()
            
            if results and results.get('result'):
                for video in results.get('result', []):
                    video_id = video.get('id')
                    if not video_id: continue
                    
                    with lock:
                        if video_id in seen_ids: continue
                        seen_ids.add(video_id)
                    
                    pub_time_raw = video.get('publishedTime')
                    if not pub_time_raw: continue
                    
                    age_minutes = parse_published_time(pub_time_raw)
                    is_recent = age_minutes <= MAX_VIDEO_AGE_MINUTES
                    
                    if is_recent:
                        title = video.get('title') or ""
                        title_lower = title.lower()
                        has_keyword = keyword.lower() in title_lower
                        
                        matches = False
                        if lang == 'en':
                            has_free = 'free' in title_lower
                            matches = has_free and has_keyword
                        else:
                            has_ru = bool(re.search('[а-яА-Я]', title))
                            has_free_ru = 'бесплат' in title_lower
                            matches = has_ru and has_keyword and has_free_ru
                        
                        if matches:
                            thumb_url = f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg"
                            local_results.append({
                                "id": video_id,
                                "title": title or "No title",
                                "link": video.get('link') or "#",
                                "channel": (video.get('channel') or {}).get('name') or "?",
                                "time": pub_time_raw,
                                "sort_key": parse_published_time(pub_time_raw),
                                "thumbnail": thumb_url
                            })
        except Exception as e:
            logging.warning(f"Error searching ({lang}) '{search_query}': {e}")
        return local_results

    # Use ThreadPoolExecutor for parallel keyword searching
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(search_keyword, kw) for kw in IT_KEYWORDS]
        for future in concurrent.futures.as_completed(futures):
            all_videos.extend(future.result())
            
    return all_videos


# Persistent stats for timing
STATS_FILE = os.path.join(os.path.dirname(__file__), 'bot_stats.json')

def load_stats():
    """Load last execution time from file"""
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Error loading stats: {e}")
    return {"last_total_time": 15}  # Optimized default 15s

def save_stats(total_time):
    """Save execution time to file"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump({"last_total_time": int(total_time)}, f)
    except Exception as e:
        logging.warning(f"Error saving stats: {e}")


async def run_search_and_report(context: ContextTypes.DEFAULT_TYPE, chat_id: str):
    """Core logic to search, generate HTML and send to a chat"""
    start_all = time.time()
    loop = asyncio.get_event_loop()
    
    # 1. Parallel search
    tasks = [
        loop.run_in_executor(None, search_free_ai_videos, 'en'),
        loop.run_in_executor(None, search_free_ai_videos, 'ru')
    ]
    videos_en, videos_ru = await asyncio.gather(*tasks)
    
    # Sort both lists
    videos_en.sort(key=lambda x: x.get('sort_key', 999999))
    videos_ru.sort(key=lambda x: x.get('sort_key', 999999))
    
    if not videos_en and not videos_ru:
        await context.bot.send_message(chat_id=chat_id, text="Халява за последние 24 часа не найдена.")
        return
    
    # 2. Parallel thumbnails and translations
    proxy_config = {"all://": PROXY_URL} if PROXY_URL else None
    async with httpx.AsyncClient(timeout=15.0, proxies=proxy_config) as client:
        en_thumb_tasks = [get_base64_image(client, v.get('thumbnail')) for v in videos_en]
        ru_thumb_tasks = [get_base64_image(client, v.get('thumbnail')) for v in videos_ru]
        en_trans_tasks = [loop.run_in_executor(None, translate_title, v['title']) for v in videos_en]
        
        results = await asyncio.gather(
            asyncio.gather(*en_thumb_tasks),
            asyncio.gather(*ru_thumb_tasks),
            asyncio.gather(*en_trans_tasks)
        )
        en_thumbs, ru_thumbs, en_translations = results
    
    for video, trans in zip(videos_en, en_translations):
        video['translated_title'] = trans

    # 3. Generate HTML
    html_content = f'''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="referrer" content="no-referrer">
    <title>AI Халява - Daily Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            padding: 10px;
            margin: 0;
            min-height: 100vh;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00d4ff; text-align: center; margin-bottom: 5px; }}
        .info {{ text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }}
        
        .columns {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .column {{
            flex: 1;
            min-width: 300px;
        }}
        .column h2 {{
            color: #00d4ff;
            border-bottom: 2px solid rgba(0,212,255,0.3);
            padding-bottom: 5px;
            margin-bottom: 15px;
            font-size: 20px;
            text-align: center;
        }}
        
        .video {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 12px;
            transition: transform 0.2s;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }}
        .video:hover {{ transform: scale(1.02); background: rgba(255,255,255,0.1); }}
        .thumb {{
            width: 120px;
            height: 68px;
            object-fit: cover;
            border-radius: 4px;
            background: #000;
        }}
        .video-content {{ flex: 1; }}
        .video a {{
            color: #ccc;
            text-decoration: none;
            font-size: 13px;
            font-weight: 400;
            display: block;
            margin-bottom: 3px;
            opacity: 0.8;
            line-height: 1.3;
        }}
        .video a:hover {{ text-decoration: underline; color: #00d4ff; }}
        .translation {{ 
            color: #00d4ff; 
            font-size: 16px; 
            margin-top: 2px; 
            font-weight: 700;
            line-height: 1.2;
        }}
        .v-date {{ color: #888; font-size: 10px; margin-top: 4px; }}
        
        .warning {{ color: #ff4444; text-align: center; font-weight: bold; margin: 15px 0; padding: 10px; background: rgba(255,68,68,0.1); border-radius: 8px; }}
        @media (max-width: 768px) {{
            .columns {{ flex-direction: column; }}
            .column {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Халява (Daily)</h1>
        <p class="info">Сгенерировано: {datetime.now().strftime("%d.%m.%Y %H:%M")}<br> Найдено: {len(videos_en)} EN / {len(videos_ru)} RU за 24ч</p>
        <p class="warning">⚠️ Если вас какой-то блогер призывает ввести логин и пароль на неизвестном сайте — откажитесь!</p>
        <div class="columns">
            <div class="column">
                <h2>English (Translated)</h2>'''
    
    # English videos
    for video, base64_thumb in zip(videos_en, en_thumbs):
        title = video['title'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        trans = video.get('translated_title')
        trans_html = f'<div class="translation">{trans}</div>' if trans else ''
        thumb_img = f'<img src="{base64_thumb}" class="thumb">' if base64_thumb else '<div class="thumb"></div>'
        html_content += f'''
                <div class="video">
                    {thumb_img}
                    <div class="video-content">
                        <a href="{video['link']}">{title}</a>
                        {trans_html}
                        <div class="v-date">{video['time']} | {video['channel']}</div>
                    </div>
                </div>'''

    html_content += '''
            </div>
            <div class="column">
                <h2>Russian</h2>'''
    
    # Russian videos
    for video, base64_thumb in zip(videos_ru, ru_thumbs):
        title = video['title'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        thumb_img = f'<img src="{base64_thumb}" class="thumb">' if base64_thumb else '<div class="thumb"></div>'
        html_content += f'''
                <div class="video">
                    {thumb_img}
                    <div class="video-content">
                        <div class="translation" style="color: #00ff88;">
                            <a href="{video['link']}" style="color:inherit;">{title}</a>
                        </div>
                        <div class="v-date">{video['time']} | {video['channel']}</div>
                    </div>
                </div>'''

    html_content += '''
        </div></div></div></body></html>'''
    
    static_path = os.path.join(os.path.dirname(__file__), 'static', 'last.html')
    with open(static_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # 4. Send report
    warning_text = "⚠️ Если вас какой-то блогер призывает ввести логин и пароль на неизвестном сайте — откажитесь!"
    await context.bot.send_document(
        chat_id=chat_id,
        document=open(static_path, 'rb'),
        filename=f'ai_halyava_{datetime.now().strftime("%Y-%m-%d")}.html',
        caption=f"📰 Ежедневный отчет по AI Халяве\nНайдено сегодня: {len(videos_en)} EN / {len(videos_ru)} RU\n\n{warning_text}"
    )
    
    end_all = time.time()
    save_stats(end_all - start_all)


async def halyava_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /халява command"""
    stats = load_stats()
    last_total = stats.get("last_total_time", 15)
    
    minutes = last_total // 60
    seconds = last_total % 60
    time_str = f"~{minutes} мин {seconds} сек" if minutes > 0 else f"~{seconds} сек"
    
    await update.message.reply_text(f"Ищу свежую халяву (за 24ч)... Примерное время: {time_str}")
    await run_search_and_report(context, update.effective_chat.id)


async def scheduled_halyava(context: ContextTypes.DEFAULT_TYPE):
    """Daily scheduled task"""
    logging.info("Starting scheduled daily search...")
    if TELEGRAM_CHAT_ID:
        await run_search_and_report(context, TELEGRAM_CHAT_ID)
    else:
        logging.warning("TELEGRAM_CHAT_ID not set, skipping scheduled report")


def main():
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set in .env")
        return
    
    # Create application with JobQueue (requires apscheduler)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("halyava", halyava_command))
    application.add_handler(CommandHandler("free", halyava_command))
    application.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text(
        "Привет! Я ищу бесплатные AI-видео.\n\nКоманды:\n/halyava - поиск вручную\nАвто-отчет: ежедневно в 9:00 МСК"
    )))
    
    async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message and update.message.text:
            if 'halyava' in update.message.text.lower():
                await halyava_command(update, context)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    
    # Schedule job at 9:00 MSK (Europe/Moscow)
    moscow_tz = pytz.timezone('Europe/Moscow')
    nine_am = dt_time(hour=9, minute=0, second=0, tzinfo=moscow_tz)
    
    job_queue = application.job_queue
    job_queue.run_daily(scheduled_halyava, time=nine_am)
    
    print("Telegram bot with Daily Job (09:00 MSK) started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
