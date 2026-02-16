from flask import Flask, render_template, request, jsonify
from youtubesearchpython import CustomSearch, VideoUploadDateFilter
from dotenv import load_dotenv
import os
import logging
import re
import concurrent.futures
from threading import Lock
from youtubesearchpython.handlers.componenthandler import ComponentHandler

# Monkey-patching the library to fix a bug: TypeError when channel ID is None
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
            'duration': safe_get(video, ['lengthText', 'simpleText']),
            'viewCount': {
                'text': safe_get(video, ['viewCountText', 'simpleText']),
                'short': safe_get(video, ['shortViewCountText', 'simpleText']),
            },
            'thumbnails': safe_get(video, ['thumbnail', 'thumbnails']),
            'channel': {
                'name': safe_get(video, ['ownerText', 'runs', 0, 'text']),
                'id': safe_get(video, ['ownerText', 'runs', 0, 'navigationEndpoint', 'browseEndpoint', 'browseId']),
            },
        }
        video_id = component['id'] or ''
        channel_id = component['channel']['id'] or ''
        component['link'] = 'https://www.youtube.com/watch?v=' + video_id
        component['channel']['link'] = 'https://www.youtube.com/channel/' + channel_id
        component['shelfTitle'] = shelfTitle
        return component

ComponentHandler._getVideoComponent = patched_getVideoComponent

# Load environment variables
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
PROXY_URL = os.getenv('PROXY_URL')

# AI & LLM Keywords
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
        import os as os_lib
        old_http_proxy = os_lib.environ.get('HTTP_PROXY')
        old_https_proxy = os_lib.environ.get('HTTPS_PROXY')
        if 'HTTP_PROXY' in os_lib.environ: del os_lib.environ['HTTP_PROXY']
        if 'HTTPS_PROXY' in os_lib.environ: del os_lib.environ['HTTPS_PROXY']
        
        try:
            preserved = {}
            text = title
            for i, term in enumerate(sorted(PRESERVE_TERMS, key=len, reverse=True)):
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                matches = pattern.findall(text)
                for match in matches:
                    placeholder = f"__TERM{i}_{matches.index(match)}__"
                    preserved[placeholder] = match
                    text = text.replace(match, placeholder, 1)
            
            translated = GoogleTranslator(source='en', target='ru').translate(text)
            for placeholder, original in preserved.items():
                translated = translated.replace(placeholder, original)
            return translated
        finally:
            if old_http_proxy: os_lib.environ['HTTP_PROXY'] = old_http_proxy
            if old_https_proxy: os_lib.environ['HTTPS_PROXY'] = old_https_proxy
    except Exception as e:
        logging.warning(f"Translation error: {e}")
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/keywords')
def get_keywords():
    return jsonify(IT_KEYWORDS)

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    is_it = request.json.get('is_it', False)
    
    try:
        if PROXY_URL:
            os.environ['HTTP_PROXY'] = PROXY_URL
            os.environ['HTTPS_PROXY'] = PROXY_URL
        
        if is_it:
            def search_for_lang(lang):
                lang_videos = []
                lang_seen_ids = set()
                lock = Lock()
                
                def search_keyword(keyword):
                    if lang == 'en':
                        search_query = f"{query} {keyword} free".strip() if query else f"{keyword} free"
                    else:
                        search_query = f"{query} {keyword} бесплатно".strip() if query else f"{keyword} бесплатно"
                    
                    local_res = []
                    try:
                        videos_search = CustomSearch(search_query, VideoUploadDateFilter.thisWeek, limit=20)
                        results = videos_search.result()
                        if results and results.get('result'):
                            for video in results.get('result', []):
                                video_id = video.get('id')
                                if not video_id: continue
                                with lock:
                                    if video_id in lang_seen_ids: continue
                                    lang_seen_ids.add(video_id)
                                
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
                                        local_res.append({
                                            "title": title,
                                            "link": video.get('link') or "#",
                                            "id": video_id,
                                            "publishedTime": pub_time_raw,
                                            "sort_key": parse_published_time(pub_time_raw),
                                            "channel": (video.get('channel') or {}).get('name') or "Неизвестный канал"
                                        })
                    except Exception as e:
                        logging.warning(f"Error ({lang}) '{search_query}': {e}")
                    return local_res

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(search_keyword, kw) for kw in IT_KEYWORDS]
                    for future in concurrent.futures.as_completed(futures):
                        lang_videos.extend(future.result())
                
                # Parallel translation for English
                if lang == 'en' and lang_videos:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                        trans_futures = {executor.submit(translate_title, v['title']): v for v in lang_videos}
                        for future in concurrent.futures.as_completed(trans_futures):
                            v = trans_futures[future]
                            trans = future.result()
                            if trans:
                                v['title'] = f"{trans} ({v['title']})"

                lang_videos.sort(key=lambda x: x.get('sort_key', 999999))
                return lang_videos

            # Run EN and RU searches concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                f_en = executor.submit(search_for_lang, 'en')
                f_ru = executor.submit(search_for_lang, 'ru')
                videos_en = f_en.result()
                videos_ru = f_ru.result()

            logging.info(f"Total results: {len(videos_en)} EN / {len(videos_ru)} RU")
            return jsonify({"en": videos_en, "ru": videos_ru})
        else:
            if not query: return jsonify({"error": "Введите запрос"}), 400
            videos_search = CustomSearch(query, VideoUploadDateFilter.thisWeek, limit=100)
            results = videos_search.result()
            all_videos = []
            if results and results.get('result'):
                for video in results.get('result', []):
                    pub_time_raw = video.get('publishedTime')
                    if not pub_time_raw: continue
                    
                    age_minutes = parse_published_time(pub_time_raw)
                    if age_minutes <= MAX_VIDEO_AGE_MINUTES:
                        all_videos.append({
                            "title": video.get('title') or "Без названия",
                            "link": video.get('link') or "#",
                            "id": video.get('id'),
                            "publishedTime": pub_time_raw,
                            "channel": (video.get('channel') or {}).get('name') or "Неизвестный канал"
                        })
            return jsonify(all_videos)
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
