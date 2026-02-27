<<<<<<< HEAD
import streamlit as st
from rapidocr_onnxruntime import RapidOCR
import openai
import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import html
import base64
from io import BytesIO

import json

def normalize_base_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return "https://api.openai.com/v1"
    if not value.endswith("/v1"):
        value = value.rstrip("/") + "/v1"
    return value


def fetch_models_from_endpoint(base_url: str, api_key: str):
    api_key = (api_key or "").strip()
    if not api_key:
        return [], "请先填写 API Key。"
    try:
        client = openai.OpenAI(api_key=api_key, base_url=normalize_base_url(base_url))
        response = client.models.list()
        model_ids = sorted(
            {item.id for item in response.data if getattr(item, "id", None)},
            key=str.lower
        )
        return model_ids, None
    except Exception as exc:
        return [], f"接口模型列表拉取失败: {exc}"


# Read defaults from config (fallback-safe)
def load_config_defaults():
    default_base_url = ""
    default_api_key = ""
    default_model_name = "gpt-4o-mini"

    config_path = os.path.join(os.path.dirname(__file__), "config")
    if not os.path.exists(config_path):
        return default_base_url, default_api_key, default_model_name

    with open(config_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) >= 3:
        default_model_name = lines[2]

    return default_base_url, default_api_key, default_model_name


default_base_url, default_api_key, default_model_name = load_config_defaults()

# Initialize RapidOCR reader
@st.cache_resource
def load_ocr_reader():
    return RapidOCR()

reader = load_ocr_reader()

def tokenize_text(text):
    # Normalize internal spaces in brackets
    text = re.sub(r'\[([^\[\]{}]+?)\s*\{([^{}]+)\}\]', r'[\1]{\2}', text)
    text = re.sub(r'\(([^\(\){}]+?)\s*\{([^{}]+)\}\)', r'(\1){\2}', text)

    pattern = r'\[([^\[\]]+)\](?:\s*\{([^{}]+)\})?|\(([^\(\)]+)\)(?:\s*\{([^{}]+)\})?'
    
    tokens = []
    last_end = 0
    for m in re.finditer(pattern, text):
        start, end = m.span()
        if start > last_end:
            normal_text = text[last_end:start]
            normal_text = re.sub(r'\{[^{}]{1,30}\}', '', normal_text)
            if normal_text:
                tokens.append({'type': 'normal', 'text': normal_text})
        
        if m.group(1) is not None:
            tokens.append({'type': 'add', 'text': m.group(1), 'reason': m.group(2)})
        elif m.group(3) is not None:
            tokens.append({'type': 'del', 'text': m.group(3), 'reason': m.group(4)})
        
        last_end = end
        
    if last_end < len(text):
        normal_text = text[last_end:]
        normal_text = re.sub(r'\{[^{}]{1,30}\}', '', normal_text)
        if normal_text:
            tokens.append({'type': 'normal', 'text': normal_text})

    return tokens

def highlight_brackets(text):
    tokens = tokenize_text(text)
    html_parts = []
    for token in tokens:
        if token['type'] == 'normal':
            html_parts.append(html.escape(token['text']).replace('\n', '<br>'))
        else:
            is_add = token['type'] == 'add'
            bg_color = "#d4f7d4" if is_add else "#f8d7da"
            fg_color = "#155724" if is_add else "#721c24"
            strike = not is_add
            content = html.escape(token['text'])
            reason = html.escape(token['reason']) if token['reason'] else None
            
            decoration = "text-decoration: line-through;" if strike else ""
            base_span = (
                f'<span style="background:{bg_color}; padding: 2px 5px; border-radius: 4px; '
                f'font-weight: 600; color: {fg_color}; {decoration}">{content}</span>'
            )
            if reason:
                span = (
                    '<span style="display: inline-flex; flex-direction: column; align-items: center; '
                    f'vertical-align: baseline; line-height: 1.05;">{base_span}'
                    f'<span style="font-size: 11px; color: {fg_color}; white-space: nowrap; margin-top: 0; '
                    f'font-weight: normal; line-height: 1; text-decoration: none;">{reason}</span></span>'
                )
            else:
                span = base_span
            html_parts.append(span)
            
    return "".join(html_parts)


def build_jpg_report(corrected_text, rate_text, perfect_text, theme_text, original_image_data_url=None):
    def pick_font(size):
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def decode_data_url_image(data_url):
        if not data_url or "," not in data_url:
            return None
        try:
            encoded = data_url.split(",", 1)[1]
            raw = base64.b64decode(encoded)
            return Image.open(BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def wrap_lines(draw, text, font, max_width):
        lines = []
        for paragraph in str(text or "").split("\n"):
            if paragraph == "":
                lines.append("")
                continue
            current = ""
            for ch in paragraph:
                candidate = current + ch
                if draw.textlength(candidate, font=font) <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = ch
            if current:
                lines.append(current)
        return lines

    def build_rate_text(rate_raw):
        try:
            data = json.loads(rate_raw)
        except Exception:
            return str(rate_raw or "")

        lines = []
        dim_map = [
            ("fluency", "语言通顺度"),
            ("coherence", "上下文连贯度"),
            ("accuracy", "词汇与语法"),
        ]
        for key, name in dim_map:
            dim = data.get(key, {})
            lines.append(f"【{name}】{dim.get('score', 0)}/100")
            comments = dim.get("comments", [])
            if comments:
                lines.append("评价：")
                lines.extend([f"- {c}" for c in comments])
            suggestions = dim.get("suggestions", [])
            if suggestions:
                lines.append("建议：")
                lines.extend([f"- {s}" for s in suggestions])
            lines.append("")
        return "\n".join(lines).strip()

    def build_theme_text(theme_raw):
        try:
            data = json.loads(theme_raw)
        except Exception:
            return str(theme_raw or "")

        lines = []
        expressions = data.get("expressions", [])
        for expr in expressions:
            lines.append(f"【{expr.get('category', '未分类')}】")
            sentences = expr.get("sentences", [])
            phrases = expr.get("phrases", [])
            words = expr.get("words", [])
            if sentences:
                lines.append("完整句子/句型：")
                lines.extend([f"- {s}" for s in sentences])
            if phrases:
                lines.append("实用短语：")
                lines.extend([f"- {p}" for p in phrases])
            if words:
                lines.append("高级词汇：")
                lines.extend([f"- {w}" for w in words])
            lines.append("")
        return "\n".join(lines).strip()

    def draw_rich_text(draw, tokens, font, max_width, start_x, start_y):
        x = start_x
        y = start_y
        line_height = font.size * 2.2
        reason_font = pick_font(int(font.size * 0.6))
        
        for token in tokens:
            if token['type'] == 'normal':
                paragraphs = token['text'].split('\n')
                for i, para in enumerate(paragraphs):
                    if i > 0:
                        x = start_x
                        y += line_height
                    
                    parts = re.findall(r'[a-zA-Z0-9_.,!?;:\'"\-]+|\s+|.', para)
                    for part in parts:
                        if not part: continue
                        part_width = draw.textlength(part, font=font)
                        if x + part_width > start_x + max_width and x > start_x:
                            if part_width > max_width:
                                for ch in part:
                                    ch_width = draw.textlength(ch, font=font)
                                    if x + ch_width > start_x + max_width:
                                        x = start_x
                                        y += line_height
                                    draw.text((x, y), ch, font=font, fill="#0f172a")
                                    x += ch_width
                            else:
                                x = start_x
                                y += line_height
                                if part.isspace():
                                    continue
                                draw.text((x, y), part, font=font, fill="#0f172a")
                                x += part_width
                        else:
                            if x == start_x and part.isspace():
                                continue
                            draw.text((x, y), part, font=font, fill="#0f172a")
                            x += part_width
            else:
                is_add = token['type'] == 'add'
                bg_color = "#e6ffe6" if is_add else "#ffe6e6"
                fg_color = "#006600" if is_add else "#cc0000"
                
                text_content = token['text']
                reason_content = token['reason']
                
                text_width = draw.textlength(text_content, font=font)
                reason_width = draw.textlength(reason_content, font=reason_font) if reason_content else 0
                
                box_width = text_width + 12
                if reason_content:
                    box_width = max(box_width, reason_width + 12)
                
                if x + box_width > start_x + max_width and x > start_x:
                    x = start_x
                    y += line_height
                
                box_height = font.size + 12
                box_y = y - 6
                draw.rounded_rectangle([x, box_y, x + box_width, box_y + box_height], radius=6, fill=bg_color)
                
                text_x = x + (box_width - text_width) / 2
                draw.text((text_x, y), text_content, font=font, fill=fg_color)
                
                if not is_add:
                    strike_y = y + font.size / 2 + 2
                    draw.line([text_x, strike_y, text_x + text_width, strike_y], fill=fg_color, width=2)
                
                if reason_content:
                    reason_x = x + (box_width - reason_width) / 2
                    reason_y = box_y + box_height + 2
                    draw.text((reason_x, reason_y), reason_content, font=reason_font, fill=fg_color)
                
                x += box_width + 6
                
        return y + line_height

    width = 1800
    margin = 60
    content_width = width - margin * 2
    canvas = Image.new("RGB", (width, 36000), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = pick_font(48)
    heading_font = pick_font(36)
    body_font = pick_font(24)
    meta_font = pick_font(20)

    y = margin
    draw.text((margin, y), "English Essay Assessment Report", font=title_font, fill="#0f172a")
    y += 70
    draw.text((margin, y), "Generated by EEAR", font=meta_font, fill="#64748b")
    y += 46

    src_image = decode_data_url_image(original_image_data_url)
    if src_image is not None:
        draw.text((margin, y), "原始作文图片", font=heading_font, fill="#1d4ed8")
        y += 52
        preview = src_image.copy()
        preview.thumbnail((content_width, 900))
        canvas.paste(preview, (margin, y))
        y += preview.height + 28

    sections = [
        ("批改结果", corrected_text),
        ("详细评价", build_rate_text(rate_text)),
        ("优秀范文", perfect_text),
        ("积累表达", build_theme_text(theme_text)),
    ]

    for title, content in sections:
        draw.text((margin, y), title, font=heading_font, fill="#1d4ed8")
        y += 52
        
        if title == "批改结果":
            tokens = tokenize_text(content)
            y = draw_rich_text(draw, tokens, body_font, content_width, margin, y)
        else:
            lines = wrap_lines(draw, content, body_font, content_width)
            for line in lines:
                draw.text((margin, y), line, font=body_font, fill="#0f172a")
                y += 38
        y += 16

    final = canvas.crop((0, 0, width, min(y + margin, canvas.height)))
    out = BytesIO()
    final.save(out, format="JPEG", quality=95, optimize=True)
    return out.getvalue()

st.set_page_config(page_title="English Essay Assessment Review", page_icon=None, layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .reading-text {
        font-family: "Cambria", "Georgia", "Times New Roman", serif;
        font-size: 20px;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">English Essay Assessment Review (EEAR)</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; margin-bottom: 2rem;'>智能英语作文批改系统 - 拍照上传，即刻获取专业点评与范文</p>", unsafe_allow_html=True)

common_model_options = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.5-preview",
    "o3-mini",
    "o3",
    "o3-pro",
    "o4-mini",
    "o4-mini-high",
    "deepseek-chat",
    "deepseek-reasoner",
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
    "qwen-max-2025-01-25",
    "qwen3.5-flash",
    "qwen3.5-plus",
    "claude-opus-4-6",
    "claude-opus-4-1",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-haiku-4-5",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]
common_model_options = sorted(set(common_model_options), key=str.lower)

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = default_base_url
if "api_key" not in st.session_state:
    st.session_state.api_key = default_api_key
if "models_from_api" not in st.session_state:
    st.session_state.models_from_api = []
if "models_fetch_error" not in st.session_state:
    st.session_state.models_fetch_error = None
if "models_cache_key" not in st.session_state:
    st.session_state.models_cache_key = ""

model_selector_options = sorted(
    set(common_model_options + st.session_state.models_from_api),
    key=str.lower
) + ["自定义"]

if "model_selector" not in st.session_state:
    st.session_state.model_selector = default_model_name if default_model_name in model_selector_options else "自定义"
if "custom_model" not in st.session_state:
    st.session_state.custom_model = default_model_name
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "图片识别"
if "essay_text_manual" not in st.session_state:
    st.session_state.essay_text_manual = ""
if "review_results" not in st.session_state:
    st.session_state.review_results = None
if "ui_expand_config" not in st.session_state:
    st.session_state.ui_expand_config = True
if "ui_expand_input" not in st.session_state:
    st.session_state.ui_expand_input = True
if "start_requested" not in st.session_state:
    st.session_state.start_requested = False
if "source_image_data_url" not in st.session_state:
    st.session_state.source_image_data_url = None

right_col, left_col = st.columns(2)

with left_col:
    with st.expander("接口设置", expanded=st.session_state.ui_expand_config):
        st.markdown('<div class="panel-title">请配置模型接口参数</div>', unsafe_allow_html=True)
        st.text_input(
            "Base URL",
            key="api_base_url",
            placeholder="例如: https://api.openai.com/v1"
        )
        st.text_input(
            "API Key",
            key="api_key",
            type="password",
            placeholder="请输入 API Key"
        )
        current_cache_key = f"{normalize_base_url(st.session_state.api_base_url)}::{(st.session_state.api_key or '').strip()}"
        if current_cache_key != st.session_state.models_cache_key:
            if (st.session_state.api_key or "").strip():
                models, error = fetch_models_from_endpoint(st.session_state.api_base_url, st.session_state.api_key)
                if error:
                    st.session_state.models_fetch_error = error
                    st.session_state.models_from_api = []
                else:
                    st.session_state.models_from_api = models
                    st.session_state.models_fetch_error = None
            else:
                st.session_state.models_from_api = []
                st.session_state.models_fetch_error = None
            st.session_state.models_cache_key = current_cache_key
            st.rerun()

        st.selectbox(
            "Model",
            options=model_selector_options,
            key="model_selector"
        )

        if st.session_state.models_from_api:
            st.caption(f"已自动从接口加载 {len(st.session_state.models_from_api)} 个模型")
        if st.session_state.models_fetch_error:
            st.caption(st.session_state.models_fetch_error)

        if st.session_state.model_selector == "自定义":
            st.text_input(
                "自定义 Model",
                key="custom_model",
                placeholder="请输入模型名称"
            )

uploaded_file = None
image = None
with right_col:
    with st.expander("上传作文", expanded=st.session_state.ui_expand_input):
        input_mode = st.radio(
            "选择输入来源",
            ["图片识别", "直接输入文字"],
            key="input_mode",
            horizontal=True
        )

        if input_mode == "图片识别":
            uploaded_file = st.file_uploader(
                "请上传英语作文图片",
                type=["png", "jpg", "jpeg", "bmp"],
                key="essay_image_file",
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            st.text_area(
                "请输入英文作文正文",
                height=280,
                placeholder="在这里粘贴或输入英文作文正文",
                key="essay_text_manual",
                label_visibility="collapsed"
            )

start_button = st.button(
    "开始智能批改",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.start_requested
)

if start_button:
    st.session_state.ui_expand_config = False
    st.session_state.ui_expand_input = False
    st.session_state.start_requested = True
    st.rerun()

if st.session_state.start_requested:
    runtime_base_url = normalize_base_url(st.session_state.api_base_url)
    runtime_api_key = (st.session_state.api_key or "").strip()
    runtime_model_name = (st.session_state.custom_model if st.session_state.model_selector == "自定义" else st.session_state.model_selector).strip()

    if not runtime_api_key:
        st.error("请先填写 API Key。")
        st.session_state.start_requested = False
        st.rerun()

    if not runtime_model_name:
        st.error("请先填写 Model。")
        st.session_state.start_requested = False
        st.rerun()

    runtime_client = openai.OpenAI(api_key=runtime_api_key, base_url=runtime_base_url)

    with st.spinner("正在执行智能批改"):
        if input_mode == "图片识别":
            if uploaded_file is None or image is None:
                st.error("请先上传图片")
                st.session_state.start_requested = False
                st.rerun()

            uploaded_bytes = uploaded_file.getvalue()
            uploaded_mime = uploaded_file.type if uploaded_file.type else "image/png"
            st.session_state.source_image_data_url = f"data:{uploaded_mime};base64,{base64.b64encode(uploaded_bytes).decode('utf-8')}"

            img_array = np.array(image)
            result, _ = reader(img_array)

            if not result:
                st.error("文字识别失败，请尝试上传更清晰的图片")
                st.session_state.start_requested = False
                st.rerun()

            essay_text = "\n".join([item[1] for item in result])
        else:
            essay_text = (st.session_state.essay_text_manual or "").strip()
            if not essay_text:
                st.error("请输入作文正文后再开始批改")
                st.session_state.start_requested = False
                st.rerun()
            st.session_state.source_image_data_url = None

        if input_mode == "图片识别":
            prompt = (
                "请帮我整理下面的英语作文文本格式，只整理英文正文部分（忽略英文正文外所有部分）。\n"
                + "把应该属于同一段的内容整理到同一段，取消换行。需要自动分段。\n\n"
                + "特别注意：如果是因为OCR识别错误导致的拼写错误（例如把'the'识别成'tne'，把'and'识别成'ancl'等），请直接将其修正为正确的单词。\n\n"
                + "严格要求：不能增添或者删减单词。\n\n"
                + "下面是需要整理的OCR英文文本：\n\n"
                + essay_text
            )
            completion = runtime_client.chat.completions.create(
                model=runtime_model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            revised_text = completion.choices[0].message.content
        else:
            revised_text = essay_text

        prompt = (
            "请帮我把下面的英语作文的语法错误改正，输出改正后的文章（改错误和不流畅之处）,请参照下面的格式要求\n\n"
            + "有更合适的表达也可以进行替换，注意要求地道流畅，不能使用生僻的表达。\n\n"
            + "这是格式要求：原文修改部分用()括起来，修改的部分用[]括起来。你可以选择在()或[]后面紧跟{}写出3-8个字的修改理由。需要删除就标明(删除的内容){删除理由}，需要添加就标明[添加的内容]{添加理由}。精准标注，不要错了一个单词把整个句子框起来。\n"
            + "严格要求：修改理由只能出现在()或[]后面的{}中，不能写在[]或()内部，不能写在[]或()前面，不能单独悬空出现{}。保证{}和对应的()或[]中不能有别的元素，包括符号。\n"
            + "【极其重要】：当发生替换（即同时有删除和添加）时，修改理由{}必须且只能紧跟在()后面，绝对不能放在[]后面！[]后面不要加任何理由！\n"
            + "正确示范：(wrong word){用词不当} [correct word]\n"
            + "错误示范：(wrong word) [correct word]{用词不当}\n"
            + "错误示范：(wrong word){用词不当} [correct word]{用词更合适}\n"
            + "输出规范：只输出批改后的正文，不要输出说明、不要输出格式解释、不要输出代码块。\n"
            + "例如：[The]{冠词缺失} (rabbish){拼写错误} [rubbish] thrown by (the) visitors has piled up and its lush (verdure no longer flourish){主谓一致} [vegetation no longer flourishes] as it (did once){语序调整} [once did].\n\n"
            + "下面是需要批改的英语习作：\n\n"
            + revised_text
        )
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        corrected_text = completion.choices[0].message.content

        rate_prompt = """
你是⼀位资深英语写作批改专家，熟悉 IELTS、CEFR 及 EFL 写作评价体系。特别注意语言准确度。
请按照下列三个维度，详细具体分析，写明原因（结合原文相关内容，提到原文中哪里有问题哪里写得好），并且需要提出改进方向和方案。
请务必严格按照以下JSON格式输出，不要输出任何其他内容：
{
  "fluency": {
    "score": 85,
    "comments": ["文中的句子结构变化适中，使用了一些复杂的句子和短语，如“first and foremost”和“in other words”，展现了作者对语言的掌握。", "语句的流畅性有待提高。例如，“making an adequately choosing”该句结构不正确，应为“making an adequate choice”。此外，部分句子显得较为冗长，可能会影响读者的理解。"],
    "suggestions": ["适当分割冗长的句子，确保每个句子传递清晰的信息。例如，可以将“making an adequately choosing, which presents us with more faith when facing arduous challenges and high-demanding tasks”改为“making an adequate choice boosts our confidence when facing arduous challenges and high-demanding tasks”。"]
  },
  "coherence": {
    "score": 78,
    "comments": ["文章整体上能够进行较为清晰的逻辑展开，提纲挈领地介绍了选择的重要性。", "尽管主要论点较为明确，但论述的衔接和转折有时显得不够流畅。例如，从“none of our outstanding achievements would be obtained if we haven't preserved the preciousness of this virtue”到“choosing is an action word”的过渡有些突兀，缺乏自然的逻辑连接。"],
    "suggestions": ["使用更有效的过渡词或短语来增强段落之间的连贯性。例如，增加“Furthermore”或“Moreover”来连接不同的思想，使其流畅过渡。"]
  },
  "accuracy": {
    "score": 82,
    "comments": ["使用了一些较为高级的词汇，如“prerequisite”和“unleash”，展现了较好的词汇量。", "词汇使用上还有提升空间，存在一些使用不当和语法错误，如“making an adequately choosing”不符合语法，应为“making an adequate choice”；“the possibility to meet triumph”应为“the possibility of achieving success”。"],
    "suggestions": ["清晰检查词汇的搭配和语法使用，确保语法结构的正确性，尽量避免错误。建议多做语法练习，增强对词汇搭配的理解，在写作中多样化词汇的使用。"]
  }
}

下面是这篇英文习作的原文：
"""
        prompt_review = rate_prompt + "\n\n" + revised_text
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt_review}],
            response_format={"type": "json_object"}
        )
        rate_text = completion.choices[0].message.content

        rewrite_prompt = """
你是⼀位资深英语写作批改专家，熟悉 IELTS、CEFR 及 EFL 写作评价体系。特别注意语言准确度。
使用优秀的英语表达重写下面这篇英文习作，要展现优秀的词汇和语法，使用地道的表达方式，尽量使用多样化的句式、短语和词汇（但是不要通篇生僻，可以使用一两个表情达意的优秀词汇，需要保证流畅度）。
不要使用生僻的表达，加粗可供学习的部分。
文风如（一些示例句子）：
I hope this message finds you well. Recently, I’ve been struggling to balance my academic workload with my extracurricular responsibilities. Although I try to plan my day, I often find myself overwhelmed and falling behind.Since you always manage to stay so organized and composed, I was wondering if you could share some practical tips or strategies for managing time more effectively. Any advice would be greatly appreciated, and I’m eager to make meaningful changes.
He stood frozen at the doorway, eyes fixed on the scattered pages lying across the floor.For a moment, his chest tightened, the sting of betrayal sharp and unexpected.A flicker of warmth rose in his chest. Not everything had been destroyed.He stood frozen at the doorway, the torn letter still in his hand. His heart, once heavy with anger, now sank with regret. Every accusing word he had thrown at Emma replayed in his mind, sharper than any silence.His breath caught in his throat as he reached her door.For a moment, he hesitated.

下面是这篇英文习作原文：\n\n"""
        prompt = rewrite_prompt + revised_text
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        perfect_text = completion.choices[0].message.content

        theme_prompt = """
你是一位英语教学专家，擅长从学生作文中提炼和总结学习要点。
根据下面给出的作文内容，提取该主题相关的优秀表达、常用句型、高级短语和词汇，帮助学生积累表达、拓展词汇。
提取的内容应该：
1. 与作文主题和内容契合，可直接应用于类似写作
2. 包含多个不同的表达维度（如开头结尾、论证、总结等）
3. 涵盖从基础到高级的各层次表达
4. 提供实际可用的完整句子、常用短语和核心词汇

请务必严格按照以下JSON格式输出，不要输出任何其他内容：
{
  "expressions": [
    {
      "category": "具体的内容概括（如引入什么，论证什么，结尾什么等自行根据主题精炼内容要点,有多少个总结多少个点）",
      "sentences": ["完整的例句1", "完整的例句2", "完整的例句3",...],
      "phrases": ["常用短语1", "常用短语2", "常用短语3",...],
      "words": ["高级词汇1", "高级词汇2", "高级词汇3",...]
    }
  ]
}

下面是这篇英文习作原文：\n\n"""
        prompt = theme_prompt + revised_text
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        theme_text = completion.choices[0].message.content

        st.session_state.review_results = {
            "corrected_text": corrected_text,
            "rate_text": rate_text,
            "perfect_text": perfect_text,
            "theme_text": theme_text,
            "source_image_data_url": st.session_state.source_image_data_url,
        }
        st.session_state.start_requested = False
        st.rerun()

if st.session_state.review_results is not None:
    st.markdown("---")
    st.markdown("### 批改结果")
    
    corrected_text = st.session_state.review_results["corrected_text"]
    rate_text = st.session_state.review_results["rate_text"]
    perfect_text = st.session_state.review_results["perfect_text"]
    theme_text = st.session_state.review_results["theme_text"]
    source_image_data_url = st.session_state.review_results.get("source_image_data_url")

    tab1, tab2, tab3, tab4 = st.tabs(["批改结果", "详细评价", "优秀范文", "积累表达"])

    with tab1:
        st.markdown(f'<div class="reading-text">{highlight_brackets(corrected_text)}</div>', unsafe_allow_html=True)

    with tab2:
        try:
            rate_data = json.loads(rate_text)
            overall_score = int((rate_data.get("fluency", {}).get("score", 0) + rate_data.get("coherence", {}).get("score", 0) + rate_data.get("accuracy", {}).get("score", 0)) / 3)
            st.markdown(f"<h3 style='text-align: center; color: #1E3A8A;'>综合得分: {overall_score}/100</h3>", unsafe_allow_html=True)
            st.progress(overall_score / 100)
            st.markdown("<br>", unsafe_allow_html=True)

            col_f, col_c, col_a = st.columns(3)
            with col_f:
                st.metric(label="语言通顺度 (Fluency)", value=f"{rate_data.get('fluency', {}).get('score', 0)}/100")
                st.markdown("**评价:**")
                for comment in rate_data.get("fluency", {}).get("comments", []):
                    st.markdown(f"- {comment}")
                st.markdown("**建议:**")
                for suggestion in rate_data.get("fluency", {}).get("suggestions", []):
                    st.markdown(f"- {suggestion}")

            with col_c:
                st.metric(label="上下文连贯度 (Coherence)", value=f"{rate_data.get('coherence', {}).get('score', 0)}/100")
                st.markdown("**评价:**")
                for comment in rate_data.get("coherence", {}).get("comments", []):
                    st.markdown(f"- {comment}")
                st.markdown("**建议:**")
                for suggestion in rate_data.get("coherence", {}).get("suggestions", []):
                    st.markdown(f"- {suggestion}")

            with col_a:
                st.metric(label="词汇与语法 (Accuracy)", value=f"{rate_data.get('accuracy', {}).get('score', 0)}/100")
                st.markdown("**评价:**")
                for comment in rate_data.get("accuracy", {}).get("comments", []):
                    st.markdown(f"- {comment}")
                st.markdown("**建议:**")
                for suggestion in rate_data.get("accuracy", {}).get("suggestions", []):
                    st.markdown(f"- {suggestion}")
        except json.JSONDecodeError:
            st.error("评分数据解析失败，请重试。")

    with tab3:
        st.markdown(f'<div class="reading-text">{html.escape(perfect_text).replace("\n", "<br>")}</div>', unsafe_allow_html=True)

    with tab4:
        try:
            theme_data = json.loads(theme_text)
            expressions = theme_data.get("expressions", [])
            if not expressions:
                st.info("暂无可展示的积累表达。")
            for expr in expressions:
                category = expr.get("category", "未分类")
                with st.expander(category, expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**完整句子/句型**")
                        for sentence in expr.get("sentences", []):
                            st.markdown(f"- {sentence}")
                    with c2:
                        st.markdown("**实用短语**")
                        for phrase in expr.get("phrases", []):
                            st.markdown(f"- {phrase}")
                    with c3:
                        st.markdown("**高级词汇**")
                        for word in expr.get("words", []):
                            st.markdown(f"- {word}")
        except json.JSONDecodeError:
            st.error("表达数据解析失败，请重试。")

    image_report = build_jpg_report(
        corrected_text=corrected_text,
        rate_text=rate_text,
        perfect_text=perfect_text,
        theme_text=theme_text,
        original_image_data_url=source_image_data_url
    )
    st.markdown("---")
    st.download_button(
        label="下载完整批改报告 (JPG)",
        data=image_report,
        file_name="essay_review_report.jpg",
        mime="image/jpeg",
        use_container_width=True
=======
import streamlit as st
from rapidocr_onnxruntime import RapidOCR
import openai
import os
import re
import numpy as np
from PIL import Image
import html
import base64
from datetime import datetime

import json

def normalize_base_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return "https://api.openai.com/v1"
    if not value.endswith("/v1"):
        value = value.rstrip("/") + "/v1"
    return value


def fetch_models_from_endpoint(base_url: str, api_key: str):
    api_key = (api_key or "").strip()
    if not api_key:
        return [], "请先填写 API Key。"
    try:
        client = openai.OpenAI(api_key=api_key, base_url=normalize_base_url(base_url))
        response = client.models.list()
        model_ids = sorted(
            {item.id for item in response.data if getattr(item, "id", None)},
            key=str.lower
        )
        return model_ids, None
    except Exception as exc:
        return [], f"接口模型列表拉取失败: {exc}"


# Read defaults from config (fallback-safe)
def load_config_defaults():
    default_base_url = ""
    default_api_key = ""
    default_model_name = "gpt-4o-mini"

    config_path = os.path.join(os.path.dirname(__file__), "config")
    if not os.path.exists(config_path):
        return default_base_url, default_api_key, default_model_name

    with open(config_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) >= 3:
        default_model_name = lines[2]

    return default_base_url, default_api_key, default_model_name


default_base_url, default_api_key, default_model_name = load_config_defaults()

# Initialize RapidOCR reader
@st.cache_resource
def load_ocr_reader():
    return RapidOCR()

reader = load_ocr_reader()

def tokenize_text(text):
    # Normalize internal spaces in brackets
    text = re.sub(r'\[([^\[\]{}]+?)\s*\{([^{}]+)\}\]', r'[\1]{\2}', text)
    text = re.sub(r'\(([^\(\){}]+?)\s*\{([^{}]+)\}\)', r'(\1){\2}', text)

    pattern = r'\[([^\[\]]+)\](?:\s*\{([^{}]+)\})?|\(([^\(\)]+)\)(?:\s*\{([^{}]+)\})?'
    
    tokens = []
    last_end = 0
    for m in re.finditer(pattern, text):
        start, end = m.span()
        if start > last_end:
            normal_text = text[last_end:start]
            normal_text = re.sub(r'\{[^{}]{1,30}\}', '', normal_text)
            if normal_text:
                tokens.append({'type': 'normal', 'text': normal_text})
        
        if m.group(1) is not None:
            tokens.append({'type': 'add', 'text': m.group(1), 'reason': m.group(2)})
        elif m.group(3) is not None:
            tokens.append({'type': 'del', 'text': m.group(3), 'reason': m.group(4)})
        
        last_end = end
        
    if last_end < len(text):
        normal_text = text[last_end:]
        normal_text = re.sub(r'\{[^{}]{1,30}\}', '', normal_text)
        if normal_text:
            tokens.append({'type': 'normal', 'text': normal_text})

    return tokens

def highlight_brackets(text):
    tokens = tokenize_text(text)
    html_parts = []
    for token in tokens:
        if token['type'] == 'normal':
            html_parts.append(html.escape(token['text']).replace('\n', '<br>'))
        else:
            is_add = token['type'] == 'add'
            bg_color = "#d4f7d4" if is_add else "#f8d7da"
            fg_color = "#155724" if is_add else "#721c24"
            strike = not is_add
            content = html.escape(token['text'])
            reason = html.escape(token['reason']) if token['reason'] else None
            
            decoration = "text-decoration: line-through;" if strike else ""
            base_span = (
                f'<span style="background:{bg_color}; padding: 2px 5px; border-radius: 4px; '
                f'font-weight: 600; color: {fg_color}; {decoration}">{content}</span>'
            )
            if reason:
                span = (
                    '<span style="display: inline-flex; flex-direction: column; align-items: center; '
                    f'vertical-align: baseline; line-height: 1.05;">{base_span}'
                    f'<span style="font-size: 11px; color: {fg_color}; white-space: nowrap; margin-top: 0; '
                    f'font-weight: normal; line-height: 1; text-decoration: none;">{reason}</span></span>'
                )
            else:
                span = base_span
            html_parts.append(span)
            
    return "".join(html_parts)


# ---------------------------------------------------------------------------
#  Report section renderers (used by HTML export)
# ---------------------------------------------------------------------------

def _render_corrected_section(text: str) -> str:
    """Render corrected essay with the same highlight style as the main page."""
    return highlight_brackets(text)


def _render_rate_section(rate_raw: str) -> str:
    """Render scoring / evaluation section as HTML."""
    try:
        data = json.loads(rate_raw)
    except Exception:
        return f"<p>{html.escape(str(rate_raw or ''))}</p>"

    overall = int(
        (data.get("fluency", {}).get("score", 0)
         + data.get("coherence", {}).get("score", 0)
         + data.get("accuracy", {}).get("score", 0)) / 3
    )

    parts: list[str] = [
        f'<h3 style="text-align:center;color:#1E3A8A;">综合得分: {overall}/100</h3>',
        ('<table style="width:100%;margin:0 auto 24px;"><tr>'
         '<td style="background:#e5e7eb;height:18px;padding:0;">'
         f'<div style="background:#2563EB;height:18px;width:{overall}%;"></div>'
         '</td></tr></table>'),
    ]

    dim_map = [
        ("fluency", "语言通顺度 (Fluency)"),
        ("coherence", "上下文连贯度 (Coherence)"),
        ("accuracy", "词汇与语法 (Accuracy)"),
    ]
    for key, label in dim_map:
        dim = data.get(key, {})
        score = dim.get("score", 0)
        parts.append(
            '<div style="background:#f8fafc;padding:14px 18px;'
            'border:1px solid #e2e8f0;margin-bottom:12px;">'
        )
        parts.append(f'<div style="font-size:14px;color:#64748b;">{html.escape(label)}</div>')
        parts.append(f'<div style="font-size:22px;font-weight:700;color:#0f172a;">{score}/100</div>')
        for heading, items in [("评价", dim.get("comments", [])), ("建议", dim.get("suggestions", []))]:
            if items:
                parts.append(f'<div style="margin-top:10px;"><strong>{heading}:</strong>'
                             '<ul style="margin:4px 0 0 18px;">')
                parts.extend(f"<li>{html.escape(c)}</li>" for c in items)
                parts.append("</ul></div>")
        parts.append("</div>")

    return "\n".join(parts)


def _render_perfect_section(text: str) -> str:
    """Render model essay section."""
    return (
        '<div style="font-size:16px;line-height:1.8;">'
        + html.escape(text or "").replace("\n", "<br>")
        + "</div>"
    )


def _render_theme_section(theme_raw: str) -> str:
    """Render accumulated expressions section."""
    try:
        data = json.loads(theme_raw)
    except Exception:
        return f"<p>{html.escape(str(theme_raw or ''))}</p>"

    expressions = data.get("expressions", [])
    if not expressions:
        return "<p>暂无可展示的积累表达。</p>"

    parts: list[str] = []
    for expr in expressions:
        category = html.escape(expr.get("category", "未分类"))
        parts.append(
            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
            f'padding:14px 18px;margin-bottom:14px;">'
        )
        parts.append(
            f'<div style="font-weight:700;font-size:15px;'
            f'margin-bottom:10px;color:#1E3A8A;">{category}</div>'
        )
        parts.append('<table style="width:100%;border-collapse:collapse;">'
                     '<tr style="vertical-align:top;">')

        col_data = [
            ("完整句子/句型", expr.get("sentences", [])),
            ("实用短语", expr.get("phrases", [])),
            ("高级词汇", expr.get("words", [])),
        ]
        for col_title, items in col_data:
            parts.append('<td style="width:33%;padding:0 8px 0 0;">')
            parts.append(f"<strong>{col_title}</strong>")
            if items:
                parts.append('<ul style="margin:4px 0 0 18px;">')
                parts.extend(f"<li>{html.escape(item)}</li>" for item in items)
                parts.append("</ul>")
            parts.append("</td>")

        parts.append("</tr></table></div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  HTML report assembly
# ---------------------------------------------------------------------------

_REPORT_CSS = """
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Microsoft YaHei", "SimSun", sans-serif;
    background: #ffffff;
    color: #0f172a;
    line-height: 1.7;
    padding: 32px 16px;
  }
  .container {
    max-width: 960px;
    margin: 0 auto;
    background: #ffffff;
    padding: 20px 30px;
  }
  h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 4px;
  }
  .subtitle {
    text-align: center;
    color: #64748B;
    margin-bottom: 32px;
    font-size: 14px;
  }
  .section {
    margin-bottom: 32px;
  }
  .section h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2563EB;
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 6px;
    border-bottom: 2px solid #dbeafe;
  }
  .reading-text {
    font-family: "Cambria", "Georgia", "Times New Roman", serif;
    font-size: 20px;
    line-height: 1.8;
  }
  ul { padding-left: 20px; }
  li { margin-bottom: 4px; }
  .footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 32px;
  }
"""


def build_html_report(
    corrected_text: str,
    rate_text: str,
    perfect_text: str,
    theme_text: str,
    original_image_data_url: str | None = None,
) -> str:
    """Build a self-contained HTML report string (no PDF conversion)."""

    image_section = ""
    if original_image_data_url:
        image_section = (
            '<div class="section"><h2>原始作文图片</h2>'
            f'<img src="{original_image_data_url}" '
            'style="max-width:100%;border:1px solid #e2e8f0;">'
            '</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>English Essay Assessment Report</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<div class="container">
  <h1>English Essay Assessment Report</h1>
  <p class="subtitle">Generated by EEAR &mdash; 智能英语作文批改系统</p>

  {image_section}

  <div class="section">
    <h2>批改结果</h2>
    <div class="reading-text">{_render_corrected_section(corrected_text)}</div>
  </div>

  <div class="section">
    <h2>详细评价</h2>
    {_render_rate_section(rate_text)}
  </div>

  <div class="section">
    <h2>优秀范文</h2>
    {_render_perfect_section(perfect_text)}
  </div>

  <div class="section">
    <h2>积累表达</h2>
    {_render_theme_section(theme_text)}
  </div>

  <div class="footer">English Essay Assessment Review &copy; EEAR</div>
</div>
</body>
</html>"""


def generate_report_filename() -> str:
    """Return a timestamped filename like essay_report_20260227_153012.html"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"essay_report_{ts}.html"

st.set_page_config(page_title="English Essay Assessment Review", page_icon=None, layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .reading-text {
        font-family: "Cambria", "Georgia", "Times New Roman", serif;
        font-size: 20px;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">English Essay Assessment Review (EEAR)</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B; margin-bottom: 2rem;'>智能英语作文批改系统 - 拍照上传，即刻获取专业点评与范文</p>", unsafe_allow_html=True)

common_model_options = [default_model_name] if default_model_name else []
common_model_options = sorted(set(common_model_options), key=str.lower)

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = default_base_url
if "api_key" not in st.session_state:
    st.session_state.api_key = default_api_key
if "models_from_api" not in st.session_state:
    st.session_state.models_from_api = []
if "models_fetch_error" not in st.session_state:
    st.session_state.models_fetch_error = None

model_selector_options = sorted(
    set(common_model_options + st.session_state.models_from_api),
    key=str.lower
)
if "自定义" not in model_selector_options:
    model_selector_options.append("自定义")

if "model_selector" not in st.session_state:
    st.session_state.model_selector = default_model_name if default_model_name in model_selector_options else "自定义"
if "custom_model" not in st.session_state:
    st.session_state.custom_model = default_model_name
if "essay_text_manual" not in st.session_state:
    st.session_state.essay_text_manual = ""
if "pending_essay_text_manual" not in st.session_state:
    st.session_state.pending_essay_text_manual = None
if "review_results" not in st.session_state:
    st.session_state.review_results = None
if "ui_expand_config" not in st.session_state:
    st.session_state.ui_expand_config = True
if "ui_expand_input" not in st.session_state:
    st.session_state.ui_expand_input = True
if "start_requested" not in st.session_state:
    st.session_state.start_requested = False
if "source_image_data_url" not in st.session_state:
    st.session_state.source_image_data_url = None
if "last_ocr_image_signature" not in st.session_state:
    st.session_state.last_ocr_image_signature = None

right_col, left_col = st.columns(2)

with left_col:
    with st.expander("接口设置", expanded=st.session_state.ui_expand_config):
        st.markdown('<div class="panel-title">请配置模型接口参数</div>', unsafe_allow_html=True)
        st.text_input(
            "Base URL",
            key="api_base_url",
            placeholder="例如: https://api.openai.com/v1"
        )
        st.text_input(
            "API Key",
            key="api_key",
            type="password",
            placeholder="请输入 API Key"
        )

        if st.button("拉取模型列表", use_container_width=True):
            if (st.session_state.api_key or "").strip():
                models, error = fetch_models_from_endpoint(st.session_state.api_base_url, st.session_state.api_key)
                if error:
                    st.session_state.models_fetch_error = error
                    st.session_state.models_from_api = []
                else:
                    st.session_state.models_from_api = models
                    st.session_state.models_fetch_error = None
            else:
                st.session_state.models_fetch_error = "请先填写 API Key。"
                st.session_state.models_from_api = []
            st.rerun()

        st.selectbox(
            "Model",
            options=model_selector_options,
            key="model_selector"
        )

        if st.session_state.models_from_api:
            st.caption(f"已从接口加载 {len(st.session_state.models_from_api)} 个模型")
        if st.session_state.models_fetch_error:
            st.caption(st.session_state.models_fetch_error)

        if st.session_state.model_selector == "自定义":
            st.text_input(
                "自定义 Model",
                key="custom_model",
                placeholder="请输入模型名称"
            )

uploaded_file = None
image = None
organize_button = False
submit_button = False
with right_col:
    with st.expander("作文输入", expanded=st.session_state.ui_expand_input):
        if st.session_state.pending_essay_text_manual is not None:
            st.session_state.essay_text_manual = st.session_state.pending_essay_text_manual
            st.session_state.pending_essay_text_manual = None

        uploaded_file = st.file_uploader(
            "请上传英语作文图片（可选）",
            type=["png", "jpg", "jpeg", "bmp"],
            key="essay_image_file"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

        st.text_area(
            "请输入英文作文正文",
            height=280,
            placeholder="可手动输入，或上传图片后自动识别并填充后微调",
            key="essay_text_manual"
        )
        essay_text_ready = bool((st.session_state.essay_text_manual or "").strip())
        organize_col, submit_col = st.columns(2)
        organize_button = organize_col.button(
            "自动整理",
            use_container_width=True,
            disabled=st.session_state.start_requested or (not essay_text_ready)
        )
        submit_button = submit_col.button(
            "提交批改",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.start_requested or (not essay_text_ready)
        )

if uploaded_file is not None:
    current_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.last_ocr_image_signature != current_signature:
        with st.spinner("正在识别图片中的文字"):
            try:
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_mime = uploaded_file.type if uploaded_file.type else "image/png"
                st.session_state.source_image_data_url = f"data:{uploaded_mime};base64,{base64.b64encode(uploaded_bytes).decode('utf-8')}"

                image = Image.open(uploaded_file)
                img_array = np.array(image)
                result, _ = reader(img_array)
                st.session_state.last_ocr_image_signature = current_signature

                if not result:
                    st.error("文字识别失败，请尝试上传更清晰的图片")
                else:
                    st.session_state.pending_essay_text_manual = "\n".join([item[1] for item in result])
                    st.rerun()
            except Exception as exc:
                st.session_state.last_ocr_image_signature = current_signature
                st.error(f"图片识别失败: {exc}")
else:
    st.session_state.last_ocr_image_signature = None

if organize_button:
    current_text = (st.session_state.essay_text_manual or "").strip()
    runtime_base_url = normalize_base_url(st.session_state.api_base_url)
    runtime_api_key = (st.session_state.api_key or "").strip()
    runtime_model_name = (st.session_state.custom_model if st.session_state.model_selector == "自定义" else st.session_state.model_selector).strip()

    if not current_text:
        st.error("请先输入或识别作文正文。")
    elif not runtime_api_key:
        st.error("请先填写 API Key。")
    elif not runtime_model_name:
        st.error("请先填写 Model。")
    else:
        with st.spinner("正在自动整理文本"):
            runtime_client = openai.OpenAI(api_key=runtime_api_key, base_url=runtime_base_url)
            prompt = (
                "你是英语作文文本整理助手。请只输出“整理后的英文正文”。\n\n"
                + "任务目标：\n"
                + "1) 合并被错误切开的行，使同一段内容在同一段落中。\n"
                + "2) 基于语义与常见作文结构进行合理分段（开头、论证、结尾等）。\n"
                + "3) 清理OCR噪声：乱码、异常符号、明显误识别字符（如 tne->the, ancl->and）。\n\n"
                + "严格约束：\n"
                + "- 不得新增观点、句子或信息；不得删除原文有效信息。\n"
                + "- 不做润色改写，不替换正常词汇，不增删单词。\n"
                + "- 仅在“明显OCR错误”时修正拼写。\n"
                + "- 保留原有时态、人称与语气。\n"
                + "- 输出纯英文正文，不要解释、不要标题、不要代码块。\n\n"
                + "待整理文本：\n\n"
                + current_text
            )
            completion = runtime_client.chat.completions.create(
                model=runtime_model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            organized_text = (completion.choices[0].message.content or "").strip()
            if organized_text:
                st.session_state.pending_essay_text_manual = organized_text
                st.rerun()
            else:
                st.error("自动整理失败，请重试。")

if submit_button:
    st.session_state.start_requested = True
    st.rerun()

if st.session_state.start_requested:
    runtime_base_url = normalize_base_url(st.session_state.api_base_url)
    runtime_api_key = (st.session_state.api_key or "").strip()
    runtime_model_name = (st.session_state.custom_model if st.session_state.model_selector == "自定义" else st.session_state.model_selector).strip()

    if not runtime_api_key:
        st.error("请先填写 API Key。")
        st.session_state.start_requested = False
        st.rerun()

    if not runtime_model_name:
        st.error("请先填写 Model。")
        st.session_state.start_requested = False
        st.rerun()

    runtime_client = openai.OpenAI(api_key=runtime_api_key, base_url=runtime_base_url)

    with st.spinner("正在执行智能批改"):
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_mime = uploaded_file.type if uploaded_file.type else "image/png"
            st.session_state.source_image_data_url = f"data:{uploaded_mime};base64,{base64.b64encode(uploaded_bytes).decode('utf-8')}"
        else:
            st.session_state.source_image_data_url = None

        essay_text = (st.session_state.essay_text_manual or "").strip()
        if not essay_text:
            st.error("请先输入作文正文，或上传图片自动识别")
            st.session_state.start_requested = False
            st.rerun()

        revised_text = essay_text

        prompt = (
            "你是英语作文精细批改助手。请只输出批改后的英文正文。\n\n"
            + "批改目标：\n"
            + "- 找出并修正语法、拼写、搭配、不地道表达。\n"
            + "- 保持原意，做最小必要修改，不随意重写整句。\n"
            + "- 优先自然、准确、清晰。\n\n"
            + "标注格式（严格执行）：\n"
            + "- 删除： (text){理由}\n"
            + "- 新增： [text]{理由}\n"
            + "- 替换： (old){理由} [new]{理由}\n"
            + "- 红框()与绿框[]都可加理由，但按情况决定添加的位置，同一个位置同时出现两种框时如果无都写必要就只选择红框写理由。\n"
            + "- 必加理由：删除、替换旧词(old)。\n"
            + "- 选加理由：纯补全类新增（如冠词/介词/to）可不写；若涉及语法、搭配、语义纠偏则要写。\n"
            + "- 理由必须紧跟对应括号，不能悬空。\n"
            + "- 每条理由最多8个汉字。\n"
            + "- 标注尽量到词/短语级，避免整句大范围加括号。\n\n"
            + "标注示例：\n"
            + "- 删除：In my opinion, (I think){重复表达} we should act now.\n"
            + "- 新增：I want [to]{缺不定式} go home.\n"
            + "- 替换：She (are){主谓不一致} [is]{单数形式} a student.\n"
            + "- 替换：(Becouse){拼写错误} [Because]{正确拼写} I was busy.\n\n"
            + "输出要求：\n"
            + "- 只输出批改后的正文；不要解释、不要标题、不要代码块。\n"
            + "- 保留原段落结构。\n\n"
            + "待批改作文：\n\n"
            + revised_text
        )
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        corrected_text = completion.choices[0].message.content

        rate_prompt = """
你是资深英语写作评估专家（IELTS/CEFR/EFL）。请基于学生原文给出客观、可执行的诊断。

评分规则：
1) fluency（语言通顺度）
2) coherence（上下文连贯度）
3) accuracy（词汇与语法准确度）

每个维度要求：
- score: 0-100 的整数
- comments: 2-4 条，必须结合原文具体现象（指出优点与问题）
- suggestions: 2-3 条，可直接落地执行，尽量给出可替换表达或修改方向

输出要求：
- 只能输出合法 JSON，不要任何额外文字，评价使用中文，结合的具体内容使用英文
- 字段必须完整：fluency/coherence/accuracy，每个都要有 score/comments/suggestions

请严格按以下结构输出：
{
    "fluency": {"score": 0, "comments": [], "suggestions": []},
    "coherence": {"score": 0, "comments": [], "suggestions": []},
    "accuracy": {"score": 0, "comments": [], "suggestions": []}
}

下面是这篇英文习作的原文：
"""
        prompt_review = rate_prompt + "\n\n" + revised_text
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt_review}],
            response_format={"type": "json_object"}
        )
        rate_text = completion.choices[0].message.content

        rewrite_prompt = """
    你是英语写作提升教练。请在不改变核心立意的前提下，重写这篇作文为“可学习、可模仿”的高质量版本。

    要求：
    - 语言自然地道，优先清晰与流畅，不堆砌生僻词
    - 句式有变化（简单句、复合句、过渡句）
    - 保持原文主题与主要信息，不凭空添加事实
    - 保留合理段落结构

    输出要求：
    - 只输出英文重写正文，不要解释、不要标题、不要代码块、不要加粗强调等任何格式化，保持纯文本输出

    下面是这篇英文习作原文：\n\n"""
        prompt = rewrite_prompt + revised_text
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        perfect_text = completion.choices[0].message.content

        theme_prompt = """
你是英语教学专家。请从作文中提炼“可复用表达素材”，用于学生积累。

提取原则：
- 紧贴原文主题，适用于同类写作
- 覆盖不同功能场景（如引入、论证、让步、总结、呼吁）
- 句子/短语/词汇都要“可直接使用”
- 语言自然，不要生造表达

数量建议（可弹性）：
- category: 2-5 个（使用中文描述具体内容）
- 每个 category 下：sentences 2-4 条，phrases 3-6 条，words 3-6 条

输出要求：
- 只能输出合法 JSON，不要任何额外文字
- 严格使用以下结构：
{
    "expressions": [
        {
            "category": "...",
            "sentences": ["..."],
            "phrases": ["..."],
            "words": ["..."]
        }
    ]
}

下面是这篇英文习作原文：\n\n"""
        prompt = theme_prompt + revised_text
        completion = runtime_client.chat.completions.create(
            model=runtime_model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        theme_text = completion.choices[0].message.content

        st.session_state.review_results = {
            "corrected_text": corrected_text,
            "rate_text": rate_text,
            "perfect_text": perfect_text,
            "theme_text": theme_text,
            "source_image_data_url": st.session_state.source_image_data_url,
        }
        st.session_state.start_requested = False
        st.rerun()

if st.session_state.review_results is not None:
    st.markdown("---")
    st.markdown("### 批改结果")
    
    corrected_text = st.session_state.review_results["corrected_text"]
    rate_text = st.session_state.review_results["rate_text"]
    perfect_text = st.session_state.review_results["perfect_text"]
    theme_text = st.session_state.review_results["theme_text"]
    source_image_data_url = st.session_state.review_results.get("source_image_data_url")

    tab1, tab2, tab3, tab4 = st.tabs(["批改结果", "详细评价", "优秀范文", "积累表达"])

    with tab1:
        st.markdown(f'<div class="reading-text">{highlight_brackets(corrected_text)}</div>', unsafe_allow_html=True)

    with tab2:
        try:
            rate_data = json.loads(rate_text)
            overall_score = int((rate_data.get("fluency", {}).get("score", 0) + rate_data.get("coherence", {}).get("score", 0) + rate_data.get("accuracy", {}).get("score", 0)) / 3)
            st.markdown(f"<h3 style='text-align: center; color: #1E3A8A;'>综合得分: {overall_score}/100</h3>", unsafe_allow_html=True)
            st.progress(overall_score / 100)
            st.markdown("<br>", unsafe_allow_html=True)

            col_f, col_c, col_a = st.columns(3)
            with col_f:
                st.metric(label="语言通顺度 (Fluency)", value=f"{rate_data.get('fluency', {}).get('score', 0)}/100")
                st.markdown("**评价:**")
                for comment in rate_data.get("fluency", {}).get("comments", []):
                    st.markdown(f"- {comment}")
                st.markdown("**建议:**")
                for suggestion in rate_data.get("fluency", {}).get("suggestions", []):
                    st.markdown(f"- {suggestion}")

            with col_c:
                st.metric(label="上下文连贯度 (Coherence)", value=f"{rate_data.get('coherence', {}).get('score', 0)}/100")
                st.markdown("**评价:**")
                for comment in rate_data.get("coherence", {}).get("comments", []):
                    st.markdown(f"- {comment}")
                st.markdown("**建议:**")
                for suggestion in rate_data.get("coherence", {}).get("suggestions", []):
                    st.markdown(f"- {suggestion}")

            with col_a:
                st.metric(label="词汇与语法 (Accuracy)", value=f"{rate_data.get('accuracy', {}).get('score', 0)}/100")
                st.markdown("**评价:**")
                for comment in rate_data.get("accuracy", {}).get("comments", []):
                    st.markdown(f"- {comment}")
                st.markdown("**建议:**")
                for suggestion in rate_data.get("accuracy", {}).get("suggestions", []):
                    st.markdown(f"- {suggestion}")
        except json.JSONDecodeError:
            st.error("评分数据解析失败，请重试。")

    with tab3:
        st.markdown(f'<div class="reading-text">{html.escape(perfect_text).replace("\n", "<br>")}</div>', unsafe_allow_html=True)

    with tab4:
        try:
            theme_data = json.loads(theme_text)
            expressions = theme_data.get("expressions", [])
            if not expressions:
                st.info("暂无可展示的积累表达。")
            for expr in expressions:
                category = expr.get("category", "未分类")
                with st.expander(category, expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**完整句子/句型**")
                        for sentence in expr.get("sentences", []):
                            st.markdown(f"- {sentence}")
                    with c2:
                        st.markdown("**实用短语**")
                        for phrase in expr.get("phrases", []):
                            st.markdown(f"- {phrase}")
                    with c3:
                        st.markdown("**高级词汇**")
                        for word in expr.get("words", []):
                            st.markdown(f"- {word}")
        except json.JSONDecodeError:
            st.error("表达数据解析失败，请重试。")

    # ---- Build report & provide download ----
    report_html = build_html_report(
        corrected_text=corrected_text,
        rate_text=rate_text,
        perfect_text=perfect_text,
        theme_text=theme_text,
        original_image_data_url=source_image_data_url,
    )

    st.markdown("---")
    st.download_button(
        label="下载完整批改报告",
        data=report_html,
        file_name=generate_report_filename(),
        mime="text/html",
        use_container_width=True,
>>>>>>> e09da4f (Add requirements and runtime files; implement English essay assessment tool with OCR and OpenAI integration)
    )