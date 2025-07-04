#严禁外传！严禁外传！严禁外传！
#仅供学习交流使用！
#程序内服务器API密钥和密钥均为SinCircle提供，严禁外传！
#额度有限，外传会导致额度用完，无法使用
#如有需要请联系SinCircle获取！
#严禁外传！严禁外传！严禁外传！

#以上皆为校园使用的原文（防止其他班级取走，导致请求次数太多），为了保留生活气息，所以把这部分留下来了：)
import requests
import base64
import tkinter as tk
import time
import os
import threading
import markdown
import pdfkit
import re
import openai
import base64
from tkinter import filedialog, messagebox
from plyer import notification
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path #预防PDF那个神奇的Path问题

#百度OCR的KEY
API_KEY = ""
SECRET_KEY = ""

#openai的API密钥和base_url
openai.api_key = ""
openai.base_url = ""


def get_access_token(api_key, secret_key):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    response = requests.post(url, params=params)
    return response.json().get("access_token")

def ocr_image(image_path, access_token):
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/handwriting?access_token={access_token}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "image": img_data,
        "language_type": "ENG"
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()

def loading_html(text="加载中..."):
    return f"""
<div style="margin:32px 0;">
  <span class="loading-text">{text}</span>
  <style>
    .loading-text {{
      position: relative;
      display: inline-block;
      font-size: 1em;
      color: #33333389;
      overflow: hidden;
    }}
    .loading-text::after {{
      content: '';
      position: absolute;
      left: -50%;
      top: 0;
      width: 50%;
      height: 100%;
      background: linear-gradient(90deg, transparent 0%, #ffffff 50%, transparent 100%);
      animation: shine 2s infinite;
    }}
    @keyframes shine {{
      0% {{
        left: -50%;
      }}
      100% {{
        left: 100%;
      }}
    }}
  </style>
</div>
"""

class StatusWindow(threading.Thread):
    def __init__(self, code):
        super().__init__()
        self.code = code
        self._stop_event = threading.Event()
        self.root = None
        self.label = None
        self.start_time = time.time()

    def run(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        offset = random.randint(0, 500)
        self.root.geometry("350x50+{}+{}".format(
            int((self.root.winfo_screenwidth() - 350)),
            int(50 + offset)
        ))
        self.label = tk.Label(self.root, text=f"   正在批改#{self.code}   0s", font=("微软雅黑", 14), anchor='w', justify='left')
        self.label.pack(expand=True, fill="both")
        self.update_timer()
        self.root.mainloop()

    def update_timer(self):
        if self._stop_event.is_set():
            self.root.destroy()
            return
        elapsed = int(time.time() - self.start_time)
        text = f"   正在批改#{self.code}"
        time_str = f"{elapsed}s"
        total_len = 30
        spaces = " " * max(1, total_len - len(text) - len(time_str))
        self.label.config(text=f"{text}{spaces}{time_str}")
        self.root.after(1000, self.update_timer)

    def stop(self):
        self._stop_event.set()
        if self.root:
            self.root.after(0, self.root.quit)

# 公共HTML样式（抄的，抄哪里的忘了，不过感谢他）
COMMON_STYLE = """
<style>
    body { font-family: -apple-system, sans-serif; line-height: 1.6; max-width:1024px; margin:auto; padding:32px; }
    code { background: #f5f5f5; padding: 2px 4px; border-radius: 4px; }
    pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
    blockquote { border-left: 4px solid #ddd; padding-left: 12px; color: #666; }
    ul, ol { margin-left: 2em; }
    li { margin: 4px 0; }
</style>
"""

def save_html(path, html):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def save_pdf(html, pdf_path):
    pdfkit.from_string(html, pdf_path, options={
        "encoding": "UTF-8",
        "enable-local-file-access": ""
    })

def notify(title, message, timeout=10):
    notification.notify(
        title=title,
        message=message,
        app_name="地道人地道话",
        timeout=timeout
    )

def wait(seconds):
    time.sleep(seconds)

def highlight_brackets(text):
    text = re.sub(r'\[([^\[\]]+)\]', r'<span style="background:#d4f7d4">\1</span>', text)
    text = re.sub(r'\(([^\(\)]+)\)', r'<span style="background:#ffd6d6">\1</span>', text)
    text = text.replace('\n', '<br>')
    return text

def image_html(image_path):
    try:
        with open(image_path, "rb") as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode()
            return (
                '<img src="data:image/png;base64,{0}" '
                'style="display:block;width:100%;max-width:100%;height:auto;margin:10px 0;object-fit:contain;"/><hr>'
            ).format(img_b64)
    except Exception:
        return "<p>原始图片加载失败。</p><hr>"

#这个拍照是废的，不好用后面直接删掉了，但是懒得管这部分代码，想启用可以自己拿来玩，但是不好用
def take_photo_or_choose_image():
    """
    弹出拍照窗口，支持拍照或直接选择现有图片。返回图片路径或None
    """
    class PhotoCaptureApp:
        def __init__(self, master):
            self.master = master
            self.top = tk.Toplevel(master)
            self.top.title('拍照或选择图片')
            self.top.geometry('800x600')
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开摄像头")
                self.top.destroy()
                self.photo_path = None
                return
            self.canvas = tk.Canvas(self.top, width=640, height=480)
            self.canvas.pack()
            self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
            self.canvas.bind("<B1-Motion>", self.on_mouse_move)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
            self.preview_frame = tk.Frame(self.top)
            self.preview_frame.pack(pady=5)
            self.preview_labels = []
            self.preview_images = []
            btn_frame = tk.Frame(self.top)
            btn_frame.pack(pady=10)
            self.save_button = tk.Button(btn_frame, text='提交拍摄图片', command=self.save_and_exit)
            self.save_button.pack(side=tk.LEFT, padx=10)
            self.choose_button = tk.Button(btn_frame, text='选择现有图片', command=self.choose_existing_image)
            self.choose_button.pack(side=tk.LEFT, padx=10)
            self.photo_count = 0
            self.photo_label = tk.Label(self.top, text="滑动以拍摄矩形区域的图片，可以一文多张")
            self.photo_label.pack(pady=5)
            self.start_point = None
            self.end_point = None
            self.drawing = False
            self.captured_images = []
            self.desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
            pictures_dir = os.path.join(os.path.expanduser('~'), 'Pictures')
            self.photos_dir = os.path.join(pictures_dir, 'photos', 'EEARphotos')
            os.makedirs(self.photos_dir, exist_ok=True)
            self.output_path = os.path.join(self.photos_dir, f'integPhoto_{int(time.time())}.jpg')
            self.rect_id = None
            self.freeze = False
            self.photo_path = None
            self.update_frame()

        def on_mouse_down(self, event):
            self.drawing = True
            self.freeze = True
            self.start_point = (event.x, event.y)
            self.end_point = (event.x, event.y)
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='green', width=2)

        def on_mouse_move(self, event):
            if self.drawing:
                self.end_point = (event.x, event.y)
                if self.rect_id:
                    self.canvas.coords(self.rect_id, self.start_point[0], self.start_point[1], event.x, event.y)

        def on_mouse_up(self, event):
            self.drawing = False
            self.end_point = (event.x, event.y)
            if self.rect_id:
                self.canvas.coords(self.rect_id, self.start_point[0], self.start_point[1], event.x, event.y)
            self.capture_photo()
            self.freeze = False

        def update_frame(self):
            if self.freeze:
                if hasattr(self, 'current_cv_frame'):
                    img = self.current_cv_frame.copy()
                    if self.start_point and self.end_point:
                        x1, y1 = self.start_point
                        x2, y2 = self.end_point
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    imgtk = ImageTk.PhotoImage(image=pil_image)
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                    self.current_frame = imgtk
                self.top.after(10, self.update_frame)
                return
            ret, frame = self.cap.read()
            if not ret:
                self.top.after(10, self.update_frame)
                return
            self.current_cv_frame = frame.copy()
            img = frame.copy()
            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.current_frame = imgtk
            self.top.after(10, self.update_frame)

        def capture_photo(self):
            if self.start_point is not None and self.end_point is not None and hasattr(self, 'current_cv_frame'):
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                width = abs(x1 - x2)
                height = abs(y1 - y2)
                if width > 0 and height > 0:
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(x_min + width, self.current_cv_frame.shape[1])
                    y_max = min(y_min + height, self.current_cv_frame.shape[0])
                    photo = self.current_cv_frame[y_min:y_max, x_min:x_max]
                    if photo.size > 0:
                        self.captured_images.append(photo)
                        self.photo_count += 1
                        self.photo_label.config(text=f'已拍照片: {self.photo_count}')
                        self.update_preview()

        def update_preview(self):
            for label in self.preview_labels:
                label.destroy()
            self.preview_labels.clear()
            self.preview_images.clear()
            for idx, img in enumerate(self.captured_images):
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                pil_img = pil_img.resize((80, 60), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(pil_img)
                lbl = tk.Label(self.preview_frame, image=imgtk)
                lbl.grid(row=0, column=idx, padx=2)
                self.preview_labels.append(lbl)
                self.preview_images.append(imgtk)

        def vertical_concatenate(self, images, output_path):
            if not images:
                return
            widths = [img.shape[1] for img in images]
            heights = [img.shape[0] for img in images]
            max_width = max(widths)
            total_height = sum(heights)
            new_img = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
            y_offset = 0
            for img in images:
                h, w = img.shape[:2]
                new_img[y_offset:y_offset+h, 0:w] = img
                y_offset += h
            cv2.imwrite(output_path, new_img)

        def save_and_exit(self):
            if not self.captured_images:
                self.top.destroy()
                self.photo_path = None
                return
            self.vertical_concatenate(self.captured_images, self.output_path)
            self.cap.release()
            self.photo_path = self.output_path
            self.top.destroy()

        def choose_existing_image(self):
            path = filedialog.askopenfilename(
                title="请选择一张图片，最大4MB",
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
            )
            if path:
                self.cap.release()
                self.photo_path = path
                self.top.destroy()

    root = tk._default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()
    app = PhotoCaptureApp(root)
    root.wait_window(app.top)
    return getattr(app, 'photo_path', None)

if __name__ == "__main__":
    import win32gui
    import win32con
    console_window = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(console_window, win32con.SW_HIDE)
    image_path = filedialog.askopenfilename(title="请选择一张图片，大小最大4MB", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    import random
    #用来方便同学找自己的pdf
    code = f"{random.randint(0000,9999):04}"
    def html_head(title=f"批改结果", refresh=10):
        return f"""
    <html>
    {COMMON_STYLE}
    <head>
    <meta http-equiv="refresh" content="{refresh}">
    <meta charset="utf-8">
    <title>#{code}英语作文</title>
    </head>
    <body>
    """
    file_name = "#" + str(code) + "-" + now
    print(file_name)
    if not image_path:
        print("未选择图片")
        notify("未选择图片", "请选择一张英语作文的图片", timeout=3)
    else:
        status_window = StatusWindow(code)
        status_window.start()
        notify(f"您的代码是#{code}", "已添加队列", timeout=10)
        access_token = get_access_token(API_KEY, SECRET_KEY)
        result = ocr_image(image_path, access_token)
    if "words_result" in result:
        print("识别结果：")
        essay_text = "\n".join([item["words"] for item in result["words_result"]])
        print(essay_text)
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        html_path = os.path.join(documents_path, file_name + ".html")
        # 步骤1：初始页面
        img_html = image_html(image_path)
        step_html = html_head() + "<h2>批改结果</h2>\n" + "</body>"
        save_html(html_path, step_html + loading_html("正在辨认字迹") + img_html)
        print(f"\n批改结果已保存为HTML文件：{html_path}")
        os.startfile(html_path)
        # 步骤2：整理格式
        openai.default_headers = {"x-foo": "true"}
        prompt = (
            "请帮我整理下面的英语作文文本格式，只整理英文正文部分（忽略英文正文外所有部分），保证原汁原味（明显错误空格换行、乱码、非常用字符比如☰需要改正除外），出现的拼写错误也不要帮助改正：\n\n"
            + essay_text
        )
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}],
        )
        print("\nChatGPT整理后的格式：")
        revised_text = completion.choices[0].message.content
        print(revised_text)
        revised_html = markdown.markdown(revised_text)
        step_html = html_head() + "<h2>批改结果</h2>\n" + f'<div style="font-family:Times New Roman,monospace;font-size:16px;">{revised_html}</div>\n'
        save_html(html_path, step_html + loading_html("正在查找错误"))
        print(f"\n批改结果已保存为HTML文件：{html_path}")
        wait(10)
        # 步骤3：批改作文
        prompt = (
            "请帮我把下面的英语作文的语法错误改正，输出改正后的文章（改错误和不流畅之处）,请参照下面的格式要求\n\n"
            + "有更合适的表达也可以进行替换（出色的表达进行替换），注意要求地道流畅，不能使用生僻的表达。\n\n"
            + "这是格式要求：原文修改部分用()括起来，修改的部分用[]括起来，需要删除就标明(删除的内容)，需要添加就标明[添加的内容]。精准标注，不要错了一个单词把整个句子框起来。\n\n"
            + "例如：[The] (rabbish) [rubbish] thrown by (the) visitors has piled up and its lush (verdure no longer flourish) [vegetation no longer flourishes] as it (did once) [once did].\n\n"
            + "下面是需要批改的英语习作：\n\n"
            + revised_text
        )
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}],
        )
        print("\nChatGPT批改后的结果：")
        corrected_text = completion.choices[0].message.content
        print(corrected_text)
        step_html = html_head() + "<h2>批改结果</h2>\n" + f'<div style="font-family:Times New Roman,monospace;font-size:16px;">{highlight_brackets(corrected_text)}</div>\n'
        save_html(html_path, step_html + loading_html("正在详解批改意见"))
        print(f"\n批改结果已保存为HTML文件：{html_path}")
        wait(10)
        # 步骤4：批改意见
        assessment_prompt = """
下面是一份已经批改过的英语作文（改正了一些语法错误和不流畅不准确之处），其中小括号表示原文的错误，中括号表示原文的修改部分。请你根据修改的批注，按照如下格式给出批改意见：

> 英文有错误的原文1

- 错误1和解决方案
- 错误2和解决方案，后面的以此类推

> 英文有错误的原文2

- 错误1和解决方案
- 错误2和解决方案，后面的以此类推

一个错误的句子使用一个“>”开头，错误的句子下面是批注，批注的内容是对错误的解释和修改建议。

比如批注过的原文里面出现了“The (rabbish) [rubbish] thrown by visitors has piled up and its lush (verdure no longer flourish) [verdure no longer flourishes] as it (did once) [once did].”就需要点评下面的内容：

> The rabbish thrown by visitors has piled up and its lush verdure no longer flourish as it did once.

- rabbish：拼写错误
- verdure no longer flourish：动词单复数错误，其中verdure是单数，动词使用第三人称单数
- as it did once：语序错误

接下来是经过批注过的原文："""
        prompt_review = assessment_prompt + "\n\n" + corrected_text
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt_review}],
        )
        print("\nChatGPT批改意见：")
        assessment_text = completion.choices[0].message.content
        print(assessment_text)
        assessment_html = markdown.markdown(assessment_text)
        wait(10)
        # 步骤5：评分
        rate_prompt = """
你是⼀位资深英语写作批改专家，熟悉 IELTS、CEFR 及 EFL 写作评价体系。特别注意语言准确度。
请按照下列三个维度，详细具体分析，写明原因（结合原文相关内容，提到原文中哪里有问题哪里写得好），并且需要提出改进方向和方案：

## 语言通顺度(Fluency)&可读性与风格(Readability & Style)

## 上下文连贯度(Coherence)

## 词汇多样性(Lexical Resource)&语法准确性(Grammatical Accuracy)

例如：

## 语言通顺度 (Fluency) & 可读性与风格 (Readability & Style)

- 文中的句子结构变化适中，使用了一些复杂的句子和短语，如“first and foremost”和“in other words”，展现了作者对语言的掌握。
- 语句的流畅性有待提高。例如，“making an adequately choosing”该句结构不正确，应为“making an adequate choice”。此外，部分句子显得较为冗长，可能会影响读者的理解。
- 适当分割冗长的句子，确保每个句子传递清晰的信息。例如，可以将“making an adequately choosing, which presents us with more faith when facing arduous challenges and high-demanding tasks”改为“making an adequate choice boosts our confidence when facing arduous challenges and high-demanding tasks”。

## 上下文连贯度 (Coherence)

- 文章整体上能够进行较为清晰的逻辑展开，提纲挈领地介绍了选择的重要性。
- 尽管主要论点较为明确，但论述的衔接和转折有时显得不够流畅。例如，从“none of our outstanding achievements would be obtained if we haven't preserved the preciousness of this virtue”到“choosing is an action word”的过渡有些突兀，缺乏自然的逻辑连接。
- 使用更有效的过渡词或短语来增强段落之间的连贯性。例如，增加“Furthermore”或“Moreover”来连接不同的思想，使其流畅过渡。

## 词汇多样性 (Lexical Resource) & 语法准确性 (Grammatical Accuracy)

- 使用了一些较为高级的词汇，如“prerequisite”和“unleash”，展现了较好的词汇量。
- 词汇使用上还有提升空间，存在一些使用不当和语法错误，如“making an adequately choosing”不符合语法，应为“making an adequate choice”；“the possibility to meet triumph”应为“the possibility of achieving success”。
- 清晰检查词汇的搭配和语法使用，确保语法结构的正确性，尽量避免错误。建议多做语法练习，增强对词汇搭配的理解，在写作中多样化词汇的使用。

下面是这篇英文习作的原文：
        """
        prompt_review = rate_prompt + "\n\n" + revised_text
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt_review}],
        )
        print("\nChatGPT评分：")
        rate_text = completion.choices[0].message.content
        print(rate_text)
        rate_html = markdown.markdown(rate_text)
        step_html += f"""
<h2>批改意见</h2>
<div style="font-family:Times New Roman,monospace;font-size:16px;">
{assessment_html}
{rate_html}
</div>
"""
        save_html(html_path, step_html + loading_html("正在重构优秀范文"))
        print(f"\n批改结果已保存为HTML文件：{html_path}")
        wait(10)
        # 步骤6：优秀表达范文
        rewrite_prompt = """
你是⼀位资深英语写作批改专家，熟悉 IELTS、CEFR 及 EFL 写作评价体系。特别注意语言准确度。
使用优秀的英语表达重写下面这篇英文习作，要展现优秀的词汇和语法，使用地道的表达方式，尽量使用多样化的句式、短语和词汇（但是不要通篇生僻，可以使用一两个表情达意的优秀词汇，需要保证流畅度）。
不要使用生僻的表达，加粗可供学习的部分。
文风如（一些示例句子）：
I hope this message finds you well. Recently, I’ve been struggling to balance my academic workload with my extracurricular responsibilities. Although I try to plan my day, I often find myself overwhelmed and falling behind.Since you always manage to stay so organized and composed, I was wondering if you could share some practical tips or strategies for managing time more effectively. Any advice would be greatly appreciated, and I’m eager to make meaningful changes.
He stood frozen at the doorway, eyes fixed on the scattered pages lying across the floor.For a moment, his chest tightened, the sting of betrayal sharp and unexpected.A flicker of warmth rose in his chest. Not everything had been destroyed.He stood frozen at the doorway, the torn letter still in his hand. His heart, once heavy with anger, now sank with regret. Every accusing word he had thrown at Emma replayed in his mind, sharper than any silence.His breath caught in his throat as he reached her door.For a moment, he hesitated.

下面是这篇英文习作原文：\n\n"""
        prompt = rewrite_prompt + revised_text
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}],
        )
        print("\nChatGPT优秀表达的范文：")
        perfect_text = completion.choices[0].message.content
        print(perfect_text)
        perfect_html = markdown.markdown(perfect_text)
        step_html += f"""
<h2>优秀表达的范文</h2>
<div style="font-family:Times New Roman,monospace;font-size:16px;">
{perfect_html}
</div>
"""
        save_html(html_path, step_html + loading_html("正在推测作文主题和内容，并收集优秀表达"))
        print(f"\n批改结果已保存为HTML文件：{html_path}")
        wait(10)
        # 步骤7：推测主题与表达
        theme_prompt = """
请根据下面给出的作文，推测作文的主题和内容（主题和内容应该视具体作文而定，具有灵活性），并根据主题和内容给出一系列可以替换使用的优秀表达（每个可用表达的板块应该有完整句子、句型、短语、词汇四种优秀表达形式，其中句子和句型换行分隔、短语和词汇用|分隔）。
比如给出了一篇英文习作：

Dear fellow students,
As the popularity of Le'an ancient town continues to surge, it faces mounting threats from the rampant growth of tourism. The litter discarded by visitors has accumulated, and the vibrant greenery that once flourished now struggles to thrive as it once did.
It is our collective responsibility to confront this issue. We should not only take ownership of the waste we produce but also volunteer our time to clean up the streets. Furthermore, we should consider creating signage to remind tourists of the prohibition against picking branches.
Only through collaborative efforts can we safeguard this treasured heritage. Each individual's contribution is not just valuable but essential.       
Best regards,
Li Hua

你需要分析并且输出：

## 主题（有多少主题写多少主题）
- 给同学们的信件、投稿
- 实体文物、遗迹保护
- 呼吁

## 内容（有多少内容写多少内容，要切实）
- 文物遗迹遭受破坏的现状
- 保护文物遗迹的必要性
- 针对同学提出切实可行的意见和建议
- 呼吁同学参与到文物保护中来

## 可用表达（紧跟主题和内容给出一系列可以替换使用的优秀表达，不得偏题）
### 呼吁文物保护的演讲开头结尾

The destruction of our cultural heritage is a threat to our identity.

We cannot afford to let history fade into oblivion.

The preservation of historical artifacts is not just a responsibility, but a necessity.

The destruction of cultural heritage poses a direct threat to our …….

Without immediate action, we risk the irreversible loss of …….

Cultural heritage | Irreplaceable artifacts | The relentless march of time | Stewards of our past | Cultural preservation efforts | Preserve our heritage | Irreplaceable legacy

Heritage | Conservation | Restoration | Preservation | Artifacts | Monuments | Cultural identity | Legacy | Irreplaceable | Endangered

### 文物保护重要性

The loss of cultural monuments leads to a disconnect with our past.

Preserving historical artifacts allows us to honor the legacy of our ancestors.

Each artifact is a testament to the creativity, values, and resilience of those who came before us.

Cultural preservation is crucial for maintaining our …….

Without the protection of these ……, we risk losing a part of our identity.

It is our duty to protect the …… of our ancestors, as they define who we are.

Cultural preservation | Historical significance | Connect with our past | Loss of heritage | Irreplaceable artifacts | Deteriorating monuments | Vanishing heritage | Preserving the past | Honor our ancestors | Legacy of civilization

Cultural preservation | Legacy | Monuments | Artifacts | Identity | Preservation | Deterioration | Restoration | Irreplaceable | Historical importance | Conservation | Heritage | Ancestral legacy

### 针对同学并且切实可行的倡议

One simple way to contribute is by becoming informed about the importance of local monuments and artifacts.

Volunteering for heritage conservation projects is a great way to get involved.

Let’s take part in the preservation of our cultural heritage by promoting awareness among our peers.

It is important for us to …… about the significance of …… to better advocate for its preservation.

By participating in ……, we help raise awareness of the need for conservation.

Become informed | Participate in conservation efforts | Raise awareness | Join restoration projects | Volunteer for heritage protection | Support local preservation initiatives | Advocate for stronger protection laws | Promote cultural heritage conservation | Engage in educational campaigns

Volunteer | Support | Awareness | Conservation | Restoration | Initiative | Advocacy | Local projects | Protection laws | Community involvement | Heritage education | Fundraising | Action | Preservation efforts

### 对同学参与文物保护的呼吁

I urge all of you to take action in preserving our cultural heritage.

Your involvement in heritage protection can make a lasting difference.

Let’s unite in our efforts to ensure that future generations experience the richness of our past.

We are the stewards of our heritage, and it is our duty to safeguard it.

I call upon each of you to …… in preserving our …….

The preservation of our cultural legacy requires the participation of everyone, including …….

By ……, we ensure the survival of our history for the benefit of future generations.

It is time for us to act and …… for the protection of our …….

Take action | Make a difference | Unite for preservation | Ensure the survival of our heritage | Become stewards of our past | Protect our cultural legacy | Contribute to conservation | Safeguard our history | Raise our collective voice | Advocate for heritage protection

Call to action | Stewardship | Impact | Protection | Safeguard | Legacy | Preservation | Unity | Responsibility | Contribution | Awareness | Advocacy | Duty

下面是这篇英文习作原文：\n\n"""
        prompt = theme_prompt + revised_text
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[{"role": "user", "content": prompt}],
        )
        print("\nChatGPT推测的作文主题和内容：")
        theme_text = completion.choices[0].message.content
        print(theme_text)
        theme_text = theme_text.split("## 可用表达")[1]
        theme_html = markdown.markdown(theme_text)
        # 最终HTML
        final_html = html_head(refresh=-1) + f"""
<h2>批改结果</h2>
<div style="font-family:Times New Roman,monospace;font-size:16px;">
{highlight_brackets(corrected_text)}
</div>
<h2>批改意见</h2>
<div style="font-family:Times New Roman,monospace;font-size:16px;">
{assessment_html}
{rate_html}
</div>
<h2>优秀表达的范文</h2>
<div style="font-family:Times New Roman,monospace;font-size:16px;">
{perfect_html}
</div>
<h2>周边优秀表达</h2>
<div style="font-family:Times New Roman,monospace;font-size:16px;">
{theme_html}
</div>
{img_html}
</body>
</html>
"""
        save_html(html_path, final_html)
        print(f"\n批改结果已保存为HTML文件：{html_path}")
        pdf_path = os.path.join(desktop_path, file_name + ".pdf")
        save_pdf(final_html, pdf_path)
        status_window.stop()
        status_window.join()
        print(f"\n批改结果已保存为PDF文件：{pdf_path}")
        time.sleep(10)
        notify(f"#{code}批改完成!", f"批改结果已保存为PDF文件：{pdf_path}", timeout=10)
    else:
        print(result)
        notify("识别失败！", result.get("error_msg", "未知错误"), timeout=10)