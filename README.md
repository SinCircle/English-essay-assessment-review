# EEAR - English essay assessment review , 适合高中班级体质的英语作文批改器

## 简介

本工具旨在通过OCR技术识别英语作文图片，并结合GPT等语言大模型提供语法纠错、评分分析、范文参考及周边表达，帮助用户提升英语写作能力。支持一键生成PDF的批改报告，适合学生、教师等英语学习者使用。

## 环境要求

- python 3.X
- 需要安装requests, tkinter, pdfkit, opencv-python, plyer, markdown, openai, numpy, Pillow等
- 需自行配置百度OCR和GPT的Key
- 自行安装wkhtmltopdf，记得添加路径到Path

## 使用方法

- 双击后会弹出一个黑框然后消失，这是正常的
- 然后会弹出选择图片窗口，选择一张图片即可
- 然后会弹出浏览器，展示当前进度；同时右侧屏幕会出现计时，如果时间太长可能是API的请求被吞了，没写重试机制，只能手动重试，一般不会超过5分钟
- 最后会弹出完成通知，并在桌面生成一份pdf格式的报告

## 常见问题

> pdf生成不出来？

先检查wkhtmltopdf，如果没问题，大概是Path没配置好。

> 我没问题？

那你确实没问题。

## 写在最后

本人刚刚高中毕业，此项目纯属高中搞来为班级服务，并且经验不足，写的不好看还请见谅，欢迎和我联系，邮箱qyf061209@outlook.com
