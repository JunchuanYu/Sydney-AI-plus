# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type
import logging
import json
import os
import csv
import requests
import re
import gradio as gr
from pypinyin import lazy_pinyin
import tiktoken
import mdtex2html
from markdown import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
    
use_websearch_checkbox=False
use_streaming_checkbox=True
model_select_dropdown="gpt-3.5-turbo"
# top_p=1
dockerflag = True
authflag = False

initial_prompt = "You are a helpful assistant."
API_URL = "https://api.openai.com/v1/chat/completions"
HISTORY_DIR = "history"
TEMPLATES_DIR = "./"

standard_error_msg = "Error："  
error_retrieve_prompt = "Please check the network connection and the API-Key" 
connection_timeout_prompt = "Time out" 
read_timeout_prompt = "Time out"  
proxy_error_prompt = "Proxy error"  
ssl_error_prompt = "SSL error"  
no_apikey_msg = "please check whether the input is correct"  

max_token_streaming = 3500  
timeout_streaming = 10 
max_token_all = 3500  
timeout_all = 200  
enable_streaming_option = True  
HIDE_MY_KEY = True  

SIM_K = 5
INDEX_QUERY_TEMPRATURE = 1.0
title= """\
# <p align="center">Sydney-AI 2.0<b>"""

description = """\
<p>
<p>
本应用是一款基于最新OpenAI API“gpt-3.5-turbo”开发的智能在线聊天应用。 该应用程序的运营成本由“45 Degrees Research Fellows”赞助。 目前，token 限制为 3500。如果你想取消这个限制，你可以输入你自己的 OpenAI API 密钥。 <p>
App默认角色为ChatGPT原版助手，但您也可以从模板提供的角色中进行选择。 如果您对自定义Prompt有好的建议，请联系我们！<p>
This app is an intelligent online chat app developed based on the newly released OpenAI API "gpt-3.5-turbo". The app's operating costs are sponsored by "45度科研人". Currently, the tokens is limited to 3500. If you want to remove this restriction, you can input your own OpenAI API key.<p>
The default model role of the app is the original assistant of ChatGPT, but you can also choose from the provided roles. If you have good suggestions for customizing Prompt, please contact us!<p>
"""
        
MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301",]  


if TYPE_CHECKING:
    from typing import TypedDict

    class DataframeData(TypedDict):
        headers: List[str]
        data: List[List[str | int | bool]]


def count_token(message):
    encoding = tiktoken.get_encoding("cl100k_base")
    input_str = f"role: {message['role']}, content: {message['content']}"
    length = len(encoding.encode(input_str))
    return length


def markdown_to_html_with_syntax_highlight(md_str):
    def replacer(match):
        lang = match.group(1) or "text"
        code = match.group(2)

        try:
            lexer = get_lexer_by_name(lang, stripall=True)
        except ValueError:
            lexer = get_lexer_by_name("text", stripall=True)

        formatter = HtmlFormatter()
        highlighted_code = highlight(code, lexer, formatter)

        return f'<pre><code class="{lang}">{highlighted_code}</code></pre>'

    code_block_pattern = r"```(\w+)?\n([\s\S]+?)\n```"
    md_str = re.sub(code_block_pattern, replacer, md_str, flags=re.MULTILINE)

    html_str = markdown(md_str)
    return html_str


def normalize_markdown(md_text: str) -> str:
    lines = md_text.split("\n")
    normalized_lines = []
    inside_list = False

    for i, line in enumerate(lines):
        if re.match(r"^(\d+\.|-|\*|\+)\s", line.strip()):
            if not inside_list and i > 0 and lines[i - 1].strip() != "":
                normalized_lines.append("")
            inside_list = True
            normalized_lines.append(line)
        elif inside_list and line.strip() == "":
            if i < len(lines) - 1 and not re.match(
                r"^(\d+\.|-|\*|\+)\s", lines[i + 1].strip()
            ):
                normalized_lines.append(line)
            continue
        else:
            inside_list = False
            normalized_lines.append(line)

    return "\n".join(normalized_lines)

def postprocess(
    self, y: List[Tuple[str | None, str | None]]
) -> List[Tuple[str | None, str | None]]:
    """
    Parameters:
        y: List of tuples representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.
    Returns:
        List of tuples representing the message and response. Each message and response will be a string of HTML.
    """
    if y is None or y == []:
        return []
    tag_regex = re.compile(r"^<\w+>[^<]+</\w+>")
    if tag_regex.search(y[-1][1]):
        y[-1] = (convert_user(y[-1][0]), y[-1][1])
    else:
        y[-1] = (convert_user(y[-1][0]), convert_mdtext(y[-1][1]))
    return y

def convert_mdtext(md_text):
    code_block_pattern = re.compile(r"```(.*?)(?:```|$)", re.DOTALL)
    inline_code_pattern = re.compile(r"`(.*?)`", re.DOTALL)
    code_blocks = code_block_pattern.findall(md_text)
    non_code_parts = code_block_pattern.split(md_text)[::2]

    result = []
    for non_code, code in zip(non_code_parts, code_blocks + [""]):
        if non_code.strip():
            non_code = normalize_markdown(non_code)
            if inline_code_pattern.search(non_code):
                result.append(markdown(non_code, extensions=["tables"]))
            else:
                result.append(mdtex2html.convert(non_code, extensions=["tables"]))
        if code.strip():
            code = f"```{code}\n\n```"
            code = markdown_to_html_with_syntax_highlight(code)
            result.append(code)
    result = "".join(result)
    return result

def convert_user(userinput):
    userinput = userinput.replace("\n", "<br>")
    return f"<pre>{userinput}</pre>"

def construct_text(role, text):
    return {"role": role, "content": text}


def construct_user(text):
    return construct_text("user", text)


def construct_system(text):
    return construct_text("system", text)


def construct_assistant(text):
    return construct_text("assistant", text)


def construct_token_message(token, stream=False):
    return f"Token count: {token}"


def save_file(filename, system, history, chatbot):
    logging.info("saving......")
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if filename.endswith(".json"):
        json_s = {"system": system, "history": history, "chatbot": chatbot}
        print(json_s)
        with open(os.path.join(HISTORY_DIR, filename), "w") as f:
            json.dump(json_s, f)
    elif filename.endswith(".md"):
        md_s = f"system: \n- {system} \n"
        for data in history:
            md_s += f"\n{data['role']}: \n- {data['content']} \n"
        with open(os.path.join(HISTORY_DIR, filename), "w", encoding="utf8") as f:
            f.write(md_s)
    # logging.info("保存对话历史完毕")
    return os.path.join(HISTORY_DIR, filename)


def save_chat_history(filename, system, history, chatbot):
    if filename == "":
        return
    if not filename.endswith(".json"):
        filename += ".json"
    return save_file(filename, system, history, chatbot)


def export_markdown(filename, system, history, chatbot):
    if filename == "":
        return
    if not filename.endswith(".md"):
        filename += ".md"
    return save_file(filename, system, history, chatbot)


def load_chat_history(filename, system, history, chatbot):
    # logging.info("加载对话历史中……")
    if type(filename) != str:
        filename = filename.name
    try:
        with open(os.path.join(HISTORY_DIR, filename), "r") as f:
            json_s = json.load(f)
        try:
            if type(json_s["history"][0]) == str:
                # logging.info("历史记录格式为旧版，正在转换……")
                new_history = []
                for index, item in enumerate(json_s["history"]):
                    if index % 2 == 0:
                        new_history.append(construct_user(item))
                    else:
                        new_history.append(construct_assistant(item))
                json_s["history"] = new_history
                logging.info(new_history)
        except:
            # 没有对话历史
            pass
        # logging.info("加载对话历史完毕")
        return filename, json_s["system"], json_s["history"], json_s["chatbot"]
    except FileNotFoundError:
        # logging.info("没有找到对话历史文件，不执行任何操作")
        return filename, system, history, chatbot


def load_template(filename, mode=0):
    # logging.info(f"加载模板文件{filename}，模式为{mode}（0为返回字典和下拉菜单，1为返回下拉菜单，2为返回字典）")
    lines = []
    logging.info("Loading template...")
    # filename='中文Prompts.json'
    if filename.endswith(".json"):
        with open(os.path.join(TEMPLATES_DIR, filename), "r", encoding="utf8") as f:
            lines = json.load(f)
        lines = [[i["act"], i["prompt"]] for i in lines]
    else:
        with open(
            os.path.join(TEMPLATES_DIR, filename), "r", encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)
        lines = lines[1:]
    if mode == 1:
        return sorted_by_pinyin([row[0] for row in lines])
    elif mode == 2:
        return {row[0]: row[1] for row in lines}
    else:
        choices = sorted_by_pinyin([row[0] for row in lines])
        return {row[0]: row[1] for row in lines}, gr.Dropdown.update(
            choices=choices, value=choices[0])

def sorted_by_pinyin(list):
    return sorted(list, key=lambda char: lazy_pinyin(char)[0][0])


def get_template_content(templates, selection, original_system_prompt):
    logging.info(f"Prompt:  {selection}")
    try:
        return templates[selection]
    except:
        return original_system_prompt


def reset_state():
    logging.info("Reset")
    return [], [], [], construct_token_message(0)


def reset_textbox():
    return gr.update(value="")


def hide_middle_chars(s):
    if len(s) <= 8:
        return s
    else:
        head = s[:4]
        tail = s[-4:]
        hidden = "*" * (len(s) - 8)
        return head + hidden + tail

def submit_key(key):
    key = key.strip()
    msg = f"API-Key: {hide_middle_chars(key)}"
    logging.info(msg)
    return key, msg


