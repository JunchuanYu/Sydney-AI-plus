# -*- coding:utf-8 -*-
from __future__ import annotations
from typing import TYPE_CHECKING, List

import logging
import json
import os
import requests

from tqdm import tqdm

from utils import *


if TYPE_CHECKING:
    from typing import TypedDict

    class DataframeData(TypedDict):
        headers: List[str]
        data: List[List[str | int | bool]]


initial_prompt = "You are a helpful assistant."
API_URL = "https://api.openai.com/v1/chat/completions"

def get_response(
    openai_api_key, system_prompt, history, stream, selected_model
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    history = [construct_system(system_prompt), *history]

    payload = {
        "model": selected_model,
        "messages": history,  # [{"role": "user", "content": f"{inputs}"}],
        "temperature": 1.0,  # 1.0,
        "top_p": 1.0,  # 1.0,
        "n": 1,
        "stream": stream,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    if stream:
        timeout = timeout_streaming
    else:
        timeout = timeout_all

    # 获取环境变量中的代理设置
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")

    # 如果存在代理设置，使用它们
    proxies = {}
    if http_proxy:
        logging.info(f"Using HTTP proxy: {http_proxy}")
        proxies["http"] = http_proxy
    if https_proxy:
        logging.info(f"Using HTTPS proxy: {https_proxy}")
        proxies["https"] = https_proxy

    # 如果有代理，使用代理发送请求，否则使用默认设置发送请求
    if proxies:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
            proxies=proxies,
        )
    else:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=timeout,
        )
    return response


def stream_predict(
    openai_api_key,
    system_prompt,
    history,
    inputs,
    chatbot,
    all_token_counts,
    selected_model,
    fake_input=None,
    display_append=""
):
    def get_return_value():
        return chatbot, history, status_text, all_token_counts
    # logging.info("实时回答模式")
    partial_words = ""
    counter = 0
    status_text = "answering……"
    history.append(construct_user(inputs))
    history.append(construct_assistant(""))
    if fake_input:
        chatbot.append((fake_input, ""))
    else:
        chatbot.append((inputs, ""))
    user_token_count = 0
    if len(all_token_counts) == 0:
        system_prompt_token_count = count_token(construct_system(system_prompt))
        user_token_count = (
            count_token(construct_user(inputs)) + system_prompt_token_count
        )
    else:
        user_token_count = count_token(construct_user(inputs))
    all_token_counts.append(user_token_count)
    logging.info(f"input token count: {user_token_count}")
    yield get_return_value()
    try:
        response = get_response(
            openai_api_key,
            system_prompt,
            history,
            True,
            selected_model,
        )
    except requests.exceptions.ConnectTimeout:
        status_text = (
            standard_error_msg + connection_timeout_prompt + error_retrieve_prompt
        )
        yield get_return_value()
        return
    except requests.exceptions.ReadTimeout:
        status_text = standard_error_msg + read_timeout_prompt + error_retrieve_prompt
        yield get_return_value()
        return

    yield get_return_value()
    error_json_str = ""

    for chunk in tqdm(response.iter_lines()):
        if counter == 0:
            counter += 1
            continue
        counter += 1
        # check whether each line is non-empty
        if chunk:
            chunk = chunk.decode()
            chunklength = len(chunk)
            try:
                chunk = json.loads(chunk[6:])
            except json.JSONDecodeError:
                logging.info(chunk)
                error_json_str += chunk
                status_text = f"JSON file parsing error. Please reset the conversation. received content: {error_json_str}"
                yield get_return_value()
                continue
            # decode each line as response data is in bytes
            if chunklength > 6 and "delta" in chunk["choices"][0]:
                finish_reason = chunk["choices"][0]["finish_reason"]
                status_text = construct_token_message(
                    sum(all_token_counts), stream=True
                )
                if finish_reason == "stop":
                    yield get_return_value()
                    break
                try:
                    partial_words = (
                        partial_words + chunk["choices"][0]["delta"]["content"]
                    )
                except KeyError:
                    status_text = (
                        standard_error_msg
                        + "Token count has reached the maxtoken limit. Please reset the conversation. Current Token Count: "
                        + str(sum(all_token_counts))
                    )
                    yield get_return_value()
                    break
                history[-1] = construct_assistant(partial_words)
                chatbot[-1] = (chatbot[-1][0], partial_words+display_append)
                all_token_counts[-1] += 1
                yield get_return_value()


def predict_all(
    openai_api_key,
    system_prompt,
    history,
    inputs,
    chatbot,
    all_token_counts,
    selected_model,
    fake_input=None,
    display_append=""
):
    # logging.info("一次性回答模式")
    history.append(construct_user(inputs))
    history.append(construct_assistant(""))
    if fake_input:
        chatbot.append((fake_input, ""))
    else:
        chatbot.append((inputs, ""))
    all_token_counts.append(count_token(construct_user(inputs)))
    try:
        response = get_response(
            openai_api_key,
            system_prompt,
            history,
            False,
            selected_model,
        )
    except requests.exceptions.ConnectTimeout:
        status_text = (
            standard_error_msg + connection_timeout_prompt + error_retrieve_prompt
        )
        return chatbot, history, status_text, all_token_counts
    except requests.exceptions.ProxyError:
        status_text = standard_error_msg + proxy_error_prompt + error_retrieve_prompt
        return chatbot, history, status_text, all_token_counts
    except requests.exceptions.SSLError:
        status_text = standard_error_msg + ssl_error_prompt + error_retrieve_prompt
        return chatbot, history, status_text, all_token_counts
    response = json.loads(response.text)
    content = response["choices"][0]["message"]["content"]
    history[-1] = construct_assistant(content)
    chatbot[-1] = (chatbot[-1][0], content+display_append)
    total_token_count = response["usage"]["total_tokens"]
    all_token_counts[-1] = total_token_count - sum(all_token_counts)
    status_text = construct_token_message(total_token_count)
    return chatbot, history, status_text, all_token_counts


def predict(
    openai_api_key,
    system_prompt,
    history,
    inputs,
    chatbot,
    all_token_counts,
    stream=True,
    selected_model=MODELS[0],
    use_websearch=False,
    files = None,
    should_check_token_count=True,
):  # repetition_penalty, top_k

    old_inputs = ""
    link_references = ""

    if len(openai_api_key) != 51:
        status_text = standard_error_msg + no_apikey_msg
        logging.info(status_text)
        chatbot.append((inputs, ""))
        if len(history) == 0:
            history.append(construct_user(inputs))
            history.append("")
            all_token_counts.append(0)
        else:
            history[-2] = construct_user(inputs)
        yield chatbot, history, status_text, all_token_counts
        return

    yield chatbot, history, "answering……", all_token_counts

    if stream:
        # logging.info("使用流式传输")
        iter = stream_predict(
            openai_api_key,
            system_prompt,
            history,
            inputs,
            chatbot,
            all_token_counts,
            selected_model,
            fake_input=old_inputs,
            display_append=link_references
        )
        for chatbot, history, status_text, all_token_counts in iter:
            yield chatbot, history, status_text, all_token_counts
    else:
        # logging.info("不使用流式传输")
        chatbot, history, status_text, all_token_counts = predict_all(
            openai_api_key,
            system_prompt,
            history,
            inputs,
            chatbot,
            all_token_counts,
            selected_model,
            fake_input=old_inputs,
            display_append=link_references
        )
        yield chatbot, history, status_text, all_token_counts

    logging.info(f"The current token count is{all_token_counts}")

