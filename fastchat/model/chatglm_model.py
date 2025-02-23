import torch
import torch_mlu
from typing import List, Tuple


def stream_chat_token_num(tokenizer, query: str, history: List[Tuple[str, str]] = None):
    if history is None:
        history = []
    #TODO: 如果历史记录为空，将当前问题作为提示（prompt）
    if not history:
        prompt = query
    else:
        #如果有历史记录，将历史记录和当前问题组合成一个提示
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        #TODO: 最后一轮的问题格式为 "[Round len(history)]\n问：query\n答："
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    #TODO 使用分词器tokenizer对提示进行编码
    inputs = tokenizer(prompt, return_tensors="pt")
    return sum([len(x) for x in inputs["input_ids"]])


@torch.inference_mode()
def chatglm_generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    """Generate text using model's chat api"""
    messages = params["prompt"]
    #TODO 从参数中获取生成文本时的max_new_tokens数，如果参数中未指定，则默认为 256。
    max_new_tokens = params.get("max_new_tokens", 256)
    #TODO: 从参数中获取生成文本时的temperature（控制文本生成的多样性），如果参数中未指定，则默认为 1.0。
    temperature = params.get("temperature", 1.0)
    #TODO:从参数中获取生成文本时的 top_p（控制生成文本的多样性），如果参数中未指定，则默认为 1.0。
    top_p = params.get("top_p", 1.0)
    #TODO: 从参数中获取生成文本时的 repetition penalty（抑制文本中重复的程度），如果参数中未指定，则默认为 1.0。
    repetition_penalty =params.get("repetition_penalty", 1.0)
    #TODO: 从参数中获取是否在生成的文本中包含输入的历史消息，如果参数中未指定，则默认为 True。
    echo = params.get("echo", True)

    gen_kwargs = {
        # "max_new_tokens": max_new_tokens,  disabled due to a warning.
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": None,
    }
    #TODO: 如果 temperature 大于 1e-5，将温度参数添加到 gen_kwargs 中。
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    hist = []
    for i in range(0, len(messages) - 2, 2):
        #TODO: 遍历消息列表，将奇数索引的消息作为前一方的发言，偶数索引的消息作为后一方的发言，并添加到对话历史hist中。
        hist.append((messages[i][1], messages[i+1][1]))
    query = messages[-2][1]
    
    #TODO:# 计算输入历史的 token 数
    input_echo_len = stream_chat_token_num(tokenizer, query, hist)

    for i, (response, new_hist) in enumerate(
        model.stream_chat(tokenizer, query, hist, **gen_kwargs)
    ):
        if echo:
            output = query + " " + response
        else:
            output = response

        yield {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": None,
        }

    # TODO: 最后一轮的生成结果包含完成原因，将完成原因设为 "stop"
    # Only last stream result contains finish_reason, we set finish_reason as stop
    
    ret = {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": "stop"
    }
    yield ret
