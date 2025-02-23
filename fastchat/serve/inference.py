"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Iterable, Optional
import sys
import warnings

import psutil
import torch
import torch_mlu
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.model.chatglm_model import chatglm_generate_stream


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    #TODO：如果温度大于等于 1e-5 且不等于 1.0，就添加 TemperatureLogitsWarper 处理器
    if temperature >= 1e-5 and temperature != 1.0:
        __________________________________________________________________________
    #TODO: 如果重复惩罚大于 1.0，就添加 RepetitionPenaltyLogitsProcessor 处理器
    if repetition_penalty > 1.0:
        __________________________________________________________________________
    #TODO: 如果 top_p 在 (1e-8, 1.0) 范围内，就添加 TopPLogitsWarper 处理器    
    if 1e-8 <= top_p < 1.0:
        __________________________________________________________________________
    #TODO: 如果 top_k 大于 0，就添加 TopKLogitsWarper 处理器
    if top_k > 0:
        __________________________________________________________________________
    return processor_list


#@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    #TODO:# 关闭梯度计算
    __________________________________
    prompt = params["prompt"]
    #TODO: 获取prompt的长度
    len_prompt = _________________________________________________
    #TODO: 从参数中获取生成文本时的temperature（控制文本生成的多样性），如果参数中未指定，则默认为 1.0。
    temperature = _________________________________________________
    #TODO: 从参数中获取生成文本时的 repetition penalty（抑制文本中重复的程度），如果参数中未指定，则默认为 1.0。
    repetition_penalty =_________________________________________________
    #TODO:从参数中获取生成文本时的 top_p（控制生成文本的多样性），如果参数中未指定，则默认为 1.0。
    top_p = _________________________________________________
    #TODO: 从输入的参数中获取 top_k 的值，如果参数中没有设置，则默认为 -1。
    top_k = _________________________________________________  # -1 means disable
    #TODO 从参数中获取生成文本时的max_new_tokens数，如果参数中未指定，则默认为 256。
    max_new_tokens = _________________________________________________
    #TODO: 从参数中获取 stop_str，如果未设置，则默认为 None
    stop_str =  _________________________________________________
    #TODO: 从参数中获取 echo，如果未设置，默认为 True。并将其转换为布尔值。
    echo =  _________________________________________________
    stop_token_ids = params.get("stop_token_ids", None) or []
    #TODO：将文本生成停止的标记添加到已有的停止标记列表stop_token_ids中。
    _________________________________________________

    #TODO: 创建一个 logits 处理器列表
    logits_processor = _______________________________________________

    #TODO: 使用tokenizer将输入文本转换为模型可接受的输入张量
    input_ids = ______________________________________________________
    #TODO: 记录输入文本的长度
    input_echo_len = _________________________________________
    #TODO: 创建一个名为 output_ids 的列表，其初始值等于 input_ids 列表的内容
    output_ids = _________________________________________

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    #TODO: 截取源文本的最后一部分，以适应模型的上下文长度
    input_ids = _________________________________________

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                #TODO: 使用解码器对起始标记进行处理
                out = model.decoder(
                    input_ids=__________________________,
                    encoder_hidden_states=______________,
                    use_cache=__________________________,
                )
                logits = model.lm_head(out[0])
            else:
                #TODO: 非编码解码器类型，直接使用模型进行处理
                out = _______________________________________________
                logits = out.logits
            #TODO: 记录过去的键值
            past_key_values = ______________________________________
        else:
            if model.config.is_encoder_decoder:
                #TODO: 使用解码器对当前标记进行处理
                out = model.decoder(
                    input_ids=________________________________,
                    encoder_hidden_states=____________________,
                    use_cache=________________________________,
                    past_key_values=__________________________,
                )
                #TODO: 获取logits（预测的标记分布）
                logits = ______________________________________
            else:
                out = model(
                    #TODO:非编码解码器类型，直接使用模型对当前标记进行处理
                    input_ids=________________________________,
                    use_cache=________________________________,
                    past_key_values=__________________________,
                )
                logits = out.logits
            #TODO: 更新过去的键值
            past_key_values = ______________________________________

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            #TODO: 使用logits_processor处理最后一个标记的logits
            last_token_logits = _____________________________________
        else:
            #TODO: 没有logits处理器，直接获取最后一个标记的logits
            last_token_logits = ______________________________________

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            #TODO:# 通过argmax获取概率最高的标记索引，并将其转换为整数类型。
            token = ______________________________________
        else:
            #TODO: 使用softmax将概率分布归一化
            probs = ______________________________________
            #TODO: 使用torch.multinomial函数根据概率分布采样生成标记,并转换为整数类型。
            token = ______________________________________
        
        # 将生成的token添加到输出序列output_ids中
        ______________________________________

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        #TODO：判断是否达到生成结果的间隔、已完成生成最大标记数、或者已经停止生成 
        if _______________________ or _______________________ or _______________________:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            if stop_str:
                if isinstance(stop_str, str):
                    #TODO：在输出字符串中从右向左搜索停止标记的位置,rfind_start参数指定了搜索的起始位置
                    pos = ______________________________________
                    if pos != -1:
                        #TODO: 将输出字符串截断，仅保留停止标记位置之前的部分
                        output = ______________________________________
                        stopped = True
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        #TODO: 从指定位置（rfind_start）向前搜索每个停止标记在输出字符串中的最后出现位置
                        pos = ______________________________________
                        if pos != -1:
                            #TODO: 将输出字符串截断，仅保留停止标记位置之前的部分
                            output = ______________________________________
                            stopped = True
                            break
                else:
                    raise ValueError("Invalid stop field type.")

            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    #TODO: 返回生成的文本、使用情况和结束原因的字典
    yield {
        ____________________________________
        ____________________________________
        ____________________________________
        ____________________________________
        ____________________________________
        ____________________________________
        ____________________________________
    }

    # clean
    del past_key_values, out
    gc.collect()
    #TODO:释放MLU设备上的缓存空间
    ____________________________________


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    temperature: float,
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
):
    #TODO：调用load_model函数加model和tokenizer
    model, tokenizer =_______________________________________
    is_chatglm = "chatglm" in str(type(model)).lower()

    #TODO: 如果提供了对话模板，使用提供的模板,调用get_conv_template创建会话对象
    if conv_template:
        conv = _______________________________________
    else:
        #TODO:否则使用默认的对话模板,调用get_conversation_template创建会话对象
        conv = _______________________________________
    print("GENERATE STEAM PASS!")   
    while True:
        try:
            #TODO: 尝试获取用户输入，传入 conv.roles[0] 作为角色标识
            inp = _______________________________________
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            generate_stream_func = chatglm_generate_stream
            prompt = conv.messages[conv.offset :]
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        #TODO: 获取机器对话输出，传入 conv.roles[1] 作为角色标识
        _______________________________________
        output_stream = generate_stream_func(model, tokenizer, gen_params, device)
        #TODO：# 输出对话的流式输出
        outputs =_______________________________________
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()
        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        print("FASTCHAT INFERENCE PASS!")
