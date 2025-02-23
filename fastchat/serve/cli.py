"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.cli --model ~/model_weights/vicuna-7b
"""
import argparse
import os
import re

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

from fastchat.model.model_adapter import add_model_args
from fastchat.serve.inference import chat_loop, ChatIO


#TODO:构建SimpleChatIO类，它继承自ChatIO类
class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            #TODO: 移除文本两端的空白字符，然后按空格分割
            output_text = outputs["text"].strip().split()
            #TODO: 获取处理后的文本中单词的数量
            now = len(output_text)
            if now > pre:
                # 输出不同于前次的部分,其中，新增的内容以空格分隔的形式显示，确保每次新增的内容都在同一行，并及时刷新输出缓冲区。
                print(" ".join(output_text[pre:]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        print("CHATIO PASS!")
        return " ".join(output_text)


#TODO:构建RichChatIO类，它继承自ChatIO类
class RichChatIO(ChatIO):
    def __init__(self):
        #TODO: 创建PromptSession实例，用于获取用户输入，并设置输入历史记录
        self._prompt_session = PromptSession(history=InMemoryHistory())
        #TODO: 创建自动补全器，用于用户输入的自动完成
        self._completer =WordCompleter(
            words=["!exit", "!reset"], pattern=re.compile("$")
        )
        #TODO:创建Console 实例，在命令行界面中以更丰富的样式显示文本
        self._console = Console()

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            #TODO：启用自动建议功能
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output

        #TODO: 创建Live上下文管理器，用于实现在命令行中实时更新显示。其中，需要指定要在其上执行实时更新的console实例,每秒刷新的次数设为4
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                #TODO: 将渲染后的 markdown 文本实时更新到控制台
                live.update(markdown)
        self._console.print()
        print("CHATIO PASS!")
        return text


def main(args):
    if args.device == "mlu":
        import os
        os.environ.setdefault("PYTORCH_MLU_ALLOC_CONF", "max_split_size_mb:128")
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.style == "simple":
        chatio = SimpleChatIO()
    elif args.style == "rich":
        chatio = RichChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        #TODO:调用 chat_loop 函数，启动聊天循环
        chat_loop(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.temperature,
            args.max_new_tokens,
            chatio,
            args.debug,
        )

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    #TODO： 创建一个ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser(description="FastChat CLI")
    #TODO: 向ArgumentParser对象中添加模型相关的参数，这些参数由add_model_args函数定义
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    #TODO:试一下simple模式，同时也试一下rich模式。
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich"],
        help="Display style.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()
    main(args)
