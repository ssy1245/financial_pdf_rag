"""调用 DeepSeek，根据已经组装好的消息生成回答。"""

import os

from openai import OpenAI


class DeepSeekGenerator:
    def __init__(self, model="deepseek-flash"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未设置环境变量 DEEPSEEK_API_KEY")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=60.0,
            max_retries=2,
        )

    def generate(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("回答因长度限制被截断，请调整输出限制")

        answer = choice.message.content
        if not answer or not answer.strip():
            raise RuntimeError("DeepSeek 未返回有效回答")

        return answer.strip()

    def generate_turn(self, messages, tools, tool_choice="auto"):
        """保留工具请求及 reasoning_content，供下一轮原样回传。"""
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools,
            tool_choice=tool_choice, stream=False,
        )
        choice = response.choices[0]
        if choice.finish_reason not in ("stop", "tool_calls"):
            raise RuntimeError("模型未正常完成本轮生成")
        raw = choice.message.model_dump(exclude_none=True)
        message = {key: value for key, value in raw.items()
                   if key in ("role", "content", "tool_calls", "reasoning_content")}
        message.setdefault("content", None)
        calls = message.get("tool_calls", [])
        if choice.finish_reason == "tool_calls" and not calls:
            raise RuntimeError("模型返回了空工具请求")
        if not calls and not (message.get("content") or "").strip():
            raise RuntimeError("模型未返回有效回答")
        return message


    def stream_turn(self, messages, tools, tool_choice="auto", on_text=None):
        """流式拼接工具参数；仅把回答文本发给 UI，不展示推理内容。"""
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools,
            tool_choice=tool_choice, stream=True,
        )
        message = {"role": "assistant", "content": ""}
        calls = {}
        finish = None
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta.content:
                    message["content"] += delta.content
                    if on_text:
                        on_text(delta.content)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    message["reasoning_content"] = message.get("reasoning_content", "") + reasoning
                for part in delta.tool_calls or []:
                    call = calls.setdefault(part.index, {"id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    if part.id:
                        call["id"] = part.id
                    if part.function:
                        if part.function.name:
                            call["function"]["name"] += part.function.name
                        if part.function.arguments:
                            call["function"]["arguments"] += part.function.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason
        finally:
            stream.close()
        if finish not in ("stop", "tool_calls"):
            raise RuntimeError("流式回答中断或被截断，不能当作完整回答")
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
        elif finish == "tool_calls" or not message["content"].strip():
            raise RuntimeError("模型未返回有效回答或工具请求")
        return message
