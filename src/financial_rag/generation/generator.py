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