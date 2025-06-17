import transformers
import torch
from typing import List, Dict, Any

from .base_model import BaseModel

class Llama(BaseModel):
    def __init__(
            self,
            model_name=None,
            api_key=None,
            base_url=None,
            temperature=0.7,
            max_tokens=2048,
            system_prompt=None,
            **kwargs
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="/data/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        创建消息列表

        :param text_input: 文本输入
        :param image_input: 图片输入路径列表
        :return: 消息列表
        """
        pass
    
    def generate(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> str:
        """
        调用 llama 模型生成回答

        :param text_input: 文本输入
        :param image_input: 图像输入 （llama 不支持图像，忽略）
        :return: 生成的回答
        """
        self.clean_up()
        messages = self.create_messages(text_input, image_input)
        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )
        self.clean_up()
        return outputs[0]["generated_text"][-1]['content']


class Llama_3_1(Llama):
    def __init__(
            self,
            model_name=None,
            api_key=None,
            base_url=None,
            temperature=0.7,
            max_tokens=2048,
            system_prompt=None,
            **kwargs
    ):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs
        )
        
    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        创建消息列表

        :param text_input: 文本输入
        :param image_input: 图像输入 （Qwen-2.5 不支持图像，忽略）
        :return: 消息列表
        """
        messages = [
            {
                "role": "system",
                "content":
                    [
                        {
                            "type": "text",
                            "text": self.system_prompt
                        }
                    ]
            },
            {
                "role": "user",
                "content":
                    [
                        {
                            "type": "text",
                            "text": "\n\n" + "**Input:** \n"
                                    + text_input
                        }
                    ]
            }
        ]

        return messages