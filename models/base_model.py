from typing import List, Optional, Dict, Any
import torch
import json
from utils import image_encoder
from openai import OpenAI
import os


class BaseModel:
    def __init__(
            self,
            model_name: str,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1024,
            system_prompt: Optional[str] = None,
            **kwargs
    ):
        """
        模型基类，所有具体模型需继承此类并实现 generate 方法。

        :param model_name: 模型名称
        :param api_key: API 密钥（如 DashScope）
        :param base_url: 模型服务地址
        :param temperature: 控制生成随机性
        :param max_tokens: 最大输出 token 数量
        :param system_prompt: 系统提示词（可选）
        :param kwargs: 其他自定义参数
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.system_prompt = system_prompt or ""

        # 支持额外参数存储
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def create_messages(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> List[dict]:
        """
        构建模型输入消息，需由子类实现
        :param text_input: 文本输入内容
        :param image_input: 图片输入路径列表
        :return: 构建的消息内容
        """

        raise NotImplementedError("子类必须实现 create_message 方法")

    def generate(
            self,
            text_input: dict = None,
            image_input: List[str] = None
    ) -> str:
        """
        核心生成方法，需由子类实现

        :param text_input: 文本输入内容
        :param image_input: 图片输入路径列表
        :return: 模型输出结果
        """
        raise NotImplementedError("子类必须实现 generate 方法")

    def clean_up(self):
        torch.cuda.empty_cache()

    def set_params(self, **kwargs):
        """
        动态设置模型参数（例如：temperature、max_tokens）

        :param kwargs: 任意参数（如 temperature=0.5）
        """
        valid_keys = self.__dict__.keys()
        for key, value in kwargs.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                raise AttributeError(f"未知参数 {key}，不支持设置。")

    def get_params(self) -> Dict[str, Any]:
        """
        获取当前模型参数配置

        :return: 当前参数字典
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.model_name})>"
