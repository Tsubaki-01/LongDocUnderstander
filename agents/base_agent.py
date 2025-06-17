import importlib
from typing import List, Optional, Dict, Any
import os
from pathlib import Path

import yaml

from models import qwen
from utils.yaml_loader import api_key_loader

# 支持的模型名称映射到具体类
# MODEL_REGISTRY = \
#     {
#         "qwen_2_5": qwen.Qwen_2_5_api,
#         "qwen_vl_2_5": qwen.Qwen_VL_2_5_api,
#     }
# API_REGISTRY = \
#     {
#         "qwen_2_5": "qwen",
#         "qwen_vl_2_5": "qwen",
#     }

# 读取配置文件
with open(Path(__file__).parent.parent / "config" / "registry.yaml", 'r') as f:
    config = yaml.safe_load(f)

# 动态构建 MODEL_REGISTRY
model_registry = {}
api_registry = config["API_REGISTRY"]
for key, value in config["MODEL_REGISTRY"].items():
    module_name, class_name = value.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_registry[key] = getattr(module, class_name)


class BaseAgent:
    def __init__(
            self,
            model_name: str,
            system_prompt: Optional[str] = None,
            **model_kwargs
    ):
        """
        BaseAgent 是所有 agent 的基类，负责：
        - 加载 system_prompt
        - 初始化指定的模型
        - 提供统一生成接口

        :param model_name: 模型名称，如 "qwen_2_5", "qwen_vl_2_5"
        :param model_kwargs: 模型初始化参数（如 temperature）
        """
        if model_name not in model_registry:
            raise ValueError(f"不支持的模型: {model_name}. 支持列表: {list(model_registry.keys())}")

        # 加载 system_prompt
        self.system_prompt = system_prompt

        # 加载 api-key
        self.api_key = api_key_loader(api_registry[model_name])

        # 初始化对应模型
        model_class = model_registry[model_name]
        self.model = model_class(
            system_prompt=self.system_prompt,
            api_key=self.api_key,
            **model_kwargs
        )

    def _generate(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> str:
        """
        统一生成方法，由子类调用

        :param text_input: 文本输入
        :param image_input: 图片输入路径列表
        :return: 模型生成的回答
        """
        answer = self.model.generate(
            text_input=text_input,
            image_input=image_input
        )
        return answer

    def generate(
            self,
            text_input: str = None,
            image_input: List[str] = None
    ) -> str:
        """
        统一生成方法，由子类调用

        :param text_input: 文本输入
        :param image_input: 图片输入路径列表
        :return: 模型生成的回答
        """
        return self._generate(text_input=text_input, image_input=image_input)
