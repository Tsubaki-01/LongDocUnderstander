from typing import Optional, List
import json
import ast
from .base_agent import BaseAgent
# from base_agent import BaseAgent
from utils.yaml_loader import prompt_loader


class ImageAgent(BaseAgent):
    def __init__(
            self,
            model_name: str,
            system_prompt: Optional[str] = prompt_loader("image_agent"),
            **kwargs
    ):
        """
        image_agent 是一个用于图像处理的 agent，负责：
        - 图像处理
        - 图像总结
        :param model_name: 模型名称，如 "qwen_2_5", "qwen_vl_2_5"
        :param system_prompt: 系统提示词
        :param kwargs: 模型初始化参数（如 temperature）
        """
        super().__init__(model_name, system_prompt, **kwargs)

    def image_process(self, text_input: str, image_input: List[str]) -> dict:
        """
        图像处理
        :param text_input: 用户输入的文本 json格式
        {
        "history":
            [
                {
                    "question": ...,
                    "answer": ...
                }
            ],
        "question": ...
        }
        :param image_input: 用户的图片输入路径列表
        :return: agent输出的回答，json格式
        {
        "Reasoning (Chain-of-Thought)": ...
        "answer": ...
        }
        """
        raw_output = self.generate(text_input=text_input, image_input=image_input)
        output = raw_output[raw_output.rfind('{'):raw_output.rfind('}') + 1]
        print(output)
        try:
            return ast.literal_eval(output)
        except Exception as e:
            print("json loads failed")
            return {"answer": "NaN"}


if __name__ == "__main__":
    agent = ImageAgent(model_name="qwen_vl_2_5")
    text_input = """
    {
    "history": [],
    "question": "what color of the clothes is the woman in the picture wearing?"
    }
    """
    image_input = ["../tmp/1.jpg"]
    answer = agent.image_process(text_input=text_input, image_input=image_input)
    print(answer.keys())
    print(answer["answer"])
    print(len(answer["answer"]))
