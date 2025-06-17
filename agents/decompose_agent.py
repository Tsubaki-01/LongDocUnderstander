from typing import Optional
import json
import ast
from .base_agent import BaseAgent
# from base_agent import BaseAgent
from utils.yaml_loader import prompt_loader


class DecomposeAgent(BaseAgent):
    def __init__(
            self,
            model_name: str,
            system_prompt: Optional[str] = prompt_loader("decompose_agent"),
            **kwargs
    ):
        """
        decompose_agent 是一个用于分解任务的 agent，负责：
        - 分解任务
        - 生成子任务

        :param model_name: 模型名称，如 "qwen_2_5", "qwen_vl_2_5"
        :param system_prompt: 系统提示词
        :param kwargs: 模型初始化参数（如 temperature）
        """
        super().__init__(model_name, system_prompt, **kwargs)

    def decompose(self, question: str) -> dict:
        """
        分解任务
        :param question: 用户输入的任务
        :return: agent生成的回答，dict格式
        {
        "Reasoning (Chain-of-Thought)": ...
        "answer":
            {
                "question1": ...,
                "question2": ...,
                ...
            }
        }
        """
        raw_output = self.generate(text_input="question: " + question)
        output = raw_output[raw_output.find('{'):raw_output.rfind('}') + 1]
        print(output)
        try:
            return ast.literal_eval(output)
        except Exception as e:
            print("json loads failed")
            return {"answer":  {"question1": question}}


if __name__ == "__main__":
    agent = DecomposeAgent(model_name="qwen_2_5")
    answer = agent.decompose("How many strengths and weaknesses are metioned in Appendix E?")
    print(answer.keys())
    print(answer["answer"])
    print(len(answer["answer"]))
