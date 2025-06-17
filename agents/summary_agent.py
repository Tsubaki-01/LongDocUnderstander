from typing import Optional, List
import json
import ast
from .base_agent import BaseAgent
# from base_agent import BaseAgent
from utils.yaml_loader import prompt_loader


class SummaryAgent(BaseAgent):
    def __init__(
            self,
            model_name: str,
            system_prompt: Optional[str] = prompt_loader("summary_agent"),
            **kwargs
    ):
        """
        summary_agent 是一个用于总结任务的 agent，负责：
        - 总结任务
        - 生成总结
        :param model_name: 模型名称，如 "qwen_2_5", "qwen_vl_2_5"
        :param system_prompt: 系统提示词
        :param kwargs: 模型初始化参数（如 temperature）
        """
        super().__init__(model_name, system_prompt, **kwargs)

    def summary(self, history: str) -> dict:
        """
        总结任务
        :param history: agent的历史对话，json格式
        {
        "original_question":...,
        "history":
            [
                {
                    "question":...,
                    "answer":...
                },
                {
                    "question":...,
                    "answer":...
                },
                ...
            ]
        }
        :return: agent生成的回答，json格式
        {
        "answer": ...
        }
        """
        raw_output = self.generate(text_input=history)
        output = raw_output[raw_output.rfind('{'):raw_output.rfind('}') + 1]
        print(output)
        try:
            return ast.literal_eval(output)
        except Exception as e:
            print("json.loads failed")
            return {"answer": "NaN"}


if __name__ == "__main__":
    agent = SummaryAgent(model_name="qwen_2_5")
    text = """
    {
        "original_question": "How many datasets are used for experiments of this paper in all?",
        "history":
        [
            {
                "question": "What tasks were evaluated?",
                "answer": "Four IE tasks."
            },
            {
                "question": "How many datasets are used for experiments of this paper in all?",
                "answer": "9"
            }
        ]
    }
    """
    answer = agent.summary(text)
    print(answer.keys())
    print(answer["answer"])
