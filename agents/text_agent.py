from typing import Optional
import json
import ast
from .base_agent import BaseAgent
# from base_agent import BaseAgent
from utils.yaml_loader import prompt_loader


class TextAgent(BaseAgent):
    def __init__(
            self,
            model_name: str,
            system_prompt: Optional[str] = prompt_loader("text_agent"),
            **kwargs
    ):
        """
        text_agent 是一个用于文本处理的 agent，负责：
        - 文本处理
        - 文本生成
        - 文本总结
        :param model_name: 模型名称，如 "qwen_2_5", "qwen_vl_2_5"
        :param system_prompt: 系统提示词
        :param kwargs: 模型初始化参数（如 temperature）
        """
        super().__init__(model_name, system_prompt, **kwargs)

    def text_process(self, text_input: str) -> dict:
        """
        文本处理
        :param text_input: 用户输入的文本 json格式
        {
        "history":
            [
                {
                    "question": ...,
                    "answer": ...
                }
            ],
        "question": ...,
        "text": ...
        }
        :return: agent输出的回答，json格式
        {
        "Reasoning (Chain-of-Thought)": ...
        "answer": ...
        }
        """
        raw_output = self.generate(text_input=text_input)
        output = raw_output[raw_output.rfind('{'):raw_output.rfind('}') + 1]
        print(output)
        try:
            return ast.literal_eval(output)
        except Exception as e:
            print("json.loads failed")
            return {"answer": "NaN"}


if __name__ == "__main__":
    agent = TextAgent(model_name="qwen_2_5")
    text = """
    {
	"history": [],
  "question": "What does the map in the report shows?",
  "text": "During the year, ISRO organised media visits to SDSC SHAR, Sriharikota, ISRO Satellite Centre (ISAC) and Mission Operations Complex (MOX), ISTRAC Bengaluru for the live coverage of PSLV and GSLV launches, ‘GNSS User Meet 2015’ and Mars Orbiter Mission coverage respectively. ."
}
    """
    answer = agent.text_process(text)
    print(answer.keys())
    print(answer["answer"])
    print(len(answer["answer"]))
