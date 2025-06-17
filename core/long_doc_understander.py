from typing import List, Dict, Any, Optional
from agents.decompose_agent import DecomposeAgent
from agents.text_agent import TextAgent
from agents.image_agent import ImageAgent
from agents.summary_agent import SummaryAgent


class LongDocUnderstander:
    def __init__(
        self,
        decompose_agent: DecomposeAgent,
        text_agent: TextAgent,
        image_agent: ImageAgent,
        summary_agent: SummaryAgent
    ):
        """
        长文档理解器，协调多个 agent 完成任务分解 → 理解 → 总结

        :param decompose_agent: 用于分解问题的 agent
        :param text_agent: 用于处理文本输入的 agent
        :param image_agent: 用于处理图像输入的 agent
        :param summary_agent: 用于总结的 agent
        """
        self.decompose_agent = decompose_agent
        self.text_agent = text_agent
        self.image_agent = image_agent
        self.summary_agent = summary_agent

        self.history = []  # 存储子问题和答案记录

    def understand(
        self,
        original_question: str,
        document_text: str,
        document_images: List[str]
    ) -> Dict[str, Any]:
        """
        主流程：理解文档并回答原始问题

        :param original_question: 用户提出的原始问题
        :param document_text: 文档中的文本内容（可长）
        :param document_images: 文档中的图像路径列表
        :return: 最终的回答结果
        """
        self.history = []  # 存储子问题和答案记录
        # Step 1: 分解问题
        print("Step 1: 分解问题...")
        decomposed_result = self.decompose_agent.decompose(original_question)
        sub_questions = list(decomposed_result["answer"].values())

        # Step 2-4: 处理每个子问题
        for i, question in enumerate(sub_questions):
            print(f"Step {i+2}: 处理子问题: {question}")

            # 构建上下文历史
            context_input = {
                "history": self.history,
                "question": question,
                "text": document_text
            }

            # 尝试使用文本 Agent 回答
            text_response = self.text_agent.text_process(str(context_input))
            answer = text_response.get("answer")

            # 如果返回 uncertain，尝试图像 Agent
            if str(answer).strip().lower() == "uncertain":
                print("  - 文本信息不足，尝试图像理解...")
                img_response = self.image_agent.image_process(str(context_input), document_images)
                answer = img_response.get("answer")

            # 记录答案
            self.history.append({
                "question": question,
                "answer": answer
            })

        # Step 5: 总结答案
        print("Final Step: 总结最终答案...")
        final_input = {
            "original_question": original_question,
            "history": self.history
        }

        summary_result = self.summary_agent.summary(str(final_input))
        return {
            "original_question": original_question,
            "sub_questions": sub_questions,
            "history": self.history,
            "final_answer": summary_result["answer"]
        }
