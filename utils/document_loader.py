import json
import os
from typing import List, Dict, Any, Optional
import ast


class DocumentDataset:
    def __init__(
            self,
            json_path: str,
            text_dir: str,
            image_dir: str
    ):
        """
        初始化数据集类

        :param json_path: JSON 文件路径
        :param text_dir: 文本文件存储目录
        :param image_dir: 图像文件存储目录
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.text_dir = text_dir
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        doc_id = item["doc_id"]
        evidence_pages = ast.literal_eval(item["evidence_pages"])  # 将字符串转为 list[int]
        # evidence_pages = item["evidence_pages"]
        question = item["question"]
        answer = item["answer"]
        

        # 获取文本内容
        # text_content = self._load_text(doc_id, evidence_pages)
        text_content = self._load_text(doc_id, item["text-top-4"])

        # 获取图像路径
        # image_paths = self._get_image_paths(doc_id, evidence_pages)
        image_paths = self._get_image_paths(doc_id, item["image-top-4"])

        return {
            "doc_id": doc_id,
            "question": question,
            "text": text_content,
            "images": image_paths,
            "answer": answer
        }

    def _load_text(self, doc_id: str, pages: List[int]) -> str:
        """
        加载指定页码的文本内容并拼接

        :param doc_id: 文档 ID
        :param pages: 页面编号列表
        :return: 拼接后的文本字符串
        """
        full_text = ""
        for page in pages:
            filename = f"{doc_id.rsplit('.',1)[0]}_{page}.txt"
            file_path = os.path.join(self.text_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到文本文件: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                full_text += f.read() + "\n\n"
        return full_text.strip()

    def _get_image_paths(self, doc_id: str, pages: List[int]) -> List[str]:
        """
        获取指定页码的图像路径列表

        :param doc_id: 文档 ID
        :param pages: 页面编号列表
        :return: 图像路径列表
        """
        paths = []
        for page in pages:
            filename = f"{doc_id.rsplit('.',1)[0]}_{page-1}.png"
            file_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到图像文件: {file_path}")
            paths.append(file_path)
        return paths
