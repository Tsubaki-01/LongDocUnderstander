import json

import torch
from tqdm import tqdm

from agents.decompose_agent import DecomposeAgent
from agents.text_agent import TextAgent
from agents.image_agent import ImageAgent
from agents.summary_agent import SummaryAgent
from core.long_doc_understander import LongDocUnderstander
from utils.document_loader import DocumentDataset

# device = torch.device("cuda:0")
# dummy_tensor = torch.empty((2*1024, 2*1024, 1024), dtype=torch.float32, device=device)

# 初始化各 agent（假设已正确实现）
print("creating DecomposeAgent\n------------")
decompose_agent = DecomposeAgent(model_name="llama_3_1")
print("creating TextAgent\n------------")
text_agent = TextAgent(model_name="llama_3_1")
print("creating ImageAgent\n------------")
image_agent = ImageAgent(model_name="qwen_vl_2_5")
print("creating SummaryAgent\n------------")
summary_agent = SummaryAgent(model_name="llama_3_1")

# 创建理解器
understander = LongDocUnderstander(
    decompose_agent=decompose_agent,
    text_agent=text_agent,
    image_agent=image_agent,
    summary_agent=summary_agent
)

# 加载数据集
dataset = DocumentDataset(
    json_path="data/MMLongBench/dataset.json",
    text_dir="data/MMLongBench/text",
    image_dir="data/MMLongBench/image"
)
torch.cuda.empty_cache()


# with open("result.json", "r", encoding="utf-8") as f:
#     result = json.load(f)
result = []
for i in tqdm(range(100)):
    print(f"-------------------\n正在处理第{i+200+1}个样本")
    # 获取一个样本
    sample = dataset[i+200]
    # 理解文档并回答问题
    understand = understander.understand(
        original_question=sample["question"],
        document_text=sample["text"],
        document_images=sample["images"][0:4]
    )
    understand["ground_truth"] = sample["answer"]
    result.append(understand)
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)



# # 获取一个样本
# sample = dataset[3]
# # 理解文档并回答问题
# result = understander.understand(
#     original_question=sample["question"],
#     document_text=sample["text"],
#     document_images=sample["images"]
# )
# print(result)
# with open("result.json", "w", encoding="utf-8") as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)

torch.cuda.empty_cache()