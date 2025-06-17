import os
import json
from time import sleep
from openai import OpenAI
from tqdm import tqdm
import ast

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-0808b7683ffa4b6ab10dbb421f5b55df",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

with open("result.json", "r", encoding="utf-8") as f:
    result = json.load(f)
    
result_copy = result.copy()
    
for item in tqdm(result_copy):
    question = item["original_question"]
    answer = item["final_answer"]
    gt = item["ground_truth"]

    prompt = f"""
    Question: {question}
    Predicted Answer: {answer}
    Ground Truth Answer: {gt}
    
    Please evaluate if the predicted answer is correct compared to the ground truth.
    Score the answer on:
    Binary correctness (0-1): 1 if the answer is correct, 0 if it is incorrect

    """ 
    prompt+= "Return only a string with these scores in a dictionary and can be parsed by json.loads (Return only a string with these scores in a dictionary and can be parsed by json.loads (Quotation marks must be in double quotes, and single quotes can only be used for quotes within double quotes)), e.g. {\"binary_correctness\": 1}"
    
    completion = client.chat.completions.create(
        model="qwen-plus-0112",
        messages=
        [
            {"role": "user", "content": f"{prompt}"},
        ]
    )
    tmp=json.loads(completion.choices[0].message.content)
    item["binary_correctness"] = tmp["binary_correctness"]
    sleep(1)
    
with open("result_eval.json", "w", encoding="utf-8") as f:
    json.dump(result_copy, f, ensure_ascii=False, indent=4)


with open("result_eval.json", "r", encoding="utf-8") as f:
    result = json.load(f)
    
corr=0
    
for item in result:
    if item["binary_correctness"]==1:
        corr+=1
print(corr/len(result))
    