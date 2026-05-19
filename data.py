# data.py
"""
Handles dataset loading (GSM8K, MATH) and prompt construction.
"""
import re
from datasets import load_dataset
from typing import Dict, List, Optional

from utils import extract_answer

def build_math_prompt(problem: str) -> str:
    """Creates a standardized instruction prompt for math problems."""
    return (
        "Solve the following math problem step-by-step. "
        "The final answer must be enclosed in a single \\boxed{} block. "
        f"Question: {problem}\n\n"
    )


    
class DatasetManager:
    """Manages loading and preprocessing of GSM8K and MATH datasets."""
    @staticmethod
    def load_GSM8K(num_samples: Optional[int] = None) -> List[Dict]:
        print("Loading GSM8K dataset...")
        dataset = load_dataset("/mnt/8T/xgr/shizhenning/datasets/openai/gsm8k", "main", split="test")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        problems = []
        for item in dataset:
            # Extract final number from the solution string
            answer = re.findall(r'####\s*([0-9,]+)', item['answer'])
            answer = answer[0].replace(',', '') if answer else ""
            problems.append({'question': item['question'], 'answer': answer})
        return problems
    
    @staticmethod
    def load_MATH(num_samples: Optional[int] = None) -> List[Dict]:
        print("Loading MATH dataset...")
        dataset = load_dataset("/mnt/8T/xgr/shizhenning/datasets/HuggingFaceH4/MATH-500", split="test")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        problems = []
        for item in dataset:
            problems.append({
                'question': item['problem'], 
                'answer': extract_answer(item['solution']) or item['solution']
            })
        # print(problems[0])
        # exit()
        # {'question': 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$', 'answer': '(3,\\frac{\\pi}{2})'}
        return problems
    
    @staticmethod
    def load_AIME(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the AIME dataset."""
        print("Loading AIME dataset...")
        # 注意：你需要确认 AIME 数据集的具体子集和 split 名称，这里使用 "test" 作为示例
        dataset = load_dataset("/mnt/8T/xgr/zhuyijia/huggingface_datasets/AIME_2024", split="train")
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        problems = []
        for item in dataset:
            # 假设 AIME 数据集的字段是 'question' 和 'answer'
            # 你可能需要根据实际情况调整字段名和答案提取逻辑
            problems.append({
                'question': item['Problem'],
                'answer': extract_answer(str(item['Answer'])) or str(item['Answer'])
            })
        return problems
    

    @staticmethod
    def load_ARC_Challenge(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the ARC-Challenge dataset."""
        print("Loading ARC-Challenge dataset...")
        # 默认使用 'challenge' 子集和 'test' 分割，你可能需要根据实际路径和需求调整
        dataset = load_dataset(
            "/mnt/8T/xgr/shizhenning/datasets/allenai/ai2_arc", 
            "ARC-Challenge", 
            split="train"
        )
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        problems = []
        for item in dataset:
            question = item['question']
            choices_text = item['choices']['text']
            choices_labels = item['choices']['label']
            answer_key = item['answerKey']

            # 1. 将问题和选项整合成一个字符串，作为 build_math_prompt 的输入 (problem)
            options_str = "\n".join([
                f"{lbl}. {txt}"
                for lbl, txt in zip(choices_labels, choices_text)
            ])

            prompt_input = (
                f"{question}\n\n"
                f"Options:\n{options_str}\n\n"
                "Select the correct option (A, B, C, D or E). "
                "Your final answer **must be only the letter** of the correct option (e.g., A) "
                "enclosed in the \\boxed{} block, with no additional text or explanation."
            )
            
            problems.append({
                'question': prompt_input, 
                'answer': answer_key
            })
        # print(problems[0])
        # exit()
        return problems

    @staticmethod
    def load_CommonsenseQA(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the CommonsenseQA dataset."""
        print("Loading CommonsenseQA dataset...")
        # 默认使用 'challenge' 子集和 'test' 分割，你可能需要根据实际路径和需求调整
        dataset = load_dataset(
            "/mnt/8T/xgr/shizhenning/datasets/tau/commonsense_qa/data",
            split="train"
        )
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        problems = []
        for item in dataset:
            question = item['question']
            choices_text = item['choices']['text']
            # choices_labels = item['choices']['label']
            answer_key = item['answerKey']

            # 1. 将问题和选项整合成一个字符串，作为 build_math_prompt 的输入 (problem)
            # options_str = "\n".join([
            #     f"{lbl}. {txt}"
            #     for lbl, txt in zip(choices_labels, choices_text)
            # ])
            options_str = "\n".join([
                f"{chr(65+i)}. {opt}" for i, opt in enumerate(choices_text)
            ])

            prompt_input = (
                f"{question}\n\n"
                f"Options:\n{options_str}\n\n"
                "Select the correct option (A, B, C, D or E). "
                "Your final answer **must be only the letter** of the correct option (e.g., A) "
                "enclosed in the \\boxed{} block, with no additional text or explanation."
            )
            
            problems.append({
                'question': prompt_input, 
                'answer': answer_key
            })
        # print(problems[0])
        # exit()
        return problems