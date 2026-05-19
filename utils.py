# utils.py
"""
Utility functions for entropy calculation, answer string processing,
mathematical validation, and CUDA device management.
"""
import os
import re
import subprocess
import sys
import tempfile
import numpy as np
from contextlib import contextmanager
from collections import deque
from typing import Dict, List, Optional
from math_parsing_util import strip_answer_string, math_equal, choice_answer_clean

# ===================================================================
# CUDA Environment Utility
# ===================================================================

@contextmanager
def set_cuda_devices(device_ids: List[int]):
    """Context manager to temporarily set CUDA_VISIBLE_DEVICES."""
    old_device = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
    try:
        yield
    finally:
        if old_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_device
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

# ===================================================================
# DE-Cascade Algorithm Utilities
# ===================================================================

class OnlineEntropyPeakDetector:
    """Online Z-Score based peak detection for entropy monitoring."""
    def __init__(self, lag: int, threshold: float, influence: float = 0.2):
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.history = deque(maxlen=lag * 2)
        self.filtered_y = deque(maxlen=lag)
        
    def add_datapoint(self, value: float) -> str:
        self.history.append(value)
        if len(self.history) < self.lag:
            self.filtered_y.append(value)
            return "NORMAL"
        
        window = list(self.history)[-self.lag:]
        mean = np.mean(window)
        std_dev = np.std(window)
        
        if std_dev == 0: return "NORMAL"
        
        z_score = abs((value - mean) / std_dev)
        
        if z_score > self.threshold:
            self.filtered_y.append(self.influence * value + (1 - self.influence) * self.filtered_y[-1])
            return "PEAK"
        else:
            self.filtered_y.append(value)
            return "NORMAL"

def calculate_shannon_entropy(logprobs: Dict[int, float], vocab_size: int) -> float:
    """Calculate Shannon entropy from a logprobs dictionary."""
    probs = np.zeros(vocab_size)
    for token_id, logprob_obj in logprobs.items():
        # --- ADD THIS CHECK ---
        # If the token_id is outside the vocabulary of the target model, skip it.
        if token_id >= vocab_size:
            continue
        logprob_val = logprob_obj.logprob 
        probs[token_id] = np.exp(logprob_val)
    
    probs_sum = probs.sum()
    if probs_sum > 1e-10:
        probs = probs / probs_sum
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

# ===================================================================
# Answer Parsing and Validation Utilities (Updated)
# ===================================================================

def extract_answer(pred_str: str, use_last_number: bool = True) -> Optional[str]:

    """Parses out the final expression or numeric value from a typical LLM chain-of-thought."""
    pred_str = pred_str.replace("\u043a\u0438", "")
    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        # minerva_math
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "答案是" in pred_str:
        # Handle Chinese few-shot multiple choice problem answer extraction
        pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    elif "**Final Answer:**" in pred_str or "Final Answer:" in pred_str:
        if "**Final Answer:**" in pred_str:
            answer_text = pred_str.split("**Final Answer:**")[1]
        else:
            answer_text = pred_str.split("Final Answer:")[1]
            
        for marker in ["</think>", "\n\n"]:
            if marker in answer_text:
                answer_text = answer_text.split(marker)[0]
            
        pattern = r"\$?(\d+(?:\.\d+)?)R"
        numbers = re.findall(pattern, answer_text.replace(",", ""))
        if numbers:
            pred = numbers[0]
        else:
            words = [w for w in answer_text.split() if w]
            pred = words[0] if words else ""
    elif "Answer:" in pred_str:
        # 提取 "Answer:" 之后的内容
        answer_text = pred_str.split("Answer:", 1)[1]
        
        # 使用正则表达式查找第一个非空格的大写字母 (A-E)
        # r"^\s*([A-E])" 匹配字符串开头（^），可选的空格 (\s*)，然后捕获一个大写字母 ([A-E])
        match = re.search(r"^\s*([A-E])", answer_text)
        
        if match:
            # 如果找到匹配的选项字母，则使用它
            pred = match.group(1)
        else:
            # 否则，退回到原始逻辑：提取 "Answer:" 后的所有内容并清理
            pred = answer_text.strip()
            pred = pred.split("\n")[0].strip()
            # 确保 pred 在传递给最终清理前不会是空字符串
            if not pred:
                pred = ""
    else:  # use the last number

        if use_last_number:
            
            cleaned_str = pred_str.strip()
            # 检查清理后字符串的最后一个字符
            if cleaned_str and cleaned_str[-1].upper() in 'ABCDE':
                pred = cleaned_str[-1].upper()
                return pred

            pattern = r"-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
        else:
            pred = ""

    # multiple line
    # pred = pred.split("\n")[0]
    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_answer_string(pred)

    return pred


def is_correct_answer(model_answer_str: Optional[str], ground_truth_str: str) -> bool:
    
    cleaned_ans = strip_answer_string(ground_truth_str)
    cleaned_pred = strip_answer_string(model_answer_str)
    # Check correctness
    correct = math_equal(cleaned_pred, cleaned_ans)
    return correct


def extract_choice_answer(pred_str: str) -> Optional[str]:
    """Extracts a multiple-choice option letter from model output."""
    if pred_str is None:
        return None
    answer = extract_answer(pred_str)
    if answer:
        return choice_answer_clean(answer)
    return choice_answer_clean(pred_str)


def extract_python_completion(pred_str: str) -> str:
    """Extracts the most likely Python completion from model output."""
    if pred_str is None:
        return ""

    text = str(pred_str)
    if "</think>" in text:
        text = text.split("</think>")[-1]

    fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_blocks:
        text = fenced_blocks[-1]

    return text.strip("\n")


def run_humaneval_tests(response_text: str, problem: Dict) -> bool:
    """Executes HumanEval tests in a subprocess and returns pass/fail."""
    completion = extract_python_completion(response_text)
    entry_point = problem["entry_point"]
    prompt = problem["question"]
    prompt_prefix = prompt.split(f"def {entry_point}", 1)[0]
    has_full_function = re.search(
        rf"(^|\n)\s*def\s+{re.escape(entry_point)}\s*\(",
        completion,
    )
    if has_full_function:
        candidate_program = f"{prompt_prefix}{completion}\n"
    else:
        candidate_program = f"{prompt}{completion}\n"
    test_program = f"{candidate_program}\n{problem['test']}\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(test_program)
        tmp_path = tmp_file.name

    try:
        completed = subprocess.run(
            [sys.executable, "-I", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def evaluate_problem_answer(response_text: str, problem: Dict):
    """Returns (predicted_answer, is_correct) for the problem type."""
    task_type = problem.get("task_type", "math")

    if task_type == "humaneval":
        predicted_answer = extract_python_completion(response_text)
        is_correct = run_humaneval_tests(response_text, problem)
        return predicted_answer, is_correct

    if task_type == "multiple_choice":
        predicted_answer = extract_choice_answer(response_text)
        is_correct = is_correct_answer(predicted_answer, problem["answer"])
        return predicted_answer, is_correct

    predicted_answer = extract_answer(response_text)
    is_correct = is_correct_answer(predicted_answer, problem["answer"])
    return predicted_answer, is_correct
