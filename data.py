# data.py
"""
Handles dataset loading and prompt construction.
"""
import os
import re
from typing import Dict, List, Optional

from datasets import load_dataset

from utils import extract_answer


DATASET_ROOT = os.getenv(
    "COTROUTER_DATASET_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface_datasets"),
)


def dataset_path(name: str) -> str:
    return os.path.join(DATASET_ROOT, name)


def select_samples(dataset, num_samples: Optional[int] = None):
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset


def build_math_prompt(problem: str) -> str:
    """Creates a standardized instruction prompt for math problems."""
    return (
        "Solve the following math problem step-by-step. "
        "The final answer must be enclosed in a single \\boxed{} block. "
        f"Question: {problem}\n\n"
    )


def build_multiple_choice_prompt(
    question: str,
    choices_text: List[str],
    choices_labels: List[str],
) -> str:
    options_str = "\n".join(
        f"{label}. {text}" for label, text in zip(choices_labels, choices_text)
    )
    labels = ", ".join(choices_labels)
    return (
        "Answer the following multiple-choice question. Think step by step, "
        "then put only the option letter in a single \\boxed{} block.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Final answer choices: {labels}."
    )


def build_humaneval_prompt(problem_prompt: str) -> str:
    return (
        "Complete the following Python function. Return only valid Python code "
        "that completes the function; do not include markdown fences.\n\n"
        f"{problem_prompt}"
    )


def build_prompt(problem: Dict) -> str:
    """Builds the model prompt for a normalized problem record."""
    if "prompt" in problem:
        return problem["prompt"]
    return build_math_prompt(problem["question"])


def normalize_choice_record(item: Dict, question_key: str) -> Dict:
    raw_labels = [str(label) for label in item["choices"]["label"]]
    choices_text = list(item["choices"]["text"])
    normalized_labels = [
        label.upper() if label.upper() in ["A", "B", "C", "D", "E"] else chr(65 + i)
        for i, label in enumerate(raw_labels)
    ]

    raw_answer = str(item.get("answerKey", "")).strip()
    answer = raw_answer.upper()
    if answer not in normalized_labels:
        for raw_label, normalized_label in zip(raw_labels, normalized_labels):
            if raw_answer == raw_label:
                answer = normalized_label
                break
        else:
            if raw_answer.isdigit():
                answer_index = int(raw_answer) - 1
                if 0 <= answer_index < len(normalized_labels):
                    answer = normalized_labels[answer_index]

    question = item[question_key]
    return {
        "question": question,
        "prompt": build_multiple_choice_prompt(question, choices_text, normalized_labels),
        "answer": answer,
        "task_type": "multiple_choice",
        "choices": {
            "label": normalized_labels,
            "text": choices_text,
        },
        "id": item.get("id", ""),
    }


class DatasetManager:
    """Manages loading and preprocessing of supported datasets."""

    @staticmethod
    def load_GSM8K(num_samples: Optional[int] = None) -> List[Dict]:
        print("Loading GSM8K dataset...")
        dataset = load_dataset(dataset_path("gsm8k"), "main", split="test")
        dataset = select_samples(dataset, num_samples)

        problems = []
        for item in dataset:
            answer = re.findall(r"####\s*([0-9,]+)", item["answer"])
            answer = answer[0].replace(",", "") if answer else ""
            problems.append({"question": item["question"], "answer": answer})
        return problems

    @staticmethod
    def load_MATH(num_samples: Optional[int] = None) -> List[Dict]:
        print("Loading MATH dataset...")
        dataset = load_dataset(dataset_path("MATH-500"), split="test")
        dataset = select_samples(dataset, num_samples)

        problems = []
        for item in dataset:
            problems.append({
                "question": item["problem"],
                "answer": item.get("answer") or extract_answer(item["solution"]) or item["solution"],
            })
        return problems

    @staticmethod
    def load_AIME(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the AIME 2024 dataset."""
        print("Loading AIME dataset...")
        dataset = load_dataset(dataset_path("AIME_2024"), split="train")
        dataset = select_samples(dataset, num_samples)

        problems = []
        for item in dataset:
            problems.append({
                "question": item["Problem"],
                "answer": extract_answer(str(item["Answer"])) or str(item["Answer"]),
            })
        return problems

    @staticmethod
    def load_ARC_Challenge(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the ARC-Challenge test split."""
        print("Loading ARC-Challenge dataset...")
        dataset = load_dataset(
            dataset_path("ai2_arc"),
            "ARC-Challenge",
            split="test",
        )
        dataset = select_samples(dataset, num_samples)

        return [normalize_choice_record(item, "question") for item in dataset]

    @staticmethod
    def load_CommonsenseQA(
        num_samples: Optional[int] = None,
        split: str = "validation",
    ) -> List[Dict]:
        """Loads CommonsenseQA. The public test split has blank answer keys."""
        print("Loading CommonsenseQA dataset...")
        dataset = load_dataset(dataset_path("commonsense_qa"), split=split)
        dataset = select_samples(dataset, num_samples)

        return [normalize_choice_record(item, "question") for item in dataset]

    @staticmethod
    def load_OpenBookQA(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the OpenBookQA main test split."""
        print("Loading OpenBookQA dataset...")
        dataset = load_dataset(dataset_path("openbookqa"), "main", split="test")
        dataset = select_samples(dataset, num_samples)

        return [normalize_choice_record(item, "question_stem") for item in dataset]

    @staticmethod
    def load_HumanEval(num_samples: Optional[int] = None) -> List[Dict]:
        """Loads the OpenAI HumanEval test split."""
        print("Loading HumanEval dataset...")
        dataset = load_dataset(
            dataset_path("openai_humaneval"),
            "openai_humaneval",
            split="test",
        )
        dataset = select_samples(dataset, num_samples)

        problems = []
        for item in dataset:
            problems.append({
                "question": item["prompt"],
                "prompt": build_humaneval_prompt(item["prompt"]),
                "answer": item["canonical_solution"],
                "task_type": "humaneval",
                "test": item["test"],
                "entry_point": item["entry_point"],
                "task_id": item["task_id"],
            })
        return problems
