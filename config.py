# config.py
"""
Centralized configuration, constants, and data classes for the project.
"""
import uuid
from dataclasses import dataclass, field
from typing import Dict, List
import os 

# ===================================================================
# Global Configuration
# ===================================================================
GLOBAL_MAX_TOKENS = int(os.getenv('GLOBAL_MAX_TOKENS', '8000'))
GLOBAL_THRESHOLD = float(os.getenv('GLOBAL_THRESHOLD', '1.0'))
GLOBAL_INITIAL_CHUNK_SIZE = int(os.getenv('GLOBAL_INITIAL_CHUNK_SIZE', '16'))
GLOBAL_LLM_INTER_CHUNK_SIZE = int(os.getenv('GLOBAL_LLM_INTER_CHUNK_SIZE', '10'))
GLOBAL_LAG = int(os.getenv('GLOBAL_LAG', '5'))
DE_CASCADE_BATCH_SIZE = 150

TARGET_LLM_TOKEN_RATIO = float(os.getenv('TARGET_LLM_TOKEN_RATIO', '0.3'))

# ===================================================================
# Data Class Definitions
# ===================================================================

@dataclass
class ModelConfig:
    """Configuration for LLM and SLM models"""
    llm_path: str = "/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-7B"
    slm_path: str = "/private/zhenningshi/model_weights/DeepSeek-R1-Distill-Qwen-1.5B"
    llm_gpu_ids: List[int] = None
    slm_gpu_id: int = 0
    llm_tensor_parallel_size: int = 1
    slm_tensor_parallel_size: int = 1
    llm_gpu_memory_utilization: float = 0.71
    slm_gpu_memory_utilization: float = 0.20
    temperature: float = 0.7
    def __post_init__(self):
        if self.llm_gpu_ids is None:
            self.llm_gpu_ids = [0]

@dataclass
class InferenceState:
    """
    Holds the state of each concurrent inference job for DE-Cascade.
    Uses token_ids for performance instead of string context.
    """
    request_id: str
    problem: Dict
    prompt: str
    token_ids: List[int] = field(default_factory=list)
    full_generation: str = ""
    is_finished: bool = False
    current_model: str = 'llm'
    initial_chunk_generated: bool = False
    # Note: peak_detector is initialized in the runner to avoid circular dependency
    peak_detector: 'OnlineEntropyPeakDetector' = None 
    metrics: Dict = field(default_factory=lambda: {
        'total_tokens': 0, 'llm_tokens': 0, 'slm_tokens': 0,
        'llm_interventions': 0, 'entropy_peaks': 0, 'wall_time': 0,
        'entropy_history': []
    })
