from abc import ABC, abstractmethod
from automatic_prompt_engineer import llm
import itertools

from automatic_prompt_engineer import llm, data, template
import numpy as np

import asyncio
import nest_asyncio

nest_asyncio.apply()
    
class Evaluator(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def rate_prompts(self, task, prompts, inputs):
        """
        For each prompt, rate the prompts with llm with given inputs.
        Parameters:
            prompts: A list of prompts.
            inputs: used to generate output to evaluate with prompts.
        Returns:
            rated prompts.
        """
        pass

    
class EvaluationResult:
    
    @abstractmethod
    def top_n(self, n:int):
        """Get the results in the form of a sorted prompt and score list.
        Has a method argument to support sorting by various metrics."""
        pass
    
    def sorted(self):
        """Get the results in the form of a sorted prompt and score list.
        Has a method argument to support sorting by various metrics."""
        pass
