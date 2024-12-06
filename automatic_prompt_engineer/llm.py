"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

import asyncio
import nest_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import AzureOpenAI

gpt_costs_per_thousand = {
    'davinci': 0.0200,
    'curie': 0.0020,
    'babbage': 0.0005,
    'ada': 0.0004
}

nest_asyncio.apply()


def model_from_config(config):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "AOAI":
        return AzureOpenaiLLM(config)
    else:
        raise ValueError(f"Noe implement LLM model type: {model_type}")


class LLM(ABC):
    """Abstract base class for large language models."""


    @abstractmethod
    def complete_prompt(self, config, messages):
        """Complete text from the model.
        Parameters:
            messages: The prompt to use.
        Returns:
            LLM response.
        """
        pass


class AzureOpenaiLLM(LLM):
    """Wrapper for GPT-3."""

    def __init__(self, config):
        """Initializes the model."""
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=config['api_version'],
            azure_endpoint=config['azure_openai_endpoint']
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=60))
    def complete_prompt(self, config, messages):
        """
        Generates text from the model.
        prompt is a single one.
        """
        response = self.client.chat.completions.create(
                                                    **config,
                                                    messages=messages)
        
        return response.choices
