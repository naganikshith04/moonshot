from abc import abstractmethod

from moonshot.src.connectors.connector import Connector


# to take in Prompt Arguments
import importlib

# Import the module containing PromptArguments
# module_name = 'moonshot.data.runners-modules.benchmarking'
# PromptArguments = getattr(importlib.import_module(module_name), 'PromptArguments')

class Retriever:
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def retrieve_context(self, prompt):
        pass