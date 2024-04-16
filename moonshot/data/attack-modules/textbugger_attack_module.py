from moonshot.src.redteaming.attack.attack_module import AttackModule
from moonshot.src.redteaming.attack.attack_module_arguments import AttackModuleArguments
from moonshot.src.configs.env_variables import EnvVariables, EnvironmentVars

import json
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapQWERTY
)
from textattack.augmentation import Augmenter

delta = 0.8
num_words_swap = 0.5
num_transformations = 1

class TextBuggerAttackModule(AttackModule):
    def __init__(self, am_arguments: AttackModuleArguments):
        # Initialize super class
        super().__init__(am_arguments)

    async def execute(self):
        """
        Asynchronously executes the attack module.

        This method loads the dataset contents using the `load_dataset_contents` method,
        processes the dataset through a prompt template, retrieves the connector to the first
        Language Learning Model (LLM) and sends the processed dataset as a prompt to the LLM.
        """

        # # gets the required LLM connectors to send the prompts to
        target_llm_connector = next(
            (
                conn_inst
                for conn_inst in self.connector_instances
                if conn_inst.id == "model1"
            ),
            None,
        )

        target_llm_connector2 = next(
            (
                conn_inst
                for conn_inst in self.connector_instances
                if conn_inst.id == "weakmodel"
            ),
            None,
        )

        transformation = CompositeTransformation(
                [WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                    ),
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                    )])
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=delta))
        augmenter = Augmenter(
            transformation=transformation,
            constraints=constraints,
            pct_words_to_swap=num_words_swap,
            transformations_per_example=num_transformations)
        iteration_count = 1
        files = [f"{EnvironmentVars.get_file_directory(EnvVariables.DATASETS.name)[0]}/{file}.json" for file in self.datasets]
        for file in files:
            f = open(file , "r" , encoding = "utf-8")
            dataset = json.load(f)['examples']
            for prompt in dataset:
                results = augmenter.augment(prompt['input'])
                print("-"*20)
                print(f"Original Prompt: {prompt['input']}")
                for i in results:
                    print(f"Augmented to: {i}")
                    result = await self.send_prompt(
                        target_llm_connector,
                        i,
                    )
                    results = await self.send_prompt(
                        target_llm_connector2,
                        i,
                    )

                    print(
                        f'Response from Target LLM [{target_llm_connector.id}] -> prompt ["{i}"]'
                    )
                    print(
                        f'Response from Target LLM #2 [{target_llm_connector2.id}] -> prompt ["{i}"]'
                    )
                    if self.check_stop_condition(
                        i,
                        iteration_count,
                        result
                    ):
                        return i
                    if iteration_count >= self.get_max_no_iterations():
                        print(
                            f"Stopping red teaming as max number of iterations is hit({self.get_max_no_iterations()})..."
                        )
                        return ""
                    iteration_count += 1
            
