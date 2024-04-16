from moonshot.src.redteaming.attack.attack_module import AttackModule
from moonshot.src.redteaming.attack.attack_module_arguments import AttackModuleArguments


class SampleAttackModule(AttackModule):
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
        print(self.connector_instances)
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

        print(f"target_llm_connector: {target_llm_connector}")
        toxic_llm_connector = next(
            (
                conn_inst
                for conn_inst in self.connector_instances
                if conn_inst.id == "toxicmodel"
            ),
            None,
        )

        print(f"toxic_llm_connector: {toxic_llm_connector}")


        if target_llm_connector and toxic_llm_connector:
            print("=" * 200)
            print(
                f"Preparing prompts and sending prompts to toxic generator LLM - ({toxic_llm_connector.id})"
            )
            toxic_prompt_generator = self._generate_predictions(
                self._generate_prompts(), toxic_llm_connector
            )

            toxic_prompts = [result async for result in toxic_prompt_generator]

            print("=" * 200)
            print(
                f"Sending prompts generated from Toxic LLM [{toxic_llm_connector.id}] \
                    -> Target LLM [{target_llm_connector.id}]"
            )
            print("\n")

            iteration_count = 1
            for toxic_prompt in toxic_prompts:
                print(
                    f"Sending prompt [{toxic_prompt.connector_prompt.predicted_results}] -> \
                        Target LLM [{target_llm_connector.id}]"
                )

                # send to two targets
                result = await self.send_prompt(
                    target_llm_connector,
                    toxic_prompt.connector_prompt.predicted_results,
                )

                results = await self.send_prompt(
                    target_llm_connector2,
                    toxic_prompt.connector_prompt.predicted_results,
                )

                print(
                    f'Response from Target LLM [{target_llm_connector.id}] -> prompt ["{result}"]'
                )
                print(
                    f'Response from Target LLM #2 [{target_llm_connector.id}] -> prompt ["{results}"]'
                )
                if self.check_stop_condition(
                    toxic_prompt.connector_prompt.prompt,
                    iteration_count,
                    toxic_prompt.connector_prompt.predicted_results,
                ):
                    return toxic_prompt.connector_prompt.predicted_results

                if iteration_count >= self.get_max_no_iterations():
                    print(
                        f"Stopping red teaming as max number of iterations is hit({self.get_max_no_iterations()})..."
                    )
                    return ""
                iteration_count += 1
                print("=" * 200, "\n")

            print(
                "Stopping red teaming as there is no more data in the dataset to send."
            )
