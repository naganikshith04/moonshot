# from moonshot.api import api_read_recipe,api_create_recipe,api_create_datasets,api_read_dataset
import asyncio

from moonshot.src.api.api_runner import api_load_runner

# from moonshot.src.datasets.dataset import Dataset
from moonshot.src.recipes.recipe import Recipe
from moonshot.src.recipes.recipe_arguments import RecipeArguments


class Augmenter:
    @staticmethod
    def augment_recipe(recipe_id: str, attack_module: str) -> str:
        """
        User will select 1 recipe and 1 attack module

        What to do:
        Step 1. Get recipe information
        Step 2. Get all the datasets in that recipe
        Step 3. Augment the datasets
        Step 4. Write each new augmented dataset to a new file
        Step 5. Create new recipe with new set of datasets
        """
        selected_recipe = Recipe.read(recipe_id)
        datasets = selected_recipe.datasets
        augmented_datasets_id = []

        for dataset in datasets:
            augmented_datasets_id.append(
                Augmenter.augment_dataset(dataset, attack_module)
            )

        # Create recipe with new datasets
        new_rec_name = f"{recipe_id}-{attack_module}"

        try:
            rec_args = RecipeArguments(
                id="",
                name=new_rec_name,
                description=selected_recipe.description,
                tags=selected_recipe.tags,
                categories=selected_recipe.categories,
                datasets=augmented_datasets_id,
                prompt_templates=selected_recipe.prompt_templates,
                metrics=selected_recipe.metrics,
                grading_scale=selected_recipe.grading_scale,
            )
            return Recipe.create(rec_args)
        except Exception as e:
            raise e

    @staticmethod
    def augment_dataset(dataset_id: str, attack_module_id: str) -> str:
        # dataset = Dataset.read(dataset_id)
        # inputs = dataset.examples

        # simulate pipeline returning traversed generator
        dataset_prompts = [
            "This is a very long sentence!!",
            "This is a very short sentence!!",
        ]

        if dataset_prompts:
            runner_args = {
                "attack_strategies": [
                    {
                        "attack_module_id": attack_module_id,
                        "dataset_prompts": dataset_prompts,
                        "optional_params": {"test_param_key": "test_param_value"},
                    }
                ]
            }

            # Load runner and run the attack module
            runner = api_load_runner("test-range-cookbook")
            loop = asyncio.get_event_loop()
            new_prompts = loop.run_until_complete(
                runner.run_augment_dataset(runner_args)
            )
            print("augmented prommpts in ", new_prompts)

        # try:
        #     new_name = f"{dataset.id}-{attack_module_id}"
        #     new_ds_id = slugify(new_name).lower()
        #     ds_args = DatasetArguments(
        #         id="",
        #         name=new_ds_id,
        #         description=dataset.description,
        #         reference=dataset.reference,
        #         license=dataset.license,
        #         examples=new_examples,
        #     )
        #     Dataset.create(ds_args)
        #     return new_ds_id
        # except Exception as e:
        #     raise e
