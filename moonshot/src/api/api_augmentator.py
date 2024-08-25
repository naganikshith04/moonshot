from moonshot.src.augmentor.augmenter import Augmenter


def api_augment_recipe(recipe_id: str, attack_module: str) -> str:
    return Augmenter.augment_recipe(recipe_id, attack_module)


def api_augment_dataset(dataset_id: str, attack_module: str) -> str:
    return Augmenter.augment_dataset(dataset_id, attack_module)
