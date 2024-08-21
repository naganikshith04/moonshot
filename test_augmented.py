from dotenv import dotenv_values

from moonshot.api import api_augment_dataset, api_set_environment_variables

api_set_environment_variables(dotenv_values(".env"))

# print(api_augment_recipe("norman-recipe", "charswap_attack"))

print(api_augment_dataset("kel-dataset", "charswap_attack_iterated_bm"))
