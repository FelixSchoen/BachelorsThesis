import random


def util_remove_elements(elements: list, percentage_to_drop: float) -> list:
    return random.sample(elements, int(percentage_to_drop*len(elements)))
