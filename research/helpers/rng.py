import json
import random
from typing import Callable, List, Optional

import polars as pl


def make_random_ask(
    pareto_front: pl.DataFrame,
    column_labels: List[str],
    seed: Optional[int] = None,
    p_invalid: float = 0.0,
) -> Callable[[str], str]:
    """Create a stand-in for an LLM that answers with a random recipe.

    The answer is returned as JSON text, the same format the real LLM is asked
    to use, so the parsing and validation steps are exercised.

    Parameters:
        pareto_front (pl.DataFrame): DataFrame containing the Pareto front recipes.
        column_labels (List[str]): Column labels from Investigation Space.
        seed (Optional[int]): Seed for the random number generator.
        p_invalid (float): Chance of answering with an ID that is not on the
            Pareto front, to mimic a hallucinating LLM.

    Returns:
        Callable[[str], str]: Function that ignores the prompt and returns
            a random answer.
    """
    rng = random.Random(seed)

    recipes = pareto_front.unique(
        subset=column_labels, keep="first", maintain_order=True
    ).to_dicts()

    def ask(prompt: str) -> str:  # pylint: disable=unused-argument
        recipe = rng.choice(recipes)
        recipe_id = recipe["RecipeID"]
        if rng.random() < p_invalid:
            recipe_id = pareto_front["RecipeID"].max() + 1
        return json.dumps(
            {
                "reasoning": "Random choice.",
                "recipe_id": recipe_id,
                "candidate_points": {col: recipe[col] for col in column_labels},
            }
        )

    return ask
