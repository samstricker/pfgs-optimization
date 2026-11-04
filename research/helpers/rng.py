import json
import random
from typing import Callable, List, Optional


def make_random_ask(
    valid_ids: List[int], seed: Optional[int] = None, p_invalid: float = 0.0
) -> Callable[[str], str]:
    """Create a stand-in for an LLM that answers with a random recipe ID.

    The answer is returned as JSON text, the same format the real LLM is asked
    to use, so the parsing and validation steps are exercised.

    Parameters:
        valid_ids (List[int]): Recipe IDs to choose from.
        seed (Optional[int]): Seed for the random number generator.
        p_invalid (float): Chance of answering with an ID that is not on the
            Pareto front, to mimic a hallucinating LLM.

    Returns:
        Callable[[str], str]: Function that ignores the prompt and returns
            a random answer.
    """
    rng = random.Random(seed)

    def ask(prompt: str) -> str:  # pylint: disable=unused-argument
        if rng.random() < p_invalid:
            return json.dumps({"recipe_id": max(valid_ids) + 1})
        return json.dumps({"recipe_id": rng.choice(valid_ids)})

    return ask
