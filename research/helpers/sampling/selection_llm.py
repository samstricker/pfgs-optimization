"""Wrapper function for LLM selector"""

from typing import Any, Callable, Dict, Tuple, Union

import polars as pl

from paretodo import LLMSelector


def select_llm_recipes(
    pareto_df: pl.DataFrame,
    prompt_instructions: str,
    ask: Callable[[str], str],
    llm_config: Dict[str, Any],
) -> Tuple[pl.DataFrame, Union[None, Dict[str, Any]]]:
    """
    Wrapper function for LLM selector. Checks if Pareto front is empty. Unpacks LLM
    config.

    Parameters:
        pareto_df (pl.DataFrame): Pareto DataFrame
        prompt_instructions (str): Instructions for the LLM.
        ask (Callable[[str], str]): Function that sends a prompt to the model and
            returns its text answer.
        llm_config (Dict[str, Any]): LLM config dictionary.

    Returns:
        selected_point (Optional[pl.DataFrame]): DataFrame containing the selected
            recipe, or None if there is no consensus
        stats (Dict[str, Any]): Dictionary containing the call statistics.
    """
    if pareto_df.height == 0:
        return pl.DataFrame(), None

    selector = LLMSelector()

    selected_point, stats = selector.select_recipes(
        pareto_front=pareto_df,
        prompt_instructions=prompt_instructions,
        ask=ask,
        n_votes=llm_config.get("n_votes", 10),
        max_calls=llm_config.get("max_calls", 30),
        threshold=llm_config.get("threshold", 0.5),
    )

    return selected_point, stats
