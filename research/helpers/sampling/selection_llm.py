"""Wrapper function for LLM selector"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl

from paretodo import LLMSelector, ask_server
from helpers.rng import make_random_ask


def get_ask(
    llm_config: Dict[str, Any],
    pareto_df: pl.DataFrame,
    column_labels: List[str],
    seed: int,
) -> Callable[[str], str]:
    """
    Wrapper function for LLM client.

    Parameters:
        llm_config (Dict[str, Any]): LLM config dictionary.
        pareto_df (pl.DataFrame): Pareto DataFrame.
        column_labels (List[str]): Column labels.
        seed (int): Random seed.

    Returns:
        Callable[[str], str]: Function that returns the model's text answer.
    """
    provider = llm_config["provider"]

    if provider == "server":
        ask = ask_server(
            model=llm_config["model"],
            base_url=llm_config["base_url"],
            options=llm_config.get("options"),
        )
        return ask

    if provider == "random":
        ask_random = make_random_ask(
            pareto_front=pareto_df,
            column_labels=column_labels,
            seed=seed,
            p_invalid=llm_config.get("p_invalid", 0.0),
        )
        return ask_random
    raise ValueError(f"Unsupported LLM provider: {provider}")


def select_llm_recipes(
    pareto_df: pl.DataFrame,
    prompt_instructions: str,
    column_labels: List[str],
    llm_config: Dict[str, Any],
    seed: int,
) -> Tuple[Optional[pl.DataFrame], Optional[Dict[str, Any]]]:
    """
    Wrapper function for LLM selector. Checks if Pareto front is empty. Unpacks LLM
    config.

    Parameters:
        pareto_df (pl.DataFrame): Pareto DataFrame
        prompt_instructions (str): Instructions for the LLM.
        column_labels (List[str]): Column labels from Investigation Space.
        llm_config (Dict[str, Any]): LLM config dictionary.
        seed (int): Random seed.

    Returns:
        selected_point (Optional[pl.DataFrame]): DataFrame containing the selected
            recipe, or None if there is no consensus
        stats (Dict[str, Any]): Dictionary containing the call statistics.
    """
    if pareto_df.height == 0:
        return pl.DataFrame(), None

    ask = get_ask(
        llm_config=llm_config,
        pareto_df=pareto_df,
        column_labels=column_labels,
        seed=seed,
    )

    selector = LLMSelector()

    selected_point, stats = selector.select_recipes(
        pareto_front=pareto_df,
        prompt_instructions=prompt_instructions,
        column_labels=column_labels,
        ask=ask,
        n_votes=llm_config.get("n_votes", 10),
        max_calls=llm_config.get("max_calls", 30),
        threshold=llm_config.get("threshold", 0.5),
    )

    return selected_point, stats
