"""Module for selection of a Recipe from the Pareto front by an LLM."""

import json
import logging as log
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl


class LLMSelector:
    """Selector for Recipes based on the choice of an LLM."""

    def select_recipes(
        self,
        pareto_front: pl.DataFrame,
        prompt_instructions: str,
        ask: Callable[[str], str],
        *,
        n_votes: int = 10,
        max_calls: int = 30,
        threshold: float = 0.5,
    ) -> Tuple[Optional[pl.DataFrame], Dict[str, Any]]:
        """Lets an LLM select a recipe from the Pareto front. The LLM is asked
        until it has given "n_votes" valid answers, or "max_calls" is reached.
        If the most common answer gets at least "threshold" of the votes, that
        recipe is returned. Otherwise, None is returned.

        Parameters:
            pareto_front (pl.DataFrame): DataFrame containing the Pareto front recipes
                with a "RecipeID" column.
            prompt_instructions (str): Instructions for the LLM.
            ask (Callable[[str], str]): Function that sends a prompt to the model and
                returns its text answer.
            n_votes (int): Number of valid answers to collect.
            max_calls (int): Maximum number of times the LLM is asked, so a model
                that keeps giving invalid answers does not loop forever.
            threshold (float): Fraction of the valid votes the winning recipe needs.

        Returns:
            recipe_id_row (Optional[pl.DataFrame]): DataFrame containing the selected
                recipe, or None if there is no consensus
            stats (Dict[str, Any]): Dictionary containing the call statistics.
        """
        stats: Dict[str, Any] = {
            "n_calls": 0,
            "n_invalid": 0,
            "invalid_answers": [],
            "invalid_ids": [],
            "votes": [],
        }

        valid_ids = pareto_front["RecipeID"].to_list()
        prompt = self.build_prompt(prompt_instructions, pareto_front)

        while len(stats["votes"]) < n_votes and stats["n_calls"] < max_calls:
            answer = ask(prompt)
            stats["n_calls"] += 1
            vote, valid = self.parse_answer(answer, valid_ids)
            if not valid:
                stats["n_invalid"] += 1
                stats["invalid_answers"].append(answer)
                stats["invalid_ids"].append(vote)
            else:
                stats["votes"].append(vote)

        log.info(
            "LLM calls: %d, valid votes: %d, invalid answers: %d",
            stats["n_calls"],
            len(stats["votes"]),
            stats["n_invalid"],
        )
        log.info("LLM votes: %s", stats["votes"])

        if len(stats["votes"]) < n_votes:
            log.info("Reached %d calls without %d valid votes.", max_calls, n_votes)
            return None, stats

        winner = self.pick_majority(stats["votes"], threshold)
        if winner is None:
            log.info("No LLM consensus.")
            return None, stats

        log.info("LLM selected recipe %d.", winner)
        recipe_id_row = pareto_front.filter(pl.col("RecipeID") == winner)
        return recipe_id_row, stats

    def pareto_to_json(self, pareto_front: pl.DataFrame) -> str:
        """Convert the Pareto front to JSON text.

        Parameters:
            pareto_front (pl.DataFrame): DataFrame containing the Pareto front recipes.

        Returns:
            pareto_front_json (str): JSON list of recipes, each including its "RecipeID".
        """
        pareto_front_json = json.dumps(pareto_front.to_dicts(), indent=2)
        return pareto_front_json

    def build_prompt(self, prompt_instructions: str, pareto_front: pl.DataFrame) -> str:
        """Builds the prompt to be passed into the LLM. Combines prompt instructions
        with Pareto front points converted into JSON text.

        Parameters:
            prompt_instructions (str): Instructions for the LLM taken from the prompt
                config file.
            pareto_front (pl.DataFrame): DataFrame containing the Pareto front recipes.

        Returns:
            prompt (str): Built prompt to be passed into the LLM.
        """
        prompt = prompt_instructions + "\n\n" + self.pareto_to_json(pareto_front)
        return prompt

    def parse_answer(
        self, answer: str, valid_ids: List[int]
    ) -> Tuple[Optional[int], bool]:
        """Extract the recipe ID from an answer and check it is on the Pareto front.

        Parameters:
            answer (str): Raw text answer, expected as ``{"recipe_id": <int>}``.
            valid_ids (List[int]): Recipe IDs on the Pareto front.

        Returns:
            recipe_id (Optional[int]): The recipe ID, or None if the answer is malformed.
            valid (bool): Whether the answer is valid.
        """
        valid = True

        try:
            recipe_id = int(json.loads(answer)["recipe_id"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            valid = False
            log.warning("Could not parse LLM answer: %s", answer)
            return None, valid
        if recipe_id not in valid_ids:
            valid = False
            log.warning(
                "LLM picked recipe %d, which is not on the Pareto front.", recipe_id
            )
            return recipe_id, valid
        return recipe_id, valid

    def pick_majority(self, votes: List[int], threshold: float) -> Optional[int]:
        """Return the most common recipe ID if it got enough of the votes.

        Parameters:
            votes (List[int]): Valid recipe IDs picked by the LLM.
            threshold (float): Fraction of the votes the winner needs.

        Returns:
            winner (Optional[int]): The winning recipe ID, or None if there is no
                consensus.
        """
        counts = Counter(votes)
        if not counts:
            return None
        winner, n_votes = counts.most_common(1)[0]
        if n_votes / len(votes) < threshold:
            return None
        return winner
