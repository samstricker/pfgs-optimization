"""Module for selection of a Recipe from the Pareto front by an LLM."""

import json
import logging as log
from collections import Counter
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
from pydantic import BaseModel, ValidationError


class LLMResponse(BaseModel):
    """Structured answer expected from the LLM."""

    reasoning: str
    recipe_id: int
    candidate_points: Dict[str, float]


class LLMSelector:
    """Selector for Recipes based on the choice of an LLM."""

    def select_recipes(
        self,
        pareto_front: pl.DataFrame,
        prompt_instructions: str,
        column_labels: List[str],
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
            column_labels (List[str]): Column labels from Investigation Space.
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
            "votes": [],
            "response": [],
            "outcome": None,
        }

        pareto_front = pareto_front.unique(
            subset=column_labels, keep="first", maintain_order=True
        )
        valid_ids = pareto_front["RecipeID"].to_list()
        prompt = self.build_prompt(prompt_instructions, pareto_front, column_labels)

        while len(stats["votes"]) < n_votes and stats["n_calls"] < max_calls:
            answer = ask(prompt)
            stats["n_calls"] += 1
            response, valid = self.parse_answer(
                answer=answer,
                valid_ids=valid_ids,
                pareto_front=pareto_front,
                column_labels=column_labels,
            )
            stats["response"].append(
                {
                    "recipe_id": None if response is None else response.recipe_id,
                    "reasoning": None if response is None else response.reasoning,
                    "valid": valid,
                    "raw_answer": answer,
                }
            )

            if response is None or not valid:
                stats["n_invalid"] += 1
            else:
                stats["votes"].append(response.recipe_id)

        log.info(
            "LLM calls: %d, valid votes: %d, invalid answers: %d",
            stats["n_calls"],
            len(stats["votes"]),
            stats["n_invalid"],
        )
        log.info("LLM votes: %s", stats["votes"])

        if len(stats["votes"]) < n_votes:
            log.info("Reached %d calls without %d valid votes.", max_calls, n_votes)
            stats["outcome"] = "not_enough_votes"
            return None, stats

        winner = self.pick_majority(stats["votes"], threshold)
        if winner is None:
            log.info("No LLM consensus.")
            stats["outcome"] = "no_majority"
            return None, stats

        log.info("LLM selected recipe %d.", winner)
        stats["outcome"] = "selected"
        recipe_id_row = pareto_front.filter(pl.col("RecipeID") == winner)
        return recipe_id_row, stats

    def pareto_to_json(self, pareto_front: pl.DataFrame) -> str:
        """Convert the Pareto front to JSON text.

        Parameters:
            pareto_front (pl.DataFrame): DataFrame containing the Pareto front recipes.

        Returns:
            pareto_front_json (str): JSON list of recipes, each including its "RecipeID".
        """
        pareto_front_json = json.dumps(
            pareto_front.to_dicts(), indent=2, ensure_ascii=False
        )
        return pareto_front_json

    def build_prompt(
        self,
        prompt_instructions: str,
        pareto_front: pl.DataFrame,
        column_labels: List[str],
        decimals: int = 2,
    ) -> str:
        """Builds the prompt to be passed into the LLM. Combines prompt instructions
        with the Pareto front converted into JSON text. Only the RecipeID, the input
        columns and the rounded objective columns are passed to the LLM.

        Parameters:
            prompt_instructions (str): Instructions for the LLM taken from the prompt
                config file.
            pareto_front (pl.DataFrame): DataFrame containing the Pareto front recipes.
            column_labels (List[str]): Column labels from Investigation Space.
            decimals (int): Number of decimals the objective values are rounded to.

        Returns:
            prompt (str): Built prompt to be passed into the LLM.
        """
        objective_cols = [col for col in pareto_front.columns if col.startswith("obj")]
        prompt_front = pareto_front.select(
            ["RecipeID", *column_labels, *objective_cols]
        ).with_columns(pl.col(objective_cols).round(decimals))
        prompt = prompt_instructions + "\n\n" + self.pareto_to_json(prompt_front)
        return prompt

    def parse_answer(
        self,
        answer: str,
        valid_ids: List[int],
        pareto_front: pl.DataFrame,
        column_labels: List[str],
    ) -> Tuple[Optional[LLMResponse], bool]:
        """Parse an answer and check it matches a recipe on the Pareto front.

        Parameters:
            answer (str): Raw text answer, expected to match ``LLMResponse``.
            valid_ids (List[int]): Recipe IDs on the Pareto front.
            pareto_front (pl.DataFrame): DataFrame containing the Pareto front recipes.
            column_labels (List[str]): Column labels from Investigation Space.

        Returns:
            response (Optional[LLMResponse]): The parsed answer, or None if the
                answer is malformed.
            valid (bool): Whether the answer is valid.
        """
        valid = True

        try:
            response = LLMResponse.model_validate_json(answer)
        except ValidationError:
            valid = False
            log.warning("Could not parse LLM answer: %s", answer)
            return None, valid

        if response.recipe_id not in valid_ids:
            valid = False
            log.warning(
                "LLM picked recipe %d, which is not on the Pareto front.",
                response.recipe_id,
            )
            return response, valid

        if set(response.candidate_points) != set(column_labels):
            valid = False
            log.warning(
                "Candidate point labels %s do not match investigation space labels.",
                list(response.candidate_points),
            )
            return response, valid

        row = pareto_front.filter(pl.col("RecipeID") == response.recipe_id).row(
            0, named=True
        )
        for col in column_labels:
            if not math.isclose(response.candidate_points[col], row[col], abs_tol=1e-6):
                valid = False
                log.warning(
                    "LLM value for %s does not match recipe %d.",
                    col,
                    response.recipe_id,
                )
                return response, valid

        return response, valid

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
