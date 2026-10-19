"""Tests for LLMSelector."""

import json
from typing import Callable, List

import polars as pl
import pytest

from paretodo.selection.llm_selector import LLMSelector


def make_ask(answers: List[str]) -> Callable[[str], str]:
    """Create a fake LLM that returns the given answers in order and records prompts."""
    remaining = iter(answers)
    prompts: List[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(remaining)

    ask.prompts = prompts
    return ask


def answer(recipe_id: int) -> str:
    """Format a recipe ID the way the LLM is asked to answer."""
    return json.dumps({"recipe_id": recipe_id})


@pytest.fixture(name="selector")
def fixture_selector():
    """Create an LLMSelector."""
    return LLMSelector()


@pytest.fixture(name="pareto_front")
def fixture_pareto_front():
    """Create a small Pareto front with three recipes."""
    return pl.DataFrame(
        {
            "RecipeID": [0, 1, 2],
            "temperature": [30.0, 35.0, 40.0],
            "obj1": [1.0, 2.0, 3.0],
        }
    )


def test_pareto_to_json_contains_all_recipes(selector, pareto_front):
    """Every recipe and its RecipeID should be in the JSON."""
    recipes = json.loads(selector.pareto_to_json(pareto_front))

    assert [recipe["RecipeID"] for recipe in recipes] == [0, 1, 2]
    assert recipes[1]["temperature"] == 35.0


def test_build_prompt_puts_front_after_instructions(selector, pareto_front):
    """The prompt should be the instructions followed by the Pareto front."""
    prompt = selector.build_prompt("Pick a recipe.", pareto_front)

    assert prompt.startswith("Pick a recipe.\n\n")
    assert selector.pareto_to_json(pareto_front) in prompt


def test_parse_answer_valid_id(selector):
    """An ID on the Pareto front is returned and marked valid."""
    assert selector.parse_answer(answer(1), [0, 1, 2]) == (1, True)


def test_parse_answer_id_not_on_front(selector):
    """An ID not on the Pareto front is returned but marked invalid."""
    assert selector.parse_answer(answer(99), [0, 1, 2]) == (99, False)


@pytest.mark.parametrize(
    "bad_answer",
    [
        "garbage",
        "",
        '{"wrong_key": 1}',
        '{"recipe_id": "abc"}',
        "[1, 2]",
    ],
)
def test_parse_answer_malformed(selector, bad_answer):
    """An answer in the wrong format gives no ID and is marked invalid."""
    assert selector.parse_answer(bad_answer, [0, 1, 2]) == (None, False)


def test_pick_majority_returns_winner(selector):
    """The most common ID wins if it reaches the threshold."""
    assert selector.pick_majority([1, 1, 1, 2], threshold=0.5) == 1


def test_pick_majority_exactly_at_threshold(selector):
    """Reaching the threshold exactly is enough."""
    assert selector.pick_majority([1, 1, 2, 3], threshold=0.5) == 1


def test_pick_majority_below_threshold(selector):
    """No winner if the most common ID is below the threshold."""
    assert selector.pick_majority([1, 2, 3, 4], threshold=0.5) is None


def test_pick_majority_no_votes(selector):
    """No winner without votes."""
    assert selector.pick_majority([], threshold=0.5) is None


def test_select_recipes_returns_winning_row(selector, pareto_front):
    """A unanimous LLM returns the full row of the chosen recipe."""
    ask = make_ask([answer(2)] * 5)

    selected, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", ask, n_votes=5
    )

    assert selected is not None
    assert selected.to_dicts() == [{"RecipeID": 2, "temperature": 40.0, "obj1": 3.0}]
    assert stats["n_calls"] == 5
    assert stats["n_invalid"] == 0
    assert stats["votes"] == [2] * 5


def test_select_recipes_sends_prompt_with_front(selector, pareto_front):
    """The LLM receives the instructions together with the Pareto front."""
    ask = make_ask([answer(0)] * 3)

    selector.select_recipes(pareto_front, "Pick a recipe.", ask, n_votes=3)

    assert ask.prompts[0] == selector.build_prompt("Pick a recipe.", pareto_front)


def test_select_recipes_retries_invalid_answers(selector, pareto_front):
    """Invalid answers are retried until enough valid votes are collected."""
    ask = make_ask([answer(1), answer(99), answer(1), "garbage", answer(1)])

    selected, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", ask, n_votes=3
    )

    assert selected is not None
    assert selected["RecipeID"].to_list() == [1]
    assert stats["n_calls"] == 5
    assert stats["n_invalid"] == 2
    assert stats["votes"] == [1, 1, 1]
    assert stats["invalid_answers"] == [answer(99), "garbage"]
    assert 99 in stats["invalid_ids"]


def test_select_recipes_stops_at_max_calls(selector, pareto_front):
    """The LLM is not asked more than max_calls times."""
    ask = make_ask(["garbage"] * 100)

    selected, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", ask, n_votes=3, max_calls=7
    )

    assert selected is None
    assert stats["n_calls"] == 7
    assert stats["n_invalid"] == 7
    assert stats["votes"] == []


def test_select_recipes_no_consensus(selector, pareto_front):
    """Enough valid votes but no majority returns None."""
    ask = make_ask([answer(0), answer(1), answer(2), answer(0)])

    selected, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", ask, n_votes=4, threshold=0.75
    )

    assert selected is None
    assert stats["votes"] == [0, 1, 2, 0]
