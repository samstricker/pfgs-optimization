"""Tests for LLMSelector."""

import json
from typing import Callable, Dict, List, Optional

import polars as pl
import pytest

from paretodo.selection.llm_selector import LLMResponse, LLMSelector

COLUMN_LABELS = ["temperature", "ph_value"]
VALID_IDS = [0, 1, 2]
RECIPE_POINTS = {
    0: {"temperature": 30.0, "ph_value": 6.0},
    1: {"temperature": 35.0, "ph_value": 6.5},
    2: {"temperature": 40.0, "ph_value": 7.0},
}


def make_ask(answers: List[str]) -> Callable[[str], str]:
    """Create a fake LLM that returns the given answers in order and records prompts."""
    remaining = iter(answers)
    prompts: List[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(remaining)

    ask.prompts = prompts
    return ask


def answer(recipe_id: int, points: Optional[Dict[str, float]] = None) -> str:
    """Format an answer the way the LLM is asked to, using the recipe's real values."""
    return json.dumps(
        {
            "reasoning": "Good trade-off.",
            "recipe_id": recipe_id,
            "candidate_points": (
                RECIPE_POINTS.get(recipe_id, RECIPE_POINTS[0])
                if points is None
                else points
            ),
        }
    )


@pytest.fixture(name="selector")
def fixture_selector():
    """Create an LLMSelector."""
    return LLMSelector()


@pytest.fixture(name="pareto_front")
def fixture_pareto_front():
    """Create a small Pareto front with three recipes."""
    return pl.DataFrame(
        {
            "RecipeID": VALID_IDS,
            "temperature": [30.0, 35.0, 40.0],
            "ph_value": [6.0, 6.5, 7.0],
            "obj:max(μ(yield))": [1.0, 2.0, 3.0],
        }
    )


def test_pareto_to_json_contains_all_recipes(selector, pareto_front):
    """Every recipe and its RecipeID should be in the JSON."""
    recipes = json.loads(selector.pareto_to_json(pareto_front))

    assert [recipe["RecipeID"] for recipe in recipes] == VALID_IDS
    assert recipes[1]["temperature"] == 35.0


def test_pareto_to_json_keeps_special_characters(selector, pareto_front):
    """Column names should stay readable for the LLM, not be escaped."""
    assert "μ" in selector.pareto_to_json(pareto_front)


def test_build_prompt_puts_front_after_instructions(selector, pareto_front):
    """The prompt should be the instructions followed by the Pareto front."""
    prompt = selector.build_prompt("Pick a recipe.", pareto_front)

    assert prompt.startswith("Pick a recipe.\n\n")
    assert selector.pareto_to_json(pareto_front) in prompt


def test_parse_answer_valid(selector, pareto_front):
    """A correct answer is returned as a response and marked valid."""
    response, valid = selector.parse_answer(
        answer(1), VALID_IDS, pareto_front, COLUMN_LABELS
    )

    assert valid
    assert isinstance(response, LLMResponse)
    assert response.recipe_id == 1
    assert response.reasoning == "Good trade-off."


def test_parse_answer_ignores_key_order(selector, pareto_front):
    """The order of the candidate points does not matter."""
    points = {"ph_value": 6.5, "temperature": 35.0}

    _, valid = selector.parse_answer(
        answer(1, points), VALID_IDS, pareto_front, COLUMN_LABELS
    )

    assert valid


def test_parse_answer_id_not_on_front(selector, pareto_front):
    """An ID not on the Pareto front is returned but marked invalid."""
    response, valid = selector.parse_answer(
        answer(99), VALID_IDS, pareto_front, COLUMN_LABELS
    )

    assert not valid
    assert response.recipe_id == 99


@pytest.mark.parametrize(
    "points",
    [
        {"temperature": 35.0},
        {"temp": 35.0, "ph_value": 6.5},
        {"temperature": 35.0, "ph_value": 6.5, "purity": 1.0},
    ],
)
def test_parse_answer_wrong_labels(selector, pareto_front, points):
    """Missing, misspelled or extra labels make the answer invalid."""
    response, valid = selector.parse_answer(
        answer(1, points), VALID_IDS, pareto_front, COLUMN_LABELS
    )

    assert not valid
    assert response.recipe_id == 1


@pytest.mark.parametrize(
    "points",
    [
        {"temperature": 99.0, "ph_value": 6.5},
        {"temperature": 35.0, "ph_value": 9.9},
    ],
)
def test_parse_answer_wrong_values(selector, pareto_front, points):
    """A wrong value in any column makes the answer invalid."""
    response, valid = selector.parse_answer(
        answer(1, points), VALID_IDS, pareto_front, COLUMN_LABELS
    )

    assert not valid
    assert response.recipe_id == 1


@pytest.mark.parametrize(
    "bad_answer",
    [
        "garbage",
        "",
        "[1, 2]",
        '{"wrong_key": 1}',
        '{"recipe_id": 1}',
        '{"reasoning": "x", "recipe_id": "abc", "candidate_points": {}}',
        '{"reasoning": "x", "recipe_id": 1, "candidate_points": [35.0, 6.5]}',
        '{"reasoning": "cut off", "recipe_id": 1, "candi',
    ],
)
def test_parse_answer_malformed(selector, pareto_front, bad_answer):
    """An answer in the wrong format gives no response and is marked invalid."""
    assert selector.parse_answer(
        bad_answer, VALID_IDS, pareto_front, COLUMN_LABELS
    ) == (None, False)


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
        pareto_front, "Pick a recipe.", COLUMN_LABELS, ask, n_votes=5
    )

    assert selected is not None
    assert selected.to_dicts() == [
        {
            "RecipeID": 2,
            "temperature": 40.0,
            "ph_value": 7.0,
            "obj:max(μ(yield))": 3.0,
        }
    ]
    assert stats["n_calls"] == 5
    assert stats["n_invalid"] == 0
    assert stats["votes"] == [2] * 5


def test_select_recipes_sends_prompt_with_front(selector, pareto_front):
    """The LLM receives the instructions together with the Pareto front."""
    ask = make_ask([answer(0)] * 3)

    selector.select_recipes(
        pareto_front, "Pick a recipe.", COLUMN_LABELS, ask, n_votes=3
    )

    assert ask.prompts[0] == selector.build_prompt("Pick a recipe.", pareto_front)


def test_select_recipes_retries_invalid_answers(selector, pareto_front):
    """Invalid answers, including malformed ones, are retried until enough votes."""
    ask = make_ask([answer(1), answer(99), answer(1), "garbage", answer(1)])

    selected, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", COLUMN_LABELS, ask, n_votes=3
    )

    assert selected is not None
    assert selected["RecipeID"].to_list() == [1]
    assert stats["n_calls"] == 5
    assert stats["n_invalid"] == 2
    assert stats["votes"] == [1, 1, 1]


def test_select_recipes_records_every_response(selector, pareto_front):
    """Each call is recorded with its ID, reasoning, validity and raw answer."""
    ask = make_ask([answer(99), "garbage", answer(1)])

    _, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", COLUMN_LABELS, ask, n_votes=1
    )

    assert stats["response"] == [
        {
            "recipe_id": 99,
            "reasoning": "Good trade-off.",
            "valid": False,
            "raw_answer": answer(99),
        },
        {
            "recipe_id": None,
            "reasoning": None,
            "valid": False,
            "raw_answer": "garbage",
        },
        {
            "recipe_id": 1,
            "reasoning": "Good trade-off.",
            "valid": True,
            "raw_answer": answer(1),
        },
    ]


def test_select_recipes_stats_are_json_serializable(selector, pareto_front):
    """The stats are saved to MLflow as JSON."""
    ask = make_ask([answer(99), "garbage", answer(1)])

    _, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", COLUMN_LABELS, ask, n_votes=1
    )

    assert json.loads(json.dumps(stats)) == stats


def test_select_recipes_stops_at_max_calls(selector, pareto_front):
    """The LLM is not asked more than max_calls times."""
    ask = make_ask(["garbage"] * 100)

    selected, stats = selector.select_recipes(
        pareto_front, "Pick a recipe.", COLUMN_LABELS, ask, n_votes=3, max_calls=7
    )

    assert selected is None
    assert stats["n_calls"] == 7
    assert stats["n_invalid"] == 7
    assert stats["votes"] == []


def test_select_recipes_no_consensus(selector, pareto_front):
    """Enough valid votes but no majority returns None."""
    ask = make_ask([answer(0), answer(1), answer(2), answer(0)])

    selected, stats = selector.select_recipes(
        pareto_front,
        "Pick a recipe.",
        COLUMN_LABELS,
        ask,
        n_votes=4,
        threshold=0.75,
    )

    assert selected is None
    assert stats["votes"] == [0, 1, 2, 0]
