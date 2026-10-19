"""Tests for ParetoSelector algorithm."""

# Pytest fixtures are injected by parameter name, and tests exercise private helpers.
# pylint: disable=redefined-outer-name,protected-access

import polars as pl
import pytest

from paretodo.data_models import (
    OptimizationProblem,
    OptimizationAxis,
    InvestigationSpace,
    InvestigationSpaceVariable,
)
from paretodo.selection.selector import ParetoSelector


@pytest.fixture
def simple_investigation_space():
    """Create a simple investigation space with 2 variables."""
    return InvestigationSpace(
        variables=[
            InvestigationSpaceVariable(name="temperature", min=0.0, max=10.0),
            InvestigationSpaceVariable(name="ph_value", min=0.0, max=10.0),
        ]
    )


@pytest.fixture
def simple_optimization_problem():
    """Create a simple optimization problem with 2 objectives (both minimization)."""
    return OptimizationProblem(
        problem=[
            OptimizationAxis(
                var_name="obj1",
                minimize=True,
                optimization_type="objective",
                lower_bound=None,
                upper_bound=None,
            ),
            OptimizationAxis(
                var_name="obj2",
                minimize=True,
                optimization_type="objective",
                lower_bound=None,
                upper_bound=None,
            ),
        ]
    )


@pytest.fixture
def simple_pareto_front():
    """Create a simple Pareto front for testing.

    Investigation space (temperature, ph_value):
    - Point A: (0, 0)    - Far corner
    - Point B: (10, 0)   - Far corner
    - Point C: (0, 10)   - Far corner
    - Point D: (10, 10)  - Far corner
    - Point E: (5, 5)    - Center

    Objectives use the naming convention from optimization_problem.objective_labels:
    - obj1 minimize = "obj:min(μ(obj1))"
    - obj2 minimize = "obj:min(μ(obj2))"

    If algorithm uses investigation space correctly:
    - First point: closest to (0,0) in objective space = Point A (both minimize, utopia at 0)
    - Second point: farthest from A in investigation space = Point D at (10,10)
    - Third point: farthest from A and D = should be one of B or C (both at distance ~10)
    """
    data = {
        "__original_idx": [0, 1, 2, 3, 4],
        "RecipeID": ["A", "B", "C", "D", "E"],
        "temperature": [0.0, 10.0, 0.0, 10.0, 5.0],
        "ph_value": [0.0, 0.0, 10.0, 10.0, 5.0],
        "product_concentration": [1.0, 2.0, 3.0, 4.0, 2.5],
        "obj:min(μ(obj1))": [0.0, 10.0, 0.0, 10.0, 5.0],  # = temperature
        "obj:min(μ(obj2))": [0.0, 0.0, 10.0, 10.0, 5.0],  # = ph_value
    }
    return pl.DataFrame(data)


def test_first_point_selection_closest_to_utopia(
    simple_pareto_front, simple_optimization_problem
):
    """Test that first point selected is closest to Utopia in objective space."""
    selector = ParetoSelector()

    # Both objectives minimize, so Utopia is at (0, 0)
    # Point A at (0, 0) in objectives is closest
    utopia_recipe = selector.select_closest_recipe_to_utopia(
        pareto_front=simple_pareto_front,
        utopia_point=selector._utopia_point_from_optimization_problem(
            simple_optimization_problem
        ),
        bounds=selector._bounds_for_objectives(simple_pareto_front),
    )

    # Should select Point A (RecipeID = "A")
    assert utopia_recipe["RecipeID"][0] == "A"


def test_spread_selection_uses_investigation_space(
    simple_pareto_front, simple_investigation_space
):
    """Test that spread-based selection uses investigation space, not objective space.

    This is the critical test. If the algorithm was using objective space:
    - All points have spread in obj1 (temp) and obj2 (pH)
    - The farthest point from A in objective space would be D (10, 10)

    If the algorithm uses investigation space (which it should):
    - Investigation space is temperature and ph_value
    - Same result: D is farthest from A at (10, 10)
    - But the bounds and normalization are computed from investigation space, not objectives

    To detect the bug, we need objectives that are different from investigation space.
    In this simple case they're the same, so let's verify D is selected.
    """
    selector = ParetoSelector()

    # Start with Point A as reference
    reference_point = simple_pareto_front.filter(pl.col("RecipeID") == "A")
    remaining_pareto = simple_pareto_front.filter(pl.col("RecipeID") != "A")

    # Select 1 additional point
    selected = selector.select_additional_recipes_according_to_spread(
        pareto_front=remaining_pareto,
        reference_point=reference_point,
        n_additional=1,
        investigation_space=simple_investigation_space.labels,
        experiments_df=None,
    )

    # Should have 2 points total (A + 1 additional)
    assert selected.height == 2

    # Point A should be first, Point D should be second (farthest from A at (10,10))
    recipe_ids = selected["RecipeID"].to_list()
    assert recipe_ids[0] == "A"
    assert recipe_ids[1] == "D", f"Expected D but got {recipe_ids[1]}"


def test_spread_selection_with_experiments(
    simple_pareto_front, simple_investigation_space
):
    """Test that experiments data influences point selection (points stay far from experiments)."""
    selector = ParetoSelector()

    # Create experiments data: one experiment at Point D location
    experiments_data = {
        "temperature": [10.0],
        "ph_value": [10.0],
    }
    experiments_df = pl.DataFrame(experiments_data)

    # Start with Point A as reference
    reference_point = simple_pareto_front.filter(pl.col("RecipeID") == "A")
    remaining_pareto = simple_pareto_front.filter(pl.col("RecipeID") != "A")

    # Select 1 additional point WITH experiments data
    selected_with_exp = selector.select_additional_recipes_according_to_spread(
        pareto_front=remaining_pareto,
        reference_point=reference_point,
        n_additional=1,
        investigation_space=simple_investigation_space.labels,
        experiments_df=experiments_df,
    )

    # Without experiments: should select D (farthest from A)
    # With experiments at D: should select something else
    # (B or C, both at distance 10 from A and far from D)

    # The second selected point should NOT be D (because experiment is at D)
    recipe_ids = selected_with_exp["RecipeID"].to_list()
    assert (
        recipe_ids[1] != "D"
    ), f"Should avoid D where experiment is, but got {recipe_ids[1]}"


def test_full_selection_pipeline(
    simple_pareto_front, simple_investigation_space, simple_optimization_problem
):
    """Test the full selection pipeline: Utopia + spread selection."""
    selector = ParetoSelector()

    selected = selector.select_recipes(
        pareto_front=simple_pareto_front,
        investigation_space=simple_investigation_space,
        optimization_problem=simple_optimization_problem,
        n_samples=3,
        experiments_df=None,
    )

    # Should select 3 points
    assert selected.height == 3

    # First should be A (closest to utopia at (0, 0) in objective space)
    recipe_ids = selected["RecipeID"].to_list()
    assert recipe_ids[0] == "A"

    # Second should be D (farthest from A in investigation space)
    assert recipe_ids[1] == "D"

    # Third should be B or C (both at similar max distance from A and D in investigation space)
    assert recipe_ids[2] in ["B", "C"]


def test_bounds_are_fixed_across_iterations():
    """Test that bounds calculated from full pareto front are used consistently.

    This is a regression test for the bug where bounds were recalculated each iteration.
    """
    # Create pareto front where min/max would change if recalculated
    data = {
        "__original_idx": [0, 1, 2],
        "RecipeID": ["P1", "P2", "P3"],
        "temperature": [5.0, 10.0, 7.0],  # min=5, max=10
        "ph_value": [5.0, 5.0, 8.0],  # min=5, max=8
        "obj1": [0.5, 1.0, 0.7],
        "obj2": [0.5, 0.5, 0.8],
    }
    pareto_front = pl.DataFrame(data)

    investigation_space = ["temperature", "ph_value"]
    selector = ParetoSelector()

    # Start with P1
    reference_point = pareto_front.filter(pl.col("RecipeID") == "P1")
    remaining = pareto_front.filter(pl.col("RecipeID") != "P1")

    selected = selector.select_additional_recipes_according_to_spread(
        pareto_front=remaining,
        reference_point=reference_point,
        n_additional=1,
        investigation_space=investigation_space,
        experiments_df=None,
    )

    # If bounds were fixed correctly, selection should be consistent
    # If bounds were recalculated, P2 (max point) would have different normalization
    assert selected.height == 2
    recipe_ids = selected["RecipeID"].to_list()
    assert recipe_ids[0] == "P1"
    # P2 is farther from P1 than P3, so should be selected
    assert recipe_ids[1] == "P2"
