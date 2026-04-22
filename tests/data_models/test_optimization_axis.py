"""Tests for OptimizationAxis dataclass."""

import pytest

from paretodo import OptimizationAxis


class TestOptimizationAxis:
    """Tests for OptimizationAxis class."""

    def test_valid_optimization_axis_creation(self):
        """Test creating a valid OptimizationAxis."""
        axis = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        assert axis.var_name == "temperature"
        assert axis.minimize is True
        assert axis.optimization_type == "objective"

    def test_optimization_axis_with_bounds(self):
        """Test creating OptimizationAxis with bounds."""
        axis = OptimizationAxis(
            var_name="temperature",
            minimize=False,
            optimization_type="constraint",
            lower_bound=50.0,
            upper_bound=100.0,
        )
        assert axis.lower_bound == 50.0
        assert axis.upper_bound == 100.0

    def test_invalid_var_name_type(self):
        """Test that non-string var_name raises TypeError."""
        with pytest.raises(TypeError, match="Variable name must be a string"):
            OptimizationAxis(
                var_name=123,
                minimize=True,
                optimization_type="objective",
                lower_bound=None,
                upper_bound=None,
            )

    def test_invalid_minimize_type(self):
        """Test that non-boolean minimize raises TypeError."""
        with pytest.raises(TypeError, match="Minimize flag must be a boolean"):
            OptimizationAxis(
                var_name="temperature",
                minimize="true",
                optimization_type="objective",
                lower_bound=None,
                upper_bound=None,
            )

    def test_invalid_optimization_type(self):
        """Test that non-string optimization_type raises TypeError."""
        with pytest.raises(TypeError, match="Optimization type must be a string"):
            OptimizationAxis(
                var_name="temperature",
                minimize=True,
                optimization_type=123,
                lower_bound=None,
                upper_bound=None,
            )

    def test_invalid_lower_bound_type(self):
        """Test that non-float lower_bound raises TypeError."""
        with pytest.raises(TypeError, match="Lower bound must be a float or None"):
            OptimizationAxis(
                var_name="temperature",
                minimize=True,
                optimization_type="constraint",
                lower_bound="50",
                upper_bound=100.0,
            )

    def test_invalid_upper_bound_type(self):
        """Test that non-float upper_bound raises TypeError."""
        with pytest.raises(TypeError, match="Upper bound must be a float or None"):
            OptimizationAxis(
                var_name="temperature",
                minimize=True,
                optimization_type="constraint",
                lower_bound=50.0,
                upper_bound="100",
            )

    def test_lower_bound_greater_than_upper_bound(self):
        """Test that lower_bound >= upper_bound raises ValueError."""
        with pytest.raises(
            ValueError, match="Lower bound .* must be less than upper bound"
        ):
            OptimizationAxis(
                var_name="temperature",
                minimize=True,
                optimization_type="constraint",
                lower_bound=100.0,
                upper_bound=50.0,
            )

    def test_axis_is_frozen(self):
        """Test that OptimizationAxis is immutable."""
        axis = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        with pytest.raises(AttributeError):
            axis.minimize = False
