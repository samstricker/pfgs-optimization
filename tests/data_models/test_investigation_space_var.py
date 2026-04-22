"""Tests for InvestigationSpaceVariable dataclass."""

import pytest

from paretodo import InvestigationSpaceVariable


class TestInvestigationSpaceVariable:
    """Tests for InvestigationSpaceVariable class."""

    def test_valid_variable_creation(self):
        """Test creating a valid InvestigationSpaceVariable."""
        var = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
        )
        assert var.name == "temperature"
        assert var.min == 0.0
        assert var.max == 100.0
        assert var.discretization == 1.0
        assert var.expected_error is None

    def test_variable_with_expected_error(self):
        """Test creating a variable with expected_error."""
        var = InvestigationSpaceVariable(
            name="pressure",
            min=1.0,
            max=10.0,
            discretization=0.5,
            expected_error=0.5,
        )
        assert var.expected_error == 0.5

    def test_invalid_name_type(self):
        """Test that non-string name raises TypeError."""
        with pytest.raises(TypeError, match="Variable name must be a string"):
            InvestigationSpaceVariable(
                name=123,
                min=0.0,
                max=100.0,
                discretization=1.0,
            )

    def test_invalid_min_type(self):
        """Test that non-float min raises TypeError."""
        with pytest.raises(TypeError, match="min must be a number"):
            InvestigationSpaceVariable(
                name="temperature",
                min="0",
                max=100.0,
                discretization=1.0,
            )

    def test_invalid_max_type(self):
        """Test that non-float max raises TypeError."""
        with pytest.raises(TypeError, match="max must be a number"):
            InvestigationSpaceVariable(
                name="temperature",
                min=0.0,
                max="100",
                discretization=1.0,
            )

    def test_invalid_discretization_type(self):
        """Test that non-float discretization raises TypeError."""
        with pytest.raises(TypeError, match="discretization must be a number"):
            InvestigationSpaceVariable(
                name="temperature",
                min=0.0,
                max=100.0,
                discretization="1",
            )

    def test_min_greater_than_max(self):
        """Test that min >= max raises ValueError."""
        with pytest.raises(ValueError, match="min .* must be less than max"):
            InvestigationSpaceVariable(
                name="temperature",
                min=100.0,
                max=50.0,
                discretization=1.0,
            )

    def test_negative_discretization(self):
        """Test that negative discretization raises ValueError."""
        with pytest.raises(ValueError, match="discretization .* must be positive"):
            InvestigationSpaceVariable(
                name="temperature",
                min=0.0,
                max=100.0,
                discretization=-1.0,
            )

    def test_invalid_expected_error_type(self):
        """Test that non-float expected_error raises TypeError."""
        with pytest.raises(TypeError, match="expected_error must be a number"):
            InvestigationSpaceVariable(
                name="temperature",
                min=0.0,
                max=100.0,
                discretization=1.0,
                expected_error="0.5",
            )

    def test_negative_expected_error(self):
        """Test that negative expected_error raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            InvestigationSpaceVariable(
                name="temperature",
                min=0.0,
                max=100.0,
                discretization=1.0,
                expected_error=-0.5,
            )

    def test_expected_error_greater_than_range(self):
        """Test that expected_error >= range raises ValueError."""
        with pytest.raises(
            ValueError, match="expected_error .* must be less than the range"
        ):
            InvestigationSpaceVariable(
                name="temperature",
                min=0.0,
                max=10.0,
                discretization=1.0,
                expected_error=15.0,
            )

    def test_variable_is_frozen(self):
        """Test that InvestigationSpaceVariable is immutable."""
        var = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
        )
        with pytest.raises(AttributeError):
            var.name = "pressure"
