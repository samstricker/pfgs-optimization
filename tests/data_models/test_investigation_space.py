"""Tests for InvestigationSpace dataclass."""

import json
import tempfile
from pathlib import Path

import pytest

from paretodo import InvestigationSpace, InvestigationSpaceVariable


class TestInvestigationSpace:
    """Tests for InvestigationSpace class."""

    def test_valid_investigation_space_creation(self):
        """Test creating a valid InvestigationSpace."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
        )
        var2 = InvestigationSpaceVariable(
            name="pressure",
            min=1.0,
            max=10.0,
            discretization=0.5,
        )
        space = InvestigationSpace(variables=[var1, var2])
        assert len(space.variables) == 2
        assert space.n_var == 2

    def test_labels_property(self):
        """Test that labels property returns correct variable names."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
        )
        var2 = InvestigationSpaceVariable(
            name="pressure",
            min=1.0,
            max=10.0,
            discretization=0.5,
        )
        space = InvestigationSpace(variables=[var1, var2])
        assert space.labels == ["temperature", "pressure"]

    def test_bounds_property(self):
        """Test that bounds property returns correct min and max values."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
        )
        var2 = InvestigationSpaceVariable(
            name="pressure",
            min=1.0,
            max=10.0,
            discretization=0.5,
        )
        space = InvestigationSpace(variables=[var1, var2])
        lower_bounds, upper_bounds = space.bounds
        assert lower_bounds == [0.0, 1.0]
        assert upper_bounds == [100.0, 10.0]

    def test_expected_errors_property(self):
        """Test that expected_errors property returns correct values."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
            expected_error=2.0,
        )
        var2 = InvestigationSpaceVariable(
            name="pressure",
            min=1.0,
            max=10.0,
            discretization=0.5,
        )
        space = InvestigationSpace(variables=[var1, var2])
        assert space.expected_errors == [2.0, None]

    def test_int_bounds_property(self):
        """Test that int_bounds property returns correct integer bounds."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
        )
        var2 = InvestigationSpaceVariable(
            name="pressure",
            min=1.0,
            max=10.0,
            discretization=0.5,
        )
        space = InvestigationSpace(variables=[var1, var2])
        lower_int_bounds, upper_int_bounds = space.int_bounds
        assert lower_int_bounds == [0, 2]
        assert upper_int_bounds == [100, 20]

    def test_to_json(self):
        """Test saving InvestigationSpace to JSON file."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
            expected_error=2.0,
        )
        space = InvestigationSpace(variables=[var1])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "space.json"
            space.to_json(json_path)

            assert json_path.exists()
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            assert len(data["variables"]) == 1
            assert data["variables"][0]["name"] == "temperature"
            assert data["variables"][0]["min"] == 0.0
            assert data["variables"][0]["max"] == 100.0
            assert data["variables"][0]["discretization"] == 1.0
            assert data["variables"][0]["expected_error"] == 2.0

    def test_from_json(self):
        """Test loading InvestigationSpace from JSON file."""
        var_data = {
            "variables": [
                {
                    "name": "temperature",
                    "min": 0.0,
                    "max": 100.0,
                    "discretization": 1.0,
                    "expected_error": 2.0,
                },
                {
                    "name": "pressure",
                    "min": 1.0,
                    "max": 10.0,
                    "discretization": 0.5,
                    "expected_error": None,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "space.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(var_data, f)

            space = InvestigationSpace.from_json(json_path)

            assert len(space.variables) == 2
            assert space.variables[0].name == "temperature"
            assert space.variables[0].expected_error == 2.0
            assert space.variables[1].name == "pressure"
            assert space.variables[1].expected_error is None

    def test_as_dict(self):
        """Test converting InvestigationSpace to dictionary."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
            expected_error=2.0,
        )
        space = InvestigationSpace(variables=[var1])
        result = space.as_dict()

        assert "variables" in result
        assert len(result["variables"]) == 1
        assert result["variables"][0]["name"] == "temperature"

    def test_space_is_frozen(self):
        """Test that InvestigationSpace is immutable."""
        var1 = InvestigationSpaceVariable(
            name="temperature",
            min=0.0,
            max=100.0,
            discretization=1.0,
        )
        space = InvestigationSpace(variables=[var1])
        with pytest.raises(AttributeError):
            space.variables = []
