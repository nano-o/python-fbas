import pytest
from python_fbas.config import update, get
from python_fbas.propositional_logic import (
    Formula, Atom, Not, And, Or, Implies, AtLeast, equiv,
    to_cnf, var, anonymous_var, decode_model, atoms_of_clauses
)
import python_fbas.propositional_logic as pl


class TestFormulaClasses:
    """Test the basic Formula classes and their instantiation."""

    def test_atom_creation(self):
        """Test Atom creation with different identifier types."""
        atom1 = Atom('x')
        atom2 = Atom(42)
        atom3 = Atom(('a', 'b'))

        assert atom1.identifier == 'x'
        assert atom2.identifier == 42
        assert atom3.identifier == ('a', 'b')
        assert isinstance(atom1, Formula)
        assert isinstance(atom1, Atom)

    def test_not_creation(self):
        """Test Not formula creation."""
        atom = Atom('x')
        not_atom = Not(atom)

        assert not_atom.operand == atom
        assert isinstance(not_atom, Formula)
        assert isinstance(not_atom, Not)

    def test_and_creation(self):
        """Test And formula creation."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        atom3 = Atom('z')

        and_formula = And(atom1, atom2, atom3)

        assert len(and_formula.operands) == 3
        assert atom1 in and_formula.operands
        assert atom2 in and_formula.operands
        assert atom3 in and_formula.operands
        assert isinstance(and_formula, Formula)
        assert isinstance(and_formula, And)

    def test_or_creation(self):
        """Test Or formula creation."""
        atom1 = Atom('x')
        atom2 = Atom('y')

        or_formula = Or(atom1, atom2)

        assert len(or_formula.operands) == 2
        assert atom1 in or_formula.operands
        assert atom2 in or_formula.operands
        assert isinstance(or_formula, Formula)
        assert isinstance(or_formula, Or)

    def test_implies_creation(self):
        """Test Implies formula creation."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        atom3 = Atom('z')

        implies_formula = Implies(atom1, atom2, atom3)

        assert len(implies_formula.operands) == 3
        assert implies_formula.operands == [atom1, atom2, atom3]
        assert isinstance(implies_formula, Formula)
        assert isinstance(implies_formula, Implies)

    def test_implies_creation_requires_minimum_operands(self):
        """Test that Implies requires at least 2 operands."""
        atom1 = Atom('x')

        with pytest.raises(AssertionError):
            Implies(atom1)

    def test_card_creation(self):
        """Test Card formula creation."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        atom3 = Atom('z')

        card_formula = AtLeast(2, atom1, atom2, atom3)

        assert card_formula.threshold == 2
        assert len(card_formula.operands) == 3
        assert card_formula.operands == [atom1, atom2, atom3]
        assert isinstance(card_formula, Formula)
        assert isinstance(card_formula, AtLeast)

    def test_card_creation_validation(self):
        """Test Card formula validation."""
        atom1 = Atom('x')
        atom2 = Atom('y')

        # Valid card constraint
        card_formula = AtLeast(1, atom1, atom2)
        assert card_formula.threshold == 1

        # Invalid: threshold must be positive
        with pytest.raises(AssertionError):
            AtLeast(0, atom1, atom2)

        # Invalid: threshold must be <= number of operands
        with pytest.raises(AssertionError):
            AtLeast(3, atom1, atom2)

    def test_equiv_function(self):
        """Test the equiv helper function."""
        atom1 = Atom('x')
        atom2 = Atom('y')

        equiv_formula = equiv(atom1, atom2)

        assert isinstance(equiv_formula, And)
        assert len(equiv_formula.operands) == 2
        # Check that both implications are present
        assert any(isinstance(op, Implies) for op in equiv_formula.operands)


class TestVariableManagement:
    """Test variable assignment and management."""

    def setup_method(self):
        """Clear global variables before each test."""
        pl.variables.clear()
        pl.variables_inv.clear()
        pl.next_int = 1

    def test_var_function(self):
        """Test variable assignment."""
        assert var('x') == 1
        assert var('y') == 2
        assert var('x') == 1  # Should return same variable

        assert pl.variables['x'] == 1
        assert pl.variables['y'] == 2
        assert pl.variables_inv[1] == 'x'
        assert pl.variables_inv[2] == 'y'

    def test_var_with_complex_identifiers(self):
        """Test variable assignment with complex identifiers."""
        tuple_id = ('a', 'b')
        # Use tuple instead of list since lists aren't hashable
        tuple_id2 = (1, 2, 3)

        var1 = var(tuple_id)
        var2 = var(tuple_id2)

        assert var1 == 1
        assert var2 == 2
        assert pl.variables[tuple_id] == 1
        assert pl.variables_inv[1] == tuple_id

    def test_anonymous_var(self):
        """Test anonymous variable creation."""
        anon1 = anonymous_var()
        anon2 = anonymous_var()

        assert anon1 == 1
        assert anon2 == 2
        assert anon1 not in pl.variables_inv
        assert anon2 not in pl.variables_inv


class TestCNFConversion:
    """Test CNF conversion functionality."""

    def setup_method(self):
        """Clear global variables before each test."""
        pl.variables.clear()
        pl.variables_inv.clear()
        pl.next_int = 1

    def test_atom_to_cnf(self):
        """Test converting atom to CNF."""
        atom = Atom('x')
        cnf = to_cnf(atom)

        assert cnf == [[1]]
        assert pl.variables['x'] == 1

    def test_not_atom_to_cnf(self):
        """Test converting negated atom to CNF."""
        atom = Atom('x')
        not_atom = Not(atom)
        cnf = to_cnf(not_atom)

        assert cnf == [[-1]]
        assert pl.variables['x'] == 1

    def test_and_to_cnf(self):
        """Test converting And formula to CNF."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        and_formula = And(atom1, atom2)
        cnf = to_cnf(and_formula)

        # Should create auxiliary variable for the And
        # Expected: [-1, -2, aux], [-aux, 1], [-aux, 2], [aux]
        assert len(cnf) == 4
        assert cnf[-1] == [3]  # Final unit clause for the And

        # Check that both x and y are assigned variables
        assert pl.variables['x'] == 1
        assert pl.variables['y'] == 2

    def test_or_to_cnf(self):
        """Test converting Or formula to CNF."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        or_formula = Or(atom1, atom2)
        cnf = to_cnf(or_formula)

        # Should create auxiliary variable for the Or
        # Expected: [-1, aux], [-2, aux], [-aux, 1, 2], [aux]
        assert len(cnf) == 4
        assert cnf[-1] == [3]  # Final unit clause for the Or

    def test_implies_to_cnf(self):
        """Test converting Implies formula to CNF."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        implies_formula = Implies(atom1, atom2)
        cnf = to_cnf(implies_formula)

        # Implies(x, y) should be converted to Or(Not(x), y)
        assert len(cnf) > 0
        assert cnf[-1][-1] > 0  # Final unit clause should be positive

    def test_empty_and_to_cnf(self):
        """Test converting empty And to CNF."""
        empty_and = And()
        cnf = to_cnf(empty_and)

        # Empty And should be trivially satisfiable
        assert len(cnf) == 1
        assert len(cnf[0]) == 1
        assert cnf[0][0] > 0  # Positive unit clause

    def test_empty_or_to_cnf(self):
        """Test converting empty Or to CNF."""
        empty_or = Or()
        cnf = to_cnf(empty_or)

        # Empty Or should be unsatisfiable
        assert len(cnf) == 2
        v = cnf[1][0]
        assert cnf == [[-v], [v]]

    def test_list_of_formulas_to_cnf(self):
        """Test converting list of formulas to CNF."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        formulas = [atom1, atom2]
        cnf = to_cnf(formulas)

        # Should be conjunction of the formulas
        assert cnf == [[1], [2]]
        assert pl.variables['x'] == 1
        assert pl.variables['y'] == 2


class TestCardinalityConstraints:
    """Test cardinality constraint conversion."""

    def setup_method(self):
        """Clear global variables and set encoding method."""
        pl.variables.clear()
        pl.variables_inv.clear()
        pl.next_int = 1
        # Use naive encoding for predictable test results
        self.original_encoding = get().card_encoding
        update(card_encoding='naive')

    def teardown_method(self):
        """Restore original encoding."""
        update(card_encoding=self.original_encoding)

    def test_card_naive_encoding(self):
        """Test cardinality constraint with naive encoding."""
        atom1 = Atom('x')
        atom2 = Atom('y')
        atom3 = Atom('z')

        card_formula = AtLeast(2, atom1, atom2, atom3)
        cnf = to_cnf(card_formula)

        # TODO: Should generate combinations: (x ∧ y) ∨ (x ∧ z) ∨ (y ∧ z)
        assert len(cnf) > 0
        assert 'x' in pl.variables
        assert 'y' in pl.variables
        assert 'z' in pl.variables

    def test_card_totalizer_encoding(self):
        """Test cardinality constraint with totalizer encoding."""
        update(card_encoding='totalizer')

        # TODO

    def test_card_special_cases(self):
        """Test cardinality constraint special cases."""
        update(card_encoding='totalizer')

        # TODO

    def test_card_invalid_encoding(self):
        """Test cardinality constraint with invalid encoding."""
        update(card_encoding='invalid')

        atom1 = Atom('x')
        atom2 = Atom('y')

        card_formula = AtLeast(1, atom1, atom2)

        with pytest.raises(ValueError, match="Unknown cardinality encoding"):
            to_cnf(card_formula)


class TestUtilityFunctions:
    """Test utility functions."""

    def setup_method(self):
        """Clear global variables before each test."""
        pl.variables.clear()
        pl.variables_inv.clear()
        pl.next_int = 1

    def test_atoms_of_clauses(self):
        """Test atoms_of_clauses function."""
        clauses = [[1, -2, 3], [-1, 2], [3, -4]]
        atoms = atoms_of_clauses(clauses)

        assert atoms == {1, 2, 3, 4}

    def test_atoms_of_clauses_empty(self):
        """Test atoms_of_clauses with empty clauses."""
        clauses = []
        atoms = atoms_of_clauses(clauses)

        assert atoms == set()

    def test_decode_model_basic(self):
        """Test decode_model function."""
        # Set up variables
        var('x')
        var('y')
        var('z')

        # Model: x=True, y=False, z=True
        model = [1, -2, 3]
        decoded = decode_model(model)

        assert set(decoded) == {'x', 'z'}

    def test_decode_model_with_predicate(self):
        """Test decode_model with predicate filter."""
        # Set up variables
        var('x')
        var('y')
        var('z')

        # Model with some variables
        model = [1, 2, 3]
        decoded = decode_model(model, predicate=lambda x: x != 'y')

        assert set(decoded) == {'x', 'z'}

    def test_decode_model_with_anonymous_variables(self):
        """Test decode_model ignores anonymous variables."""
        # Set up some named variables
        var('x')
        var('y')

        # Model includes anonymous variables (not in variables_inv)
        model = [1, -2, 3, -4, 5]  # 3, 4, 5 are anonymous
        decoded = decode_model(model)

        assert set(decoded) == {'x'}  # Only 'x' is true and named

    def test_decode_model_empty(self):
        """Test decode_model with empty model."""
        decoded = decode_model([])
        assert decoded == []


class TestComplexFormulas:
    """Test complex formula combinations."""

    def setup_method(self):
        """Clear global variables before each test."""
        pl.variables.clear()
        pl.variables_inv.clear()
        pl.next_int = 1

    # TODO...


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Clear global variables before each test."""
        pl.variables.clear()
        pl.variables_inv.clear()
        pl.next_int = 1

    def test_unsupported_formula_type(self):
        """Test conversion of unsupported formula type."""
        class UnsupportedFormula(Formula):
            pass

        unsupported = UnsupportedFormula()

        with pytest.raises(TypeError):
            to_cnf(unsupported)


if __name__ == '__main__':
    pytest.main([__file__])
