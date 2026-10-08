"""Consistency checks for src/utils/code_style.py and a regression guard
against the drift bug CLAUDE.md flags: visualize_ensemble_stats.py and
plot_pergroup_ens_figure12.py must import the color/display-name registry
from code_style.py rather than re-declaring it locally (it drifted twice
before this module was extracted).

The regression guard is AST-based: it parses each consumer module and fails
if it finds a local module-level assignment that redefines CODE_COLORS or
CODE_DISPLAY_NAMES, instead of importing those names from code_style.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "src" / "utils"
sys.path.insert(0, str(UTILS_DIR))

from code_style import (  # noqa: E402
    CODE_COLORS,
    CODE_DISPLAY_NAMES,
    code_of,
    code_color,
    code_display,
)

CONSUMER_FILES = [
    UTILS_DIR / "visualize_ensemble_stats.py",
    UTILS_DIR / "plot_pergroup_ens_figure12.py",
]

REGISTRY_NAMES = {"CODE_COLORS", "CODE_DISPLAY_NAMES"}


def test_every_color_code_has_a_display_name():
    missing = set(CODE_COLORS) - set(CODE_DISPLAY_NAMES)
    assert not missing, f"codes in CODE_COLORS with no CODE_DISPLAY_NAMES entry: {missing}"


def test_every_display_name_code_has_a_color():
    missing = set(CODE_DISPLAY_NAMES) - set(CODE_COLORS)
    assert not missing, f"codes in CODE_DISPLAY_NAMES with no CODE_COLORS entry: {missing}"


def test_code_of_extracts_bare_code_from_scenario_label():
    assert code_of("eqdyna/0001.A.100m") == "eqdyna"
    assert code_of("eqdyna") == "eqdyna"


def test_code_color_matches_registry_for_every_known_code():
    for code in CODE_COLORS:
        assert code_color(code) == CODE_COLORS[code]
        assert code_color(f"{code}/some_scenario") == CODE_COLORS[code]


def test_code_color_falls_back_to_gray_for_unknown_code():
    assert code_color("not_a_real_code") == "tab:gray"


def test_code_display_matches_registry_for_every_known_code():
    for code in CODE_DISPLAY_NAMES:
        assert code_display(code) == CODE_DISPLAY_NAMES[code]


def test_code_display_falls_back_to_bare_code_for_unknown_code():
    assert code_display("not_a_real_code") == "not_a_real_code"


def _module_level_assigned_names(py_path):
    """Names assigned directly at module (top) scope via AST, so the check
    survives reformatting or renaming of imports and does not depend on a
    textual grep for the literal text 'CODE_COLORS ='."""
    tree = ast.parse(py_path.read_text(), filename=str(py_path))
    assigned = set()
    for node in tree.body:  # module top level only, not nested in functions
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    return assigned


def _module_imports_from_code_style(py_path):
    tree = ast.parse(py_path.read_text(), filename=str(py_path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "code_style":
            for alias in node.names:
                imported.add(alias.name)
    return imported


def test_consumers_import_registry_rather_than_redeclare_it():
    """Regression test for the documented drift bug: each consumer file must
    import CODE_COLORS and CODE_DISPLAY_NAMES from code_style, and must not
    also bind those names via a local module-level assignment."""
    for path in CONSUMER_FILES:
        assert path.is_file(), f"expected consumer file missing: {path}"
        imported = _module_imports_from_code_style(path)
        missing_imports = REGISTRY_NAMES - imported
        assert not missing_imports, (
            f"{path.name} does not import {missing_imports} from code_style; "
            "it may be re-declaring the registry locally"
        )
        locally_assigned = _module_level_assigned_names(path)
        redeclared = REGISTRY_NAMES & locally_assigned
        assert not redeclared, (
            f"{path.name} re-declares {redeclared} at module scope instead of "
            "importing from code_style; this is the drift bug CLAUDE.md warns about"
        )


def test_mutation_sensitivity_local_redeclaration_is_caught(tmp_path):
    """Mutation probe: a scratch copy of a consumer file that re-declares
    CODE_COLORS locally (the historical drift bug) must fail the guard
    above. Confirms the AST check actually catches the failure mode it is
    meant to catch, not just the presence of an import statement."""
    original = CONSUMER_FILES[0].read_text()
    mutated_src = original + "\n\nCODE_COLORS = {'eqdyna': 'red'}\n"
    mutated = tmp_path / "visualize_ensemble_stats_mutated.py"
    mutated.write_text(mutated_src)

    locally_assigned = _module_level_assigned_names(mutated)
    assert "CODE_COLORS" in locally_assigned, (
        "Mutation probe did not introduce a detectable local redeclaration"
    )
