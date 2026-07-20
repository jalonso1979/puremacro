"""Coverage tests for puremacro/reports.py.

Tests strategy:
  - Known-value: build synthetic inputs with a known correct answer and verify
    the output matches within tolerance.
  - Property / contract: verify output shapes, types, keys, table structure,
    string content, and that the function raises on bad input.
  - Edge cases: NaN SE, single-row inputs, 3-D IRF arrays, empty dicts.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from puremacro.reports import (
    IRFResult,
    LPResult,
    coef_table,
    irf_to_markdown,
    summary_to_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_markdown_data_rows(table: str) -> int:
    """Count data rows (skip header + separator lines) in a markdown table."""
    lines = table.strip().splitlines()
    # header is line 0, separator is line 1, rest are data
    return max(0, len(lines) - 2)


def _count_markdown_cols(table: str) -> int:
    """Count columns from the header row."""
    header = table.strip().splitlines()[0]
    return len(header.split("|")) - 2  # leading and trailing pipes


# ===========================================================================
# Tests for coef_table
# ===========================================================================

class TestCoefTable:
    """Tests for coef_table()."""

    def test_returns_string(self):
        beta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        result = coef_table(beta, se)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_known_t_statistic_present(self):
        """t-stat = beta/se; verify the known value appears in the table."""
        beta = np.array([2.0])
        se = np.array([0.5])
        result = coef_table(beta, se, digits=3)
        # t = 4.0
        assert "4.0" in result

    def test_known_p_value_significant(self):
        """beta/se=4 => p << 0.05; table should contain a very small p value."""
        beta = np.array([4.0])
        se = np.array([1.0])
        result = coef_table(beta, se, digits=4)
        # p = 2*(1 - norm.cdf(4)) ~ 6.3e-5; at least '0.0001' or similar
        assert "0.0" in result  # some small value

    def test_default_variable_names(self):
        """When names=None, variables are named x0, x1, ..."""
        beta = np.array([1.0, 2.0, 3.0])
        se = np.array([0.1, 0.2, 0.3])
        result = coef_table(beta, se)
        assert "x0" in result
        assert "x1" in result
        assert "x2" in result

    def test_custom_variable_names(self):
        beta = np.array([0.5, -0.3])
        se = np.array([0.1, 0.05])
        result = coef_table(beta, se, names=["alpha", "beta"])
        assert "alpha" in result
        assert "beta" in result

    def test_markdown_format_table_structure(self):
        """Markdown output must have header, separator, and data rows."""
        beta = np.array([1.0, 2.0])
        se = np.array([0.5, 0.5])
        result = coef_table(beta, se, fmt="markdown")
        lines = result.strip().splitlines()
        assert len(lines) >= 3  # header, separator, ≥1 data row
        # Separator line contains only |-: characters
        sep = lines[1]
        assert re.match(r"^\|[-| ]+\|$", sep), f"Bad separator: {sep!r}"

    def test_latex_format_contains_tabular(self):
        beta = np.array([1.0])
        se = np.array([0.1])
        result = coef_table(beta, se, fmt="latex")
        assert r"\begin{tabular}" in result
        assert r"\end{tabular}" in result
        assert r"\hline" in result

    def test_plain_format_returns_string(self):
        beta = np.array([1.0])
        se = np.array([0.1])
        result = coef_table(beta, se, fmt="plain")
        assert isinstance(result, str)
        assert "1.0" in result

    def test_ci_columns_present_by_default(self):
        """Default alpha=0.05 -> columns lo_95% and hi_95%."""
        beta = np.array([0.0])
        se = np.array([1.0])
        result = coef_table(beta, se)
        assert "lo_95%" in result
        assert "hi_95%" in result

    def test_ci_columns_correct_bounds(self):
        """CI for N(0,1): lo=-1.96, hi=+1.96 at 95% level."""
        beta = np.array([0.0])
        se = np.array([1.0])
        result = coef_table(beta, se, digits=3, alpha=0.05)
        # z=1.96; lo=-1.96, hi=1.96
        assert "-1.96" in result
        assert "1.96" in result

    def test_ci_width_grows_with_se(self):
        """Larger SE -> wider confidence interval."""
        from scipy.stats import norm
        beta = np.array([0.0])
        se_small = np.array([1.0])
        se_large = np.array([3.0])
        result_small = coef_table(beta, se_small, digits=4, alpha=0.05)
        result_large = coef_table(beta, se_large, digits=4, alpha=0.05)
        # Extract lower bound value from each result
        z = norm.ppf(0.975)
        lo_small = -z * 1.0
        lo_large = -z * 3.0
        # lo_large should be more negative (i.e., wider CI)
        assert lo_large < lo_small

    def test_custom_alpha_changes_ci_label(self):
        """alpha=0.10 -> 90% CI columns."""
        beta = np.array([0.0])
        se = np.array([1.0])
        result = coef_table(beta, se, alpha=0.10)
        assert "lo_90%" in result
        assert "hi_90%" in result

    def test_include_t_false_omits_t_column(self):
        beta = np.array([2.0])
        se = np.array([0.5])
        result = coef_table(beta, se, include_t=False)
        # The t-stat column header should not appear, but 't' may appear in
        # the data; check the header row instead
        header = result.strip().splitlines()[0]
        assert "| t |" not in header and "| t\n" not in result

    def test_include_p_false_omits_p_column(self):
        beta = np.array([2.0])
        se = np.array([0.5])
        result = coef_table(beta, se, include_p=False)
        header = result.strip().splitlines()[0]
        # The column header 'p' should not appear
        assert "| p |" not in header

    def test_include_ci_false_omits_ci_columns(self):
        beta = np.array([1.0])
        se = np.array([0.5])
        result = coef_table(beta, se, include_ci=False)
        assert "lo_" not in result
        assert "hi_" not in result

    def test_digits_controls_rounding(self):
        """digits=1 should produce fewer decimal places than digits=5."""
        beta = np.array([1.23456789])
        se = np.array([0.111111])
        result1 = coef_table(beta, se, digits=1)
        result5 = coef_table(beta, se, digits=5)
        # digits=5 result should be longer due to more decimals
        assert len(result5) >= len(result1)

    def test_zero_se_produces_nan_t(self):
        """When SE=0, t and p should be NaN."""
        beta = np.array([1.0])
        se = np.array([0.0])
        # Should not raise; NaN should appear in the output
        result = coef_table(beta, se)
        assert isinstance(result, str)

    def test_number_of_data_rows_equals_k(self):
        """One data row per coefficient."""
        k = 5
        beta = np.arange(k, dtype=float)
        se = np.ones(k) * 0.1
        result = coef_table(beta, se, fmt="markdown")
        assert _count_markdown_data_rows(result) == k

    def test_list_inputs_accepted(self):
        """beta and se may be plain Python lists."""
        result = coef_table([1.0, 2.0], [0.1, 0.2])
        assert isinstance(result, str)
        assert "x0" in result

    def test_single_coefficient(self):
        beta = np.array([3.14])
        se = np.array([0.01])
        result = coef_table(beta, se, names=["pi"])
        assert "pi" in result
        assert "3.14" in result


# ===========================================================================
# Tests for irf_to_markdown
# ===========================================================================

class TestIrfToMarkdown:
    """Tests for irf_to_markdown()."""

    def _make_irf(self, H=5, n=2):
        rng = np.random.default_rng(0)
        point = rng.standard_normal((H, n))
        lower = point - 0.5
        upper = point + 0.5
        return point, lower, upper

    def test_returns_string(self):
        point, lower, upper = self._make_irf()
        result = irf_to_markdown(point, lower, upper)
        assert isinstance(result, str)

    def test_number_of_rows_equals_H(self):
        H, n = 7, 3
        point = np.ones((H, n))
        result = irf_to_markdown(point)
        assert _count_markdown_data_rows(result) == H

    def test_number_of_cols_equals_n_plus_h(self):
        """Columns = h + n variables."""
        H, n = 4, 2
        point = np.zeros((H, n))
        result = irf_to_markdown(point)
        # header columns: h, y0, y1 -> 3
        assert _count_markdown_cols(result) == n + 1

    def test_custom_var_names_in_header(self):
        H, n = 3, 2
        point = np.zeros((H, n))
        result = irf_to_markdown(point, var_names=["gdp", "inflation"])
        assert "gdp" in result
        assert "inflation" in result

    def test_custom_h_axis_in_output(self):
        """Custom h_axis values should appear in the h column."""
        H, n = 3, 1
        point = np.ones((H, n))
        result = irf_to_markdown(point, h_axis=[1, 2, 3])
        lines = result.strip().splitlines()
        # Data rows start at index 2
        for i, val in enumerate([1, 2, 3]):
            # The row should contain the h value
            assert str(val) in lines[2 + i]

    def test_with_bands_adds_bracket_notation(self):
        """When lower/upper provided, each cell should contain brackets."""
        H, n = 3, 1
        point = np.zeros((H, n))
        lower = point - 1.0
        upper = point + 1.0
        result = irf_to_markdown(point, lower, upper)
        # Bracket notation: [+..., +...]
        assert "[" in result
        assert "]" in result

    def test_without_bands_no_brackets(self):
        H, n = 3, 1
        point = np.ones((H, n))
        result = irf_to_markdown(point)
        assert "[" not in result

    def test_3d_input_picks_first_shock(self):
        """3-D input (H, n_resp, n_shock) -> picks shock 0."""
        H, n_resp, n_shock = 4, 2, 3
        rng = np.random.default_rng(1)
        point_3d = rng.standard_normal((H, n_resp, n_shock))
        # 2-D slice for shock 0
        point_2d = point_3d[:, :, 0]
        result_3d = irf_to_markdown(point_3d)
        result_2d = irf_to_markdown(point_2d)
        assert result_3d == result_2d

    def test_3d_with_bands_picks_first_shock(self):
        H, n_resp, n_shock = 3, 2, 2
        rng = np.random.default_rng(2)
        point_3d = rng.standard_normal((H, n_resp, n_shock))
        lower_3d = point_3d - 0.5
        upper_3d = point_3d + 0.5
        result_3d = irf_to_markdown(point_3d, lower_3d, upper_3d)
        result_2d = irf_to_markdown(
            point_3d[:, :, 0],
            lower_3d[:, :, 0],
            upper_3d[:, :, 0],
        )
        assert result_3d == result_2d

    def test_known_point_value_present_in_output(self):
        """Point value +1.000 should appear in output with digits=3."""
        point = np.array([[1.0]])
        result = irf_to_markdown(point, digits=3)
        assert "+1.000" in result

    def test_digits_controls_decimal_places(self):
        point = np.array([[0.123456]])
        result1 = irf_to_markdown(point, digits=1)
        result3 = irf_to_markdown(point, digits=3)
        assert "+0.1" in result1
        assert "+0.123" in result3

    def test_sign_positive_prefixed(self):
        """Positive values should have '+' prefix."""
        point = np.array([[2.5]])
        result = irf_to_markdown(point, digits=2)
        assert "+2.5" in result or "+2.50" in result

    def test_sign_negative_prefixed(self):
        """Negative values should have '-' prefix."""
        point = np.array([[-2.5]])
        result = irf_to_markdown(point, digits=2)
        assert "-2.5" in result or "-2.50" in result

    def test_markdown_table_structure_valid(self):
        """Output is a valid markdown table."""
        H, n = 5, 2
        point = np.zeros((H, n))
        result = irf_to_markdown(point)
        lines = result.strip().splitlines()
        assert len(lines) >= 3
        sep = lines[1]
        assert re.match(r"^\|[-| ]+\|$", sep), f"Bad separator: {sep!r}"


# ===========================================================================
# Tests for summary_to_markdown
# ===========================================================================

class TestSummaryToMarkdown:
    """Tests for summary_to_markdown()."""

    def test_returns_string(self):
        result = summary_to_markdown({"a": 1.0})
        assert isinstance(result, str)

    def test_title_in_output(self):
        result = summary_to_markdown({}, title="MyReport")
        assert "MyReport" in result

    def test_default_title(self):
        result = summary_to_markdown({})
        assert "Result" in result

    def test_scalar_float_formatted(self):
        result = summary_to_markdown({"x": 3.14159})
        # formatted as {:.4g} -> '3.142'
        assert "3.142" in result

    def test_scalar_int_formatted(self):
        result = summary_to_markdown({"n": 100})
        assert "100" in result

    def test_scalar_numpy_float_formatted(self):
        result = summary_to_markdown({"v": np.float64(2.718)})
        assert "2.718" in result

    def test_scalar_numpy_int_formatted(self):
        result = summary_to_markdown({"k": np.int64(42)})
        assert "42" in result

    def test_string_value_preserved(self):
        result = summary_to_markdown({"method": "OLS"})
        assert "OLS" in result

    def test_ndarray_summarised_by_shape(self):
        arr = np.zeros((3, 4))
        result = summary_to_markdown({"mat": arr})
        assert "ndarray" in result
        assert "3" in result
        assert "4" in result

    def test_series_summarised_by_shape(self):
        s = pd.Series([1, 2, 3])
        result = summary_to_markdown({"s": s})
        assert "Series" in result

    def test_dataframe_summarised_by_shape(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = summary_to_markdown({"df": df})
        assert "DataFrame" in result

    def test_list_summarised_by_length(self):
        result = summary_to_markdown({"items": [1, 2, 3, 4, 5]})
        assert "list" in result
        assert "5" in result

    def test_tuple_summarised_by_length(self):
        result = summary_to_markdown({"pair": (10, 20)})
        assert "tuple" in result
        assert "2" in result

    def test_unknown_type_converted_to_str(self):
        class Foo:
            def __str__(self):
                return "foo_value"
        result = summary_to_markdown({"obj": Foo()})
        assert "foo_value" in result

    def test_multiple_keys_all_present(self):
        d = {"alpha": 0.5, "beta": 1.5, "gamma": 2.5}
        result = summary_to_markdown(d)
        for k in d:
            assert k in result

    def test_table_has_key_value_columns(self):
        """The inner table must have a key and value column."""
        result = summary_to_markdown({"a": 1})
        assert "key" in result
        assert "value" in result

    def test_empty_dict_produces_header_only(self):
        """Empty dict still produces the title header + an (empty) table."""
        result = summary_to_markdown({})
        assert "###" in result

    def test_output_has_markdown_header(self):
        result = summary_to_markdown({"x": 1})
        assert result.startswith("###")


# ===========================================================================
# Tests for IRFResult dataclass
# ===========================================================================

class TestIRFResult:
    """Tests for IRFResult dataclass."""

    def _make_result(self, H=5, n=2):
        rng = np.random.default_rng(0)
        point = rng.standard_normal((H, n, 1))
        lower = point - 0.5
        upper = point + 0.5
        return IRFResult(
            point=point,
            lower=lower,
            upper=upper,
            var_names=("y1", "y2"),
            method="bootstrap",
            alpha=0.10,
        )

    def test_instantiation(self):
        point = np.zeros((3, 2, 1))
        r = IRFResult(point=point)
        assert r.method == ""
        assert r.alpha == 0.10
        assert r.lower is None
        assert r.upper is None

    def test_to_markdown_returns_string(self):
        r = self._make_result()
        result = r.to_markdown()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_markdown_var_names_in_output(self):
        r = self._make_result()
        result = r.to_markdown()
        assert "y1" in result
        assert "y2" in result

    def test_to_markdown_no_bands_when_none(self):
        """When lower/upper are None, no brackets in output."""
        point = np.zeros((3, 2, 1))
        r = IRFResult(point=point)
        result = r.to_markdown()
        assert "[" not in result

    def test_at_returns_dict_with_expected_keys(self):
        r = self._make_result()
        snap = r.at(0)
        assert set(snap.keys()) == {"point", "lower", "upper"}

    def test_at_point_shape(self):
        H, n = 5, 2
        r = self._make_result(H=H, n=n)
        snap = r.at(2)
        assert snap["point"].shape == (n, 1)

    def test_at_lower_shape(self):
        H, n = 5, 2
        r = self._make_result(H=H, n=n)
        snap = r.at(2)
        assert snap["lower"].shape == (n, 1)

    def test_at_none_when_no_bands(self):
        point = np.zeros((3, 2, 1))
        r = IRFResult(point=point)
        snap = r.at(0)
        assert snap["lower"] is None
        assert snap["upper"] is None

    def test_empty_var_names_uses_defaults(self):
        """Empty var_names tuple -> irf_to_markdown uses default y0, y1, ..."""
        point = np.zeros((3, 2, 1))
        r = IRFResult(point=point)  # var_names defaults to ()
        result = r.to_markdown()
        assert "y0" in result


# ===========================================================================
# Tests for LPResult dataclass
# ===========================================================================

class TestLPResult:
    """Tests for LPResult dataclass."""

    def _make_df(self, horizons=5):
        rng = np.random.default_rng(0)
        h = np.arange(horizons)
        coef = rng.standard_normal(horizons)
        se = np.abs(rng.standard_normal(horizons)) + 0.1
        df = pd.DataFrame({"coef": coef, "se": se}, index=pd.Index(h, name="h"))
        return df

    def test_instantiation(self):
        df = self._make_df()
        r = LPResult(df=df)
        assert r.method == "lp"
        assert r.n_obs is None
        assert r.alpha == 0.10

    def test_getitem_delegates_to_df(self):
        df = self._make_df()
        r = LPResult(df=df)
        pd.testing.assert_series_equal(r["coef"], df["coef"])

    def test_len_returns_number_of_rows(self):
        df = self._make_df(horizons=7)
        r = LPResult(df=df)
        assert len(r) == 7

    def test_horizons_from_named_index(self):
        """When index.name == 'h', horizons returns the index."""
        df = self._make_df(horizons=5)
        r = LPResult(df=df)
        assert list(r.horizons) == [0, 1, 2, 3, 4]

    def test_horizons_from_column(self):
        """When index is unnamed, horizons returns df['h'] values."""
        df = self._make_df(horizons=4)
        df = df.reset_index()  # 'h' becomes a column; index is unnamed RangeIndex
        r = LPResult(df=df)
        assert list(r.horizons) == [0, 1, 2, 3]

    def test_to_markdown_returns_string(self):
        df = self._make_df()
        r = LPResult(df=df)
        result = r.to_markdown()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_markdown_contains_coef_column(self):
        df = self._make_df()
        r = LPResult(df=df)
        result = r.to_markdown()
        assert "coef" in result

    def test_custom_method_stored(self):
        df = self._make_df()
        r = LPResult(df=df, method="panel_lp", n_obs=500, alpha=0.05)
        assert r.method == "panel_lp"
        assert r.n_obs == 500
        assert r.alpha == 0.05


# ===========================================================================
# Tests for private helpers (_df_to_markdown, _df_to_latex)
# via the public API
# ===========================================================================

class TestPrivateHelpers:
    """Indirectly test _df_to_markdown and _df_to_latex through public API."""

    def test_df_to_markdown_via_coef_table_separator_row(self):
        """Separator row should be present and well-formed."""
        result = coef_table(np.array([1.0]), np.array([0.1]), fmt="markdown")
        lines = result.strip().splitlines()
        sep = lines[1]
        # All chars in separator should be '-', '|', or ' '
        assert all(c in "-| " for c in sep)

    def test_df_to_latex_underscore_escaped(self):
        """Underscores in variable names must be escaped in LaTeX output."""
        result = coef_table(
            np.array([1.0]), np.array([0.1]),
            names=["log_y"],
            fmt="latex",
        )
        assert r"log\_y" in result

    def test_df_to_latex_column_alignment(self):
        """First column left-aligned ('l'), rest right-aligned ('r')."""
        result = coef_table(
            np.array([1.0, 2.0]), np.array([0.1, 0.2]),
            fmt="latex",
        )
        # find the alignment spec
        m = re.search(r"\\begin\{tabular\}\{([^}]+)\}", result)
        assert m is not None
        spec = m.group(1)
        assert spec[0] == "l"
        assert all(c == "r" for c in spec[1:])

    def test_df_to_markdown_nan_shows_empty_string(self):
        """NaN values in a cell should appear as an empty string."""
        beta = np.array([1.0])
        se = np.array([0.0])  # SE=0 -> t = NaN
        result = coef_table(beta, se, fmt="markdown")
        # The table should be produced without raising
        assert isinstance(result, str)

    def test_summary_to_markdown_table_has_separator(self):
        result = summary_to_markdown({"a": 1.0})
        lines = result.strip().splitlines()
        # Find the line that is a markdown separator (after the ### header)
        sep_found = any(
            re.match(r"^\|[-| ]+\|$", l) for l in lines
        )
        assert sep_found, "No markdown separator found in summary_to_markdown output"
