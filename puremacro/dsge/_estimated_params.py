"""The Dynare ``estimated_params`` block grammar.

Dynare declares what to estimate, and how, in three blocks:

.. code-block:: text

    estimated_params;
      // NAME, INITVAL, LB, UB, PRIOR_SHAPE, P1, P2, P3, P4, JSCALE;
      alpha, 0.35, beta_pdf, 0.35, 0.02;
      rho, , 0, 1, beta_pdf, 0.5, 0.2;
      stderr eps_a, inv_gamma_pdf, 0.1, 2;
      corr eps_a, eps_b, normal_pdf, 0, 0.2;
    end;
    estimated_params_init;   ... end;
    estimated_params_bounds; ... end;

Parsing it is not a matter of splitting on commas. The grammar is
**positionally ambiguous** — ``NAME, 0.35, beta_pdf, ...`` and ``NAME,
beta_pdf, ...`` differ only by a leading field — and ``corr a, b, ...`` carries
a comma *inside* its target. The rule implemented here resolves both without
lookahead heuristics:

1. consume any leading ``stderr`` / ``corr`` keyword together with its one or
   two identifiers, so the ``corr`` comma is gone before any comma splitting;
2. split what remains on commas and find the first field that is a known
   ``*_pdf`` shape, **case-insensitively** (``sw07_pfeifer.mod`` writes
   ``INV_GAMMA_PDF`` while the manual writes ``inv_gamma_pdf``; a
   case-sensitive match would silently reclassify every statement as
   "estimated without a prior");
3. everything before the shape is ``[]``, ``[INITVAL]`` or
   ``[INITVAL, LB, UB]`` **by length** — any other length is an error quoting
   the statement verbatim rather than a guess;
4. everything after it is ``P1, P2, P3, P4, JSCALE`` by position.

Naming follows Dynare's display convention so output is recognisable:
``SE_<shock>`` for a structural standard error, ``CORR_<s1>_<s2>`` for a
correlation, and ``ME_<obsvar>`` for a measurement error (Dynare writes
``SE_EOBS_<var>``; the shorter form is what puremacro reports).

``kind`` is what makes estimation affordable: **only** ``kind == "param"``
requires the model to be re-solved at a new draw. The other three write
straight into the innovation covariance ``Q`` or the measurement covariance
``H``.

Declared ``LB``/``UB`` are kept verbatim rather than intersected with the
prior's own support: what the file says is what the sampler is told. A draw
outside the prior's support still scores ``-inf``, which the estimator already
handles with a finite optimiser penalty.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from puremacro.dsge.priors import (
    BetaPrior,
    GammaPrior,
    InvGammaPrior,
    NormalPrior,
    Prior,
    UniformPrior,
    WeibullPrior,
)

__all__ = [
    "EstimatedParamSpec",
    "EstimatedParams",
    "parse_estimated_params",
    "parse_estimated_params_init",
    "parse_estimated_params_bounds",
]


_KINDS = ("param", "stderr_shock", "corr_shock", "stderr_obs")

# Dynare's PRIOR_SHAPE keywords, lowercased. dirichlet_pdf is recognised so it
# can be refused by name rather than silently read as "no prior".
_SHAPES = frozenset({
    "beta_pdf", "gamma_pdf", "normal_pdf", "uniform_pdf", "weibull_pdf",
    "inv_gamma_pdf", "inv_gamma1_pdf", "inv_gamma2_pdf", "dirichlet_pdf",
})
_UNSUPPORTED_SHAPES = frozenset({"dirichlet_pdf"})

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


def _strip_comments(text: str) -> str:
    """Strip ``/* */``, ``//`` and ``%`` comments.

    Local rather than imported from :mod:`puremacro.dsge.dynare`, which imports
    this module: a direct call may hand us raw source, and a lazy import back
    into ``dynare`` would only add an import-order hazard for four lines.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"(//|%).*$", "", text, flags=re.MULTILINE)


def _block_body(text: str, block: str) -> str | None:
    """The inside of ``<block>[(...)]; ... end;``, or ``None`` if absent.

    The block must actually be present: an earlier draft fell back to treating
    the whole argument as a bare body, which turns a .mod file with no
    ``estimated_params`` block into a pile of nonsense statements instead of an
    empty result. ``\b`` on both ends keeps ``estimated_params`` from matching
    ``estimated_params_init`` (``_`` is a word character, so the trailing
    boundary fails there).
    """
    clean = _strip_comments(text)
    m = re.search(
        rf"\b{block}\s*(?:\([^)]*\))?\s*;(.*?)\bend\s*;", clean, re.DOTALL | re.IGNORECASE
    )
    return m.group(1) if m is not None else None


def _num(token: str) -> float | None:
    """A numeric field, or ``None`` for an omitted one (``NAME, , 0, 1, ...``)."""
    token = token.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        raise ValueError(f"expected a number, got {token!r}") from None


@dataclass(frozen=True)
class EstimatedParamSpec:
    """One statement of an ``estimated_params`` block.

    Attributes
    ----------
    kind : {'param', 'stderr_shock', 'corr_shock', 'stderr_obs'}
        What the statement estimates. Only ``'param'`` needs a model re-solve.
    target : tuple[str, ...]
        The declared object: ``('alpha',)``, ``('eps_a',)`` or
        ``('eps_a', 'eps_b')``.
    name : str
        Reporting name: ``'alpha'``, ``'SE_eps_a'``, ``'CORR_eps_a_eps_b'``,
        ``'ME_dy'``.
    prior : Prior or None
        ``None`` when the statement declares no ``PRIOR_SHAPE`` (Dynare reads
        that as maximum likelihood).
    init : float or None
        ``INITVAL``, or ``None`` when omitted.
    lb, ub : float
        Declared truncation, defaulting to the prior's own support.
    jscale : float
        Per-parameter proposal scaling (Dynare's tenth field).
    """

    kind: str
    target: tuple
    name: str
    prior: Prior | None
    init: float | None
    lb: float
    ub: float
    jscale: float = 1.0

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {list(_KINDS)}")

    @property
    def start(self) -> float:
        """``init`` if declared, else the prior mean, else the bound midpoint."""
        if self.init is not None:
            return float(self.init)
        if self.prior is not None and math.isfinite(self.prior.mean):
            return float(self.prior.mean)
        lo = self.lb if math.isfinite(self.lb) else 0.0
        hi = self.ub if math.isfinite(self.ub) else lo + 1.0
        return 0.5 * (lo + hi)


@dataclass(frozen=True)
class EstimatedParams:
    """The parsed block: an ordered tuple of :class:`EstimatedParamSpec`."""

    specs: tuple

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self):
        return iter(self.specs)

    def names(self) -> tuple:
        return tuple(s.name for s in self.specs)

    def by_kind(self, kind: str) -> tuple:
        if kind not in _KINDS:
            raise ValueError(f"unknown kind {kind!r}; expected one of {list(_KINDS)}")
        return tuple(s for s in self.specs if s.kind == kind)

    def by_name(self, name: str) -> EstimatedParamSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(f"{name!r} is not an estimated parameter; have {list(self.names())}")

    def priors(self) -> dict:
        """``{name: Prior}`` for every statement that declared a shape.

        The declared ``LB``/``UB`` are written onto the returned prior so that
        ``param_bounds`` and the sampler see the file's truncation.
        """
        out = {}
        for s in self.specs:
            if s.prior is None:
                continue
            spec = dict(s.prior.to_dict())
            spec["lb"], spec["ub"] = s.lb, s.ub
            from puremacro.dsge.priors import ensure_prior

            out[s.name] = ensure_prior(spec)
        return out

    def initial_params(self) -> dict:
        return {s.name: s.start for s in self.specs}

    def jscale_vector(self, names: Sequence[str]) -> np.ndarray:
        lookup = {s.name: s.jscale for s in self.specs}
        return np.array([lookup.get(n, 1.0) for n in names], dtype=float)


def _build_prior(shape: str, p: list, statement: str) -> Prior:
    """Map a ``PRIOR_SHAPE`` plus ``P1..P4`` onto a :class:`Prior`.

    ``P1``/``P2`` are the mean and standard deviation of the actual variable in
    every family (Dynare's ``*_specification`` helpers); ``P3``/``P4`` shift and
    bound its support.
    """
    p1, p2, p3, p4 = (p + [None, None, None, None])[:4]
    if shape in _UNSUPPORTED_SHAPES:
        raise ValueError(
            f"prior shape {shape!r} is not implemented (statement: {statement!r}); "
            f"supported: {sorted(_SHAPES - _UNSUPPORTED_SHAPES)}"
        )
    if shape == "beta_pdf":
        shift = 0.0 if p3 is None else p3
        upper = 1.0 if p4 is None else p4
        return BetaPrior(mean=p1, std=p2, shift=shift, scale=upper - shift)
    if shape == "gamma_pdf":
        kw = {} if p4 is None else {"ub": p4}
        return GammaPrior(mean=p1, std=p2, shift=0.0 if p3 is None else p3, **kw)
    if shape == "normal_pdf":
        lo = -math.inf if p3 is None else p3
        hi = math.inf if p4 is None else p4
        return NormalPrior(mean=p1, std=p2, lb=lo, ub=hi)
    if shape in ("inv_gamma_pdf", "inv_gamma1_pdf", "inv_gamma2_pdf"):
        kind = "type2" if shape == "inv_gamma2_pdf" else "type1"
        lo = 1e-4 if p3 is None else p3
        hi = math.inf if p4 is None else p4
        return InvGammaPrior(mean=p1, std=p2, lb=lo, ub=hi, kind=kind)
    if shape == "uniform_pdf":
        if p3 is not None and p4 is not None:
            return UniformPrior(lb=p3, ub=p4)
        if p1 is None or p2 is None:
            raise ValueError(
                f"uniform_pdf needs either (P1, P2) as mean/std or (P3, P4) as "
                f"bounds (statement: {statement!r})"
            )
        half = 0.5 * math.sqrt(12.0) * p2
        return UniformPrior(lb=p1 - half, ub=p1 + half)
    if shape == "weibull_pdf":
        return WeibullPrior(mean=p1, std=p2, shift=0.0 if p3 is None else p3)
    raise ValueError(  # pragma: no cover - guarded by _SHAPES membership
        f"unhandled prior shape {shape!r} (statement: {statement!r})"
    )


def _split_target(statement: str) -> tuple[str, tuple, str]:
    """``(kind_hint, target, remainder)`` — step 1 of the grammar.

    Consumes any ``stderr`` / ``corr`` keyword and its identifiers *before* the
    remainder is split on commas, which is what keeps ``corr a, b`` from being
    mistaken for two fields.
    """
    m = re.match(rf"\s*corr\s+({_IDENT})\s*,\s*({_IDENT})\s*(?:,(.*))?$", statement,
                 re.DOTALL | re.IGNORECASE)
    if m:
        return "corr", (m.group(1), m.group(2)), m.group(3) or ""
    m = re.match(rf"\s*stderr\s+({_IDENT})\s*(?:,(.*))?$", statement,
                 re.DOTALL | re.IGNORECASE)
    if m:
        return "stderr", (m.group(1),), m.group(2) or ""
    m = re.match(rf"\s*({_IDENT})\s*(?:,(.*))?$", statement, re.DOTALL)
    if m:
        return "param", (m.group(1),), m.group(2) or ""
    raise ValueError(f"could not read a parameter name from {statement.strip()!r}")


def parse_estimated_params(
    text: str,
    *,
    shocks: Sequence[str],
    varobs: Sequence[str] | None = None,
    params: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
) -> EstimatedParams:
    """Parse an ``estimated_params`` block.

    Parameters
    ----------
    text : str
        The whole block (``estimated_params; ... end;``) or just its body.
    shocks : Sequence[str]
        Declared ``varexo`` names, used to classify a ``stderr`` target.
    varobs : Sequence[str], optional
        Declared observables; a ``stderr`` of one of these is a measurement
        error rather than a structural standard deviation.
    params : Sequence[str], optional
        Declared parameter names. When given, a bare target that is not among
        them (nor among ``variables``) is an error naming it, rather than a
        parameter the model will never read.
    variables : Sequence[str], optional
        Declared endogenous variables, accepted as targets for the
        ``estimated_params`` forms that name one.

    Returns
    -------
    EstimatedParams
    """
    body = _block_body(text, "estimated_params")
    if body is None:
        return EstimatedParams(specs=())

    shocks = list(shocks)
    varobs = list(varobs or [])
    known_params = set(params or [])
    known_vars = set(variables or [])

    specs: list[EstimatedParamSpec] = []
    for raw in body.split(";"):
        statement = raw.strip()
        if not statement:
            continue

        kind_hint, target, remainder = _split_target(statement)

        # -- classify the target ------------------------------------------
        if kind_hint == "corr":
            for t in target:
                if t not in shocks:
                    raise ValueError(
                        f"corr target {t!r} is not a declared varexo "
                        f"(statement: {statement!r}); have {shocks}"
                    )
            kind, name = "corr_shock", f"CORR_{target[0]}_{target[1]}"
        elif kind_hint == "stderr":
            t = target[0]
            if t in shocks:
                kind, name = "stderr_shock", f"SE_{t}"
            elif t in varobs:
                kind, name = "stderr_obs", f"ME_{t}"
            else:
                raise ValueError(
                    f"stderr target {t!r} is neither a declared varexo nor a "
                    f"varobs (statement: {statement!r}); varexo={shocks}, "
                    f"varobs={varobs}"
                )
        else:
            t = target[0]
            if (known_params or known_vars) and t not in known_params and t not in known_vars:
                raise ValueError(
                    f"estimated parameter {t!r} is not a declared parameter or "
                    f"variable (statement: {statement!r})"
                )
            kind, name = "param", t

        # -- steps 2-4: locate the shape, then read by position -------------
        fields = [f.strip() for f in remainder.split(",")] if remainder.strip() else []
        shape_at = next(
            (i for i, f in enumerate(fields) if f.lower() in _SHAPES), None
        )
        if shape_at is None:
            head, shape, tail = fields, None, []
        else:
            head = fields[:shape_at]
            shape = fields[shape_at].lower()
            tail = fields[shape_at + 1:]

        if len(head) == 0:
            init, lo, hi = None, None, None
        elif len(head) == 1:
            init, lo, hi = _num(head[0]), None, None
        elif len(head) == 3:
            init, lo, hi = _num(head[0]), _num(head[1]), _num(head[2])
        else:
            raise ValueError(
                f"cannot read {statement.strip()!r}: the fields before the prior "
                f"shape must be none, INITVAL, or INITVAL, LB, UB — got "
                f"{len(head)} ({head})"
            )

        if shape is None:
            prior = None
            jscale = 1.0
        else:
            p = [_num(f) for f in tail[:4]]
            prior = _build_prior(shape, p, statement)
            jscale_field = tail[4] if len(tail) > 4 else ""
            jscale = _num(jscale_field)
            jscale = 1.0 if jscale is None else jscale
            if len(tail) > 5:
                raise ValueError(
                    f"cannot read {statement.strip()!r}: at most P1..P4 and "
                    f"JSCALE may follow the prior shape — got {len(tail)} fields"
                )

        if lo is None:
            lo = prior.lb if prior is not None else -math.inf
        if hi is None:
            hi = prior.ub if prior is not None else math.inf

        specs.append(EstimatedParamSpec(
            kind=kind, target=tuple(target), name=name, prior=prior,
            init=init, lb=float(lo), ub=float(hi), jscale=float(jscale),
        ))

    return EstimatedParams(specs=tuple(specs))


def _named(kind_hint: str, target: tuple, shocks, varobs) -> str:
    if kind_hint == "corr":
        return f"CORR_{target[0]}_{target[1]}"
    if kind_hint == "stderr":
        return f"ME_{target[0]}" if target[0] in (varobs or []) else f"SE_{target[0]}"
    return target[0]


def parse_estimated_params_init(
    text: str, *, shocks: Sequence[str] | None = None,
    varobs: Sequence[str] | None = None,
) -> dict:
    """Parse ``estimated_params_init;`` into ``{name: initial value}``.

    Without ``varobs`` a ``stderr`` target is named ``SE_<t>``; pass the
    declared observables to have a measurement error named ``ME_<t>`` as in
    the main block.
    """
    body = _block_body(text, "estimated_params_init")
    if body is None:
        return {}
    out: dict = {}
    for raw in body.split(";"):
        statement = raw.strip()
        if not statement or re.match(r"^\s*use_calibration\s*$", statement, re.I):
            continue
        kind_hint, target, remainder = _split_target(statement)
        fields = [f.strip() for f in remainder.split(",") if f.strip()]
        if not fields:
            raise ValueError(f"estimated_params_init: no value in {statement!r}")
        out[_named(kind_hint, target, shocks, varobs)] = _num(fields[0])
    return out


def parse_estimated_params_bounds(
    text: str, *, shocks: Sequence[str] | None = None,
    varobs: Sequence[str] | None = None,
) -> dict:
    """Parse ``estimated_params_bounds;`` into ``{name: (lb, ub)}``."""
    body = _block_body(text, "estimated_params_bounds")
    if body is None:
        return {}
    out: dict = {}
    for raw in body.split(";"):
        statement = raw.strip()
        if not statement:
            continue
        kind_hint, target, remainder = _split_target(statement)
        fields = [f.strip() for f in remainder.split(",")]
        if len(fields) != 2:
            raise ValueError(
                f"estimated_params_bounds: {statement!r} needs exactly LB, UB — "
                f"got {len(fields)} fields"
            )
        lo, hi = _num(fields[0]), _num(fields[1])
        out[_named(kind_hint, target, shocks, varobs)] = (
            -math.inf if lo is None else lo,
            math.inf if hi is None else hi,
        )
    return out
