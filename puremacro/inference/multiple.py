"""Multiple-testing p-value corrections (Holm, Sidak, Bonferroni, BH, BY, GBS).

A pure-numpy drop-in for ``statsmodels.stats.multitest.multipletests`` and
``statsmodels.stats.multitest.fdrcorrection``: same signature, same 4-tuple,
same method names and aliases, same numbers.

The two steps implementations habitually get wrong are both about
*monotonicity enforcement*, and they run in opposite directions:

* **step-down** families (Holm, Holm-Sidak) take a running **maximum** over
  the p-values sorted *ascending*, so a large early p-value drags every
  later one up with it;
* **step-up** families (Benjamini-Hochberg, Benjamini-Yekutieli,
  Simes-Hochberg) take a running **minimum** over the *reversed* ascending
  order, so a small late p-value drags every earlier one down.

Get the direction or the reversal wrong and the corrected p-values are still
plausible-looking and monotone — just wrong. The parity tests in
``tests/test_inference_extras_parity.py`` sweep every method against
statsmodels for exactly this reason.

Notes
-----
The implementation deliberately mirrors the *arithmetic* of statsmodels
0.14.6 (``statsmodels/stats/multitest.py:63`` for ``multipletests``,
``:280`` for ``fdrcorrection``) expression by expression rather than
re-deriving each procedure from its paper. Two procedures have several
algebraically equivalent forms that differ in the last bits of a float —
Sidak's ``-expm1(n * log1p(-p))`` versus ``1 - (1 - p) ** n`` is the
canonical example, and statsmodels picks the numerically stable one — so
"same formula" is what buys exact parity, not "same theorem".

References
----------
Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics 6(2), 65-70.
Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery
    rate. JRSS-B 57(1), 289-300.
Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery
    rate in multiple testing under dependency. Annals of Statistics 29(4).
Gavrilov, Y., Benjamini, Y. and Sarkar, S. K. (2009). An adaptive step-down
    procedure with proven FDR control under independence. Annals of
    Statistics 37(2), 619-629.
Sidak, Z. (1967). Rectangular confidence regions for the means of
    multivariate normal distributions. JASA 62(318), 626-633.
Simes, R. J. (1986). An improved Bonferroni procedure for multiple tests of
    significance. Biometrika 73(3), 751-754.
Hommel, G. (1988). A stagewise rejective multiple test procedure based on a
    modified Bonferroni test. Biometrika 75(2), 383-386.
"""
from __future__ import annotations

import numpy as np

__all__ = ["multipletests", "fdrcorrection"]


# Method aliases, verbatim from statsmodels 0.14.6
# (``statsmodels/stats/multitest.py:_alias_list``). Kept as an explicit map
# rather than a set of ``in`` checks so an unknown-but-plausible spelling
# ("fdr_bh2", "holms") fails loudly instead of falling through to a default.
_ALIASES = {
    "b": "bonferroni", "bonf": "bonferroni", "bonferroni": "bonferroni",
    "s": "sidak", "sidak": "sidak",
    "h": "holm", "holm": "holm",
    "hs": "holm-sidak", "holm-sidak": "holm-sidak",
    "sh": "simes-hochberg", "simes-hochberg": "simes-hochberg",
    "ho": "hommel", "hommel": "hommel",
    "fdr_bh": "fdr_bh", "fdr_i": "fdr_bh", "fdr_p": "fdr_bh",
    "fdri": "fdr_bh", "fdrp": "fdr_bh",
    "fdr_by": "fdr_by", "fdr_n": "fdr_by", "fdr_c": "fdr_by",
    "fdrn": "fdr_by", "fdrcorr": "fdr_by",
    # Gavrilov-Benjamini-Sarkar adaptive step-down. statsmodels gives it no
    # aliases, so neither does this map.
    "fdr_gbs": "fdr_gbs",
}

# Recognised by statsmodels but not reimplemented here. Named separately so
# the error message can say "known method, not implemented" rather than the
# misleading "method not recognized". Membership is exactly the two-stage
# family that ``fdrcorrection_twostage`` serves — the only two methods for
# which statsmodels' ``maxiter`` does anything. ``fdr_gbs`` was here once and
# does not belong: it is a single-pass closed form, not two-stage, and it is
# implemented below.
_TWO_STAGE = {
    "fdr_tsbh": "fdr_tsbh", "fdr_2sbh": "fdr_tsbh",
    "fdr_tsbky": "fdr_tsbky", "fdr_2sbky": "fdr_tsbky",
    "fdr_twostage": "fdr_tsbky",
}


def multipletests(pvals, alpha=0.05, method="hs", maxiter=1, is_sorted=False,
                  returnsorted=False):
    """Correct a family of p-values for multiple testing.

    Drop-in replacement for
    ``statsmodels.stats.multitest.multipletests`` (statsmodels 0.14.6),
    returning the identical 4-tuple.

    Parameters
    ----------
    pvals : array_like, 1-D
        Uncorrected p-values. A pandas Series is accepted; the return values
        are always ndarrays, as in statsmodels.
    alpha : float, default 0.05
        Family-wise error rate (or, for the FDR methods, the target FDR).
        Only ``reject`` depends on it: for every method implemented here the
        corrected p-values themselves are free of ``alpha``, so they may be
        compared against a different threshold afterwards.
    method : str, default ``'hs'``
        One of ``'bonferroni'``, ``'sidak'``, ``'holm'``, ``'holm-sidak'``,
        ``'simes-hochberg'``, ``'hommel'``, ``'fdr_bh'``, ``'fdr_by'``,
        ``'fdr_gbs'``, or any of the statsmodels aliases (``'b'``, ``'s'``,
        ``'h'``, ``'hs'``, ``'sh'``, ``'ho'``, ``'bonf'``, ``'fdr_i'``,
        ``'fdr_p'``, ``'fdri'``, ``'fdrp'``, ``'fdr_n'``, ``'fdr_c'``,
        ``'fdrn'``, ``'fdrcorr'``). Case-insensitive, as in statsmodels.
    maxiter : int or bool, default 1
        Iteration cap for the two-stage FDR procedures ``'fdr_tsbh'`` and
        ``'fdr_tsbky'``, **ignored by every other method** — including
        every method implemented here. It occupies the fourth positional
        slot because that is where statsmodels puts it (statsmodels 0.14
        inserted it between ``method`` and ``is_sorted``), and a
        replacement that omitted it would silently reinterpret the fourth
        positional argument of ``multipletests(p, 0.05, 'holm', 1)`` as
        ``is_sorted`` and return unsorted-order nonsense. Since the
        two-stage family is not implemented here (see Raises), no
        implemented method's answer depends on this argument.
    is_sorted : bool, default False
        If False the p-values are sorted internally and the results are
        returned in the *original* order. If True the caller promises
        ``pvals`` is already ascending, and the results stay in that order.
    returnsorted : bool, default False
        If True the results are returned in ascending-p order even when
        ``is_sorted`` is False.

    Returns
    -------
    reject : ndarray of bool
        True where the hypothesis is rejected at ``alpha``.
    pvals_corrected : ndarray
        Multiplicity-adjusted p-values, clipped at 1.
    alphacSidak : float
        ``1 - (1 - alpha) ** (1 / n)`` — the Sidak-corrected per-test level.
    alphacBonf : float
        ``alpha / n`` — the Bonferroni-corrected per-test level.

    Raises
    ------
    ValueError
        If ``pvals`` is not 1-D, if it is empty, or if ``method`` is not a
        recognised statsmodels method name.
    NotImplementedError
        If ``method`` names a statsmodels procedure that exists but is not
        reimplemented here. That is exactly the two-stage FDR family —
        ``'fdr_tsbh'`` and ``'fdr_tsbky'`` with their aliases — whose
        corrected p-values are specific to the ``alpha`` they were computed
        at and whose iteration is what ``maxiter`` controls.

    Notes
    -----
    **Ordering with ties.** The sort is ``np.argsort(pvals)`` with numpy's
    default (introsort, *not* stable), exactly as statsmodels does it, so
    tied p-values are permuted identically to statsmodels. Since every
    method here is a function of the sorted vector and the inverse
    permutation, the corrected values are tie-order-invariant anyway; the
    match is exact regardless.

    **``reject`` is not ``pvals_corrected <= alpha``.** For the step-down
    methods statsmodels derives ``reject`` from the raw p-values against the
    step-down threshold and then floods the tail, which is equivalent but
    computed separately. Both are returned; do not re-derive one from the
    other.

    **Empty input.** statsmodels raises ``ZeroDivisionError`` from
    ``1. / ntests``; this function raises ``ValueError`` naming the problem
    instead. This is the one documented behavioural divergence, and it
    concerns only a degenerate call.

    **Two-dimensional input.** statsmodels silently produces nonsense
    (``len(pvals)`` counts rows); this function raises ``ValueError``.

    **``fdr_gbs`` is a step-down procedure, not a Benjamini-Hochberg-style
    adjustment.** It is the adaptive step-down of Gavrilov,
    Benjamini and Sarkar (2009), whose statistic is the *odds* ``p/(1-p)``
    scaled by ``(n + 1 - i) / i``. The odds ratio makes it the one method
    here that is not bounded above by ``n * p``: a p-value of exactly 1
    divides by zero and the adjusted value is ``inf`` before the final clip
    to 1, in statsmodels and here alike. The running maximum comes first
    (the step-down monotonicity) and the running minimum over the reversed
    order second; both are needed, and dropping either leaves a plausible
    non-monotone vector.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    >>> reject, p_adj, _, _ = multipletests(p, alpha=0.05, method="holm")
    >>> np.round(p_adj, 4)
    array([0.008, 0.056, 0.234, 0.234, 0.234, 0.234, 0.234, 0.234])
    >>> reject
    array([ True, False, False, False, False, False, False, False])

    The corrected p-values are non-decreasing in the sorted order even
    though the raw increments are not — that is the step-down monotonicity
    enforcement:

    >>> _, p_bh, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    >>> np.round(p_bh, 4)
    array([0.008 , 0.032 , 0.0672, 0.0672, 0.0672, 0.08  , 0.0846, 0.205 ])
    """
    pvals = np.asarray(pvals)
    if pvals.ndim != 1:
        raise ValueError(
            f"multipletests: pvals must be 1-D, got shape {pvals.shape}. "
            "Flatten or loop over rows explicitly."
        )
    if pvals.size == 0:
        raise ValueError("multipletests: pvals is empty; nothing to correct.")

    key = str(method).lower()
    if key in _TWO_STAGE:
        raise NotImplementedError(
            f"multipletests: method {method!r} ({_TWO_STAGE[key]}) is a "
            "statsmodels two-stage procedure that puremacro does not "
            "implement; its corrected p-values are specific to the alpha "
            "they were computed at, unlike every method that is. "
            "Implemented: bonferroni, sidak, holm, "
            "holm-sidak, simes-hochberg, hommel, fdr_bh, fdr_by, fdr_gbs."
        )
    if key not in _ALIASES:
        raise ValueError("method not recognized")
    name = _ALIASES[key]

    if not is_sorted:
        sortind = np.argsort(pvals)
        pvals = np.take(pvals, sortind)

    ntests = len(pvals)
    alphacSidak = 1 - np.power((1.0 - alpha), 1.0 / ntests)
    alphacBonf = alpha / float(ntests)

    if name == "bonferroni":
        reject = pvals <= alphacBonf
        pvals_corrected = pvals * float(ntests)

    elif name == "sidak":
        reject = pvals <= alphacSidak
        # -expm1(n*log1p(-p)) rather than 1-(1-p)**n: identical in exact
        # arithmetic, but the log1p/expm1 pair keeps full relative precision
        # for the tiny p-values that dominate a corrected family.
        pvals_corrected = -np.expm1(ntests * np.log1p(-pvals))

    elif name == "holm-sidak":
        alphacSidak_all = 1 - np.power((1.0 - alpha),
                                       1.0 / np.arange(ntests, 0, -1))
        notreject = pvals > alphacSidak_all
        reject = _stepdown_reject(notreject, ntests)
        pvals_corrected_raw = -np.expm1(np.arange(ntests, 0, -1) *
                                        np.log1p(-pvals))
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)

    elif name == "holm":
        notreject = pvals > alpha / np.arange(ntests, 0, -1)
        reject = _stepdown_reject(notreject, ntests)
        pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
        # Step-down: running MAXIMUM, forwards over ascending p.
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)

    elif name == "simes-hochberg":
        alphash = alpha / np.arange(ntests, 0, -1)
        reject = pvals <= alphash
        rejind = np.nonzero(reject)
        if rejind[0].size > 0:
            rejectmax = np.max(np.nonzero(reject))
            reject[:rejectmax] = True
        pvals_corrected_raw = np.arange(ntests, 0, -1) * pvals
        # Step-up: running MINIMUM, backwards over ascending p.
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]

    elif name == "hommel":
        # Closed testing over all n partitions; O(n^2) and slow for large n,
        # exactly as statsmodels warns. ``pvals`` is already ascending here.
        a = pvals.copy()
        for m in range(ntests, 1, -1):
            cim = np.min(m * pvals[-m:] / np.arange(1, m + 1.0))
            a[-m:] = np.maximum(a[-m:], cim)
            a[:-m] = np.maximum(a[:-m], np.minimum(m * pvals[:-m], cim))
        pvals_corrected = a
        reject = a <= alpha

    elif name == "fdr_bh":
        reject, pvals_corrected = fdrcorrection(pvals, alpha=alpha,
                                                method="indep", is_sorted=True)

    elif name == "fdr_by":
        reject, pvals_corrected = fdrcorrection(pvals, alpha=alpha,
                                                method="n", is_sorted=True)

    else:  # name == "fdr_gbs"
        # Adaptive step-down of Gavrilov, Benjamini and Sarkar (2009).
        # ``pvals`` is ascending here. The statistic is an odds ratio, so
        # p == 1 divides by zero and gives inf; numpy warns and statsmodels
        # lets the warning through, which is reproduced rather than
        # suppressed — the caller's warning filter should decide.
        ii = np.arange(1, ntests + 1)
        q = (ntests + 1.0 - ii) / ii * pvals / (1.0 - pvals)
        # Step-down first (running maximum, ascending), then the step-up
        # reversal. Both, in this order; see Notes.
        pvals_corrected_raw = np.maximum.accumulate(q)
        pvals_corrected = np.minimum.accumulate(
            pvals_corrected_raw[::-1])[::-1]
        # Unlike every other branch, ``reject`` here is read off the
        # corrected values — and off the *unclipped* ones, because
        # statsmodels compares before its final clip at 1. That only
        # matters for alpha >= 1, but it is free to get right.
        reject = pvals_corrected <= alpha

    pvals_corrected[pvals_corrected > 1] = 1
    if is_sorted or returnsorted:
        return reject, pvals_corrected, alphacSidak, alphacBonf

    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[sortind] = pvals_corrected
    reject_ = np.empty_like(reject)
    reject_[sortind] = reject
    return reject_, pvals_corrected_, alphacSidak, alphacBonf


def _stepdown_reject(notreject, ntests):
    """Flood a step-down rejection mask from its first failure onwards.

    Holm and Holm-Sidak reject in ascending-p order and stop dead at the
    first hypothesis that fails its threshold: everything after it is
    accepted whether or not it clears its own (looser) threshold. This
    helper implements that "stop at the first failure" rule, which is what
    makes the procedure *sequentially rejective* rather than a per-test
    comparison.

    Parameters
    ----------
    notreject : ndarray of bool
        ``pvals_sorted > threshold``, elementwise, in ascending-p order.
    ntests : int
        Length of the family.

    Returns
    -------
    ndarray of bool
        The rejection mask, in ascending-p order.
    """
    nr_index = np.nonzero(notreject)[0]
    if nr_index.size == 0:
        # No failure anywhere: every hypothesis is rejected. Indexing at
        # ntests is a no-op slice, which is the point.
        notrejectmin = ntests
    else:
        notrejectmin = np.min(nr_index)
    notreject[notrejectmin:] = True
    return ~notreject


def fdrcorrection(pvals, alpha=0.05, method="indep", is_sorted=False):
    """Benjamini-Hochberg / Benjamini-Yekutieli false-discovery-rate control.

    Drop-in replacement for
    ``statsmodels.stats.multitest.fdrcorrection`` (statsmodels 0.14.6).

    Parameters
    ----------
    pvals : array_like, 1-D
        Uncorrected p-values.
    alpha : float, default 0.05
        Target false discovery rate.
    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, default 'indep'
        ``'i' / 'indep' / 'p' / 'poscorr'`` give Benjamini-Hochberg (valid
        under independence or positive regression dependence);
        ``'n' / 'negcorr'`` give Benjamini-Yekutieli, which divides the
        step-up factor by the harmonic number ``c(n) = sum_{i<=n} 1/i`` and
        is therefore valid under arbitrary dependence.
    is_sorted : bool, default False
        If False the p-values are sorted internally and results are returned
        in the original order.

    Returns
    -------
    reject : ndarray of bool
        True where the hypothesis is rejected.
    pvals_corrected : ndarray
        FDR-adjusted p-values (q-values), clipped at 1.

    Raises
    ------
    ValueError
        If ``pvals`` is not 1-D, if it is empty, or if ``method`` is not one
        of the six accepted strings.

    Notes
    -----
    The adjustment is ``minimum.accumulate`` over the *reversed* ascending
    order — a step-up procedure. Reversing only one of the two reversals is
    the classic bug: it yields a monotone-looking vector that is
    systematically too small at the significant end.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([0.01, 0.02, 0.03, 0.9])
    >>> reject, q = fdrcorrection(p, alpha=0.05)
    >>> np.round(q, 4)
    array([0.04, 0.04, 0.04, 0.9 ])
    >>> reject
    array([ True,  True,  True, False])
    """
    pvals = np.asarray(pvals)
    if pvals.ndim != 1:
        raise ValueError(
            f"fdrcorrection: pvals must be 1-D, got shape {pvals.shape}."
        )
    if pvals.size == 0:
        raise ValueError("fdrcorrection: pvals is empty; nothing to correct.")

    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias, as in statsmodels

    n = len(pvals_sorted)
    if method in ["i", "indep", "p", "poscorr"]:
        ecdffactor = np.arange(1, n + 1) / float(n)
    elif method in ["n", "negcorr"]:
        cm = np.sum(1.0 / np.arange(1, n + 1))
        ecdffactor = (np.arange(1, n + 1) / float(n)) / cm
    else:
        raise ValueError("only indep and negcorr implemented")

    reject = pvals_sorted <= ecdffactor * alpha
    if reject.any():
        # Step-up: the largest index that clears its threshold rejects
        # everything below it, including indices that failed their own.
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    pvals_corrected[pvals_corrected > 1] = 1

    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    return reject, pvals_corrected
