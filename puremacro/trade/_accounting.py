"""Coherent producer/purchaser valuation and a fixed foreign-balance closure.

The compatibility evaluator remains separate. This mode uses homogeneous Cobb–
Douglas factor costs, a calibrated tax on output revenue, tax-inclusive final
expenditure shares, and full schedule duties rebated to the importing country.
"""
from __future__ import annotations

import numpy as np


def _parameters(calib):
    """Recover actual final-use tax shares, excluding financial saving from sales."""
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    arrays = (calib.a, calib.afd, calib.alpha, calib.beta, calib.tax, calib.theta,
              calib.ytot, calib.l_endow, calib.k_endow, calib.invforT)
    if any(a is None or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Consistent accounting requires finite calibrated coefficients and expenditure shares")
    if (np.any(calib.a < 0) or np.any(calib.afd < 0) or np.any(calib.theta < 0)
            or np.any(calib.alpha <= 0) or np.any(calib.alpha >= 1) or np.any(calib.beta <= 0)
            or np.any(calib.tax >= 1) or np.any(calib.ytot <= 0)
            or np.any(calib.l_endow <= 0) or np.any(calib.k_endow <= 0)):
        raise ValueError("Consistent accounting requires nonnegative flow/share coefficients and positive active factors/output")
    if not np.allclose(calib.theta.sum(axis=1), 1., atol=1e-10, rtol=0):
        raise ValueError("Final expenditure shares must sum to one in each country")
    if abs(np.sum(calib.invforT)) > 1e-10*max(1., np.sum(np.abs(calib.invforT))):
        raise ValueError("Baseline foreign balances must sum to zero")
    inv = 1 if nfd >= 2 else 0
    # The legacy tax_fd denominator includes foreign saving in investment.
    # Recover the original tax amount, then divide by actual purchaser spending.
    if calib.data_calibra is not None:
        n = nc*ns
        d = np.asarray(calib.data_calibra)
        if d.shape != (n+3, n+nfd*nc) or not np.isfinite(d).all():
            raise ValueError("A finite calibration table with the declared dimensions is required")
        basic = d[:n, n:].sum(0).reshape(nc, nfd).T[None]
        tax_amount = d[n, n:].reshape(nc, nfd).T[None]
        spending = basic + tax_amount
    else:
        income0 = (calib.l_endow + calib.k_endow).reshape(1, 1, nc)
        legacy_rate = np.zeros_like(calib.theta) if calib.tax_fd is None else calib.tax_fd
        if calib.T is not None:
            income0 = income0 + np.asarray(calib.T).reshape(1, 1, nc)
        else:
            income0 = (income0 + (calib.tax*calib.ytot).sum(1)[:, None, :]) / (
                1-(legacy_rate*calib.theta).sum(1)[:, None, :])
        allocated = calib.theta * income0
        tax_amount = legacy_rate * allocated
        spending = allocated.copy()
        spending[:, inv, :] -= np.asarray(calib.invforT).reshape(1, nc)
        basic = spending - tax_amount
    if np.any(spending < 0) or np.any(basic < 0):
        raise ValueError("Signed final-use aggregates need aggregation or a separate inventory model")
    inactive = basic == 0
    if np.any(inactive & ((spending != 0) | (calib.theta != 0))):
        raise ValueError("A final-use category with expenditure must have a positive goods basket")
    shares = np.divide(tax_amount, spending, out=np.zeros_like(spending), where=spending != 0)
    if not np.isfinite(shares).all() or np.any(shares >= 1):
        raise ValueError("Final-use tax shares must be finite and less than one")
    expected_sums = (~inactive).astype(float)
    if not np.allclose(calib.afd.sum(0)[None], expected_sums, rtol=0, atol=1e-10):
        raise ValueError("Final goods basket weights must sum to one for each active category")
    return shares, basic, inv


def evaluate(x, calib, tau, tau_fd, tauf, tauf_fd, *, sigma=0., fiscal_closure="lump_sum",
             recycling_params=None, capacity_margins=None):
    from .equilibrium import unpack_equilibrium_vector, _get_ces_weights
    from .solver import _resolve_tariffs

    if fiscal_closure not in ("lump_sum", "baseline", "") or recycling_params:
        raise NotImplementedError("Consistent accounting currently supports lump-sum fiscal rebates only")
    if capacity_margins is not None:
        raise NotImplementedError("Consistent accounting requires a separately derived capacity-cost model")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc*ns
    fd_tax, basic0, inv = _parameters(calib)
    ta, tf, _, _ = _resolve_tariffs(calib, tau, tau_fd, tauf, tauf_fd)
    for schedule, shape in ((ta, (n, ns, nc)), (tf, (n, nfd, nc))):
        if schedule.shape != shape or not np.isfinite(schedule).all() or np.any(schedule <= 0):
            raise ValueError("Tariff multipliers must have the expected shape and be finite and positive")
        for c in range(nc):
            if not np.allclose(schedule[c*ns:(c+1)*ns, :, c], 1., rtol=0, atol=1e-14):
                raise ValueError("Import tariffs must be one on domestic transactions")
    v = unpack_equilibrium_vector(x, ns=ns, nc=nc, nfd=nfd)
    p, y, r, w, T, B = v.p, v.y, v.r, v.w, v.T, v.invforT
    pv = p.ravel(order="F")
    purchaser_inter = pv[:, None, None]*ta
    if sigma == 0:
        inter_cost = np.sum(calib.a*purchaser_inter, axis=0)[None]
        Z = calib.a*y
    else:
        amount, weights = _get_ces_weights(calib)
        if abs(sigma-1) < 1e-8:
            index = np.exp(np.sum(weights*np.log(purchaser_inter), axis=0))
        else:
            inner = np.sum(weights*purchaser_inter**(1-sigma), axis=0)
            index = np.where(amount > 0, inner, 1.)**(1/(1-sigma))
        inter_cost = (amount*index)[None]
        Z = calib.a*y*(index[None]/purchaser_inter)**sigma
    unit_va = (r/calib.alpha)**calib.alpha * (w/(1-calib.alpha))**(1-calib.alpha)/calib.beta
    pp = (unit_va + inter_cost)/(1-calib.tax)
    labor = (1-calib.alpha)*unit_va*y/w
    capital = calib.alpha*unit_va*y/r
    Q = np.sum(calib.afd*pv[:, None, None]*tf, axis=0)[None]
    # Empty categories carry zero spending and zero flows, with a harmless index.
    Q = np.where(basic0 > 0, Q, 1.)
    P = Q/(1-fd_tax)
    income = w*calib.l_endow.reshape(1, 1, nc) + r*calib.k_endow.reshape(1, 1, nc) + T
    expenditure = calib.theta*income
    expenditure[:, inv, :] -= B
    quantity = expenditure/P
    F = calib.afd*quantity
    final_tax = fd_tax*expenditure
    production_tax = calib.tax*p*y
    duties_i = (ta-1)*pv[:, None, None]*Z
    duties_f = (tf-1)*pv[:, None, None]*F
    tariffs = duties_i.sum(axis=(0, 1)) + duties_f.sum(axis=(0, 1))
    government = production_tax.sum(axis=1).ravel() + final_tax.sum(axis=1).ravel() + tariffs
    zm = Z.reshape(n, n, order="F")
    fm = F.reshape(n, nfd*nc, order="F")
    producer_values = np.hstack([zm, fm])*pv[:, None]
    trade = producer_values[:, :n].reshape(nc, ns, nc, ns).sum(axis=(1, 3))
    trade += producer_values[:, n:].reshape(nc, ns, nc, nfd).sum(axis=(1, 3))
    np.fill_diagonal(trade, 0.)
    net_exports = trade.sum(1)-trade.sum(0)
    goods = y.ravel(order="F")-zm.sum(1)-fm.sum(1)
    prices = (pp-p).ravel(order="F")
    labor_gap = calib.l_endow.ravel()-labor.sum(1).ravel()
    capital_gap = calib.k_endow.ravel()-capital.sum(1).ravel()
    budget_gap = T.ravel()-government
    # Foreign saving is exogenous in numeraire units. Endogenous trade balances
    # plus national budget identities otherwise leave the model underidentified.
    balances = v.XN-np.asarray(calib.invforT).ravel()[:nc-1]
    solver_goods = goods.copy()
    solver_goods[0] = pv[0]-1.  # Drop one redundant goods equation (Walras' law).
    residuals = np.concatenate([solver_goods, prices, labor_gap, capital_gap, balances, budget_gap])
    physical = np.concatenate([goods, prices, labor_gap, capital_gap, net_exports-B.ravel(), budget_gap])
    return dict(p=p, pp=pp, y=y, r=r, w=w, T=T, B=B, XN=v.XN, Z=zm, F=fm,
                Q=Q, P=P, c=quantity, income=income, expenditure=expenditure,
                final_tax=final_tax, production_tax=production_tax, labor=labor, capital=capital,
                duties_i=duties_i, duties_f=duties_f, tariffs=tariffs, trade=trade,
                producer_values=producer_values, ta=ta, tf=tf, basic0=basic0, fd_tax=fd_tax,
                residuals=residuals, physical_residuals=physical, goods_gap=goods,
                net_exports_gap=net_exports-B.ravel(), government=government)


def postprocess(b, calib, base_result=None):
    nc, ns, nfd = calib.nc, calib.ns, calib.n_final_demand
    n = nc*ns
    p, y, w, r = b["p"], b["y"], b["w"], b["r"]
    pv = p.ravel(order="F")
    if base_result is not None:
        if base_result.metadata.get("accounting") != "consistent":
            raise ValueError("A consistent price index requires a consistent baseline result")
        basket, base_price, base_wage = base_result.c_fd, base_result.Pfd_final, base_result.w_sol
    else:
        basket, base_price, base_wage = b["basic0"], 1/(1-b["fd_tax"]), np.ones_like(w)
    nominal_cpi = np.sum(b["P"]*basket, axis=(0, 1))/np.sum(base_price*basket, axis=(0, 1))
    cpi = nominal_cpi/(w/base_wage).ravel()
    inter = b["Z"].reshape(ns, nc, ns, nc, order="F")
    final = b["F"].reshape(ns, nc, nfd, nc, order="F")
    qx, qf = inter.transpose(0, 2, 1, 3).copy(), final.transpose(0, 2, 1, 3).copy()
    for c in range(nc):
        qx[:, :, c, c] = 0.; qf[:, :, c, c] = 0.
    exported = qx.sum(axis=(1, 3))+qf.sum(axis=(1, 3))
    export_quantity = exported.sum(0)
    import_quantity = qx.sum(axis=(0, 1, 2))+qf.sum(axis=(0, 1, 2))
    exports, imports = b["trade"].sum(1), b["trade"].sum(0)
    px = np.divide(exports, export_quantity, out=np.ones(nc), where=export_quantity > 0)
    pm = np.divide(imports, import_quantity, out=np.ones(nc), where=import_quantity > 0)
    duty_i = b["duties_i"].sum(0).T
    duty_f = b["duties_f"].sum(0).T
    tax_row = np.hstack([b["production_tax"].ravel(order="F"), b["final_tax"].ravel(order="F")])
    tariff_row = np.hstack([duty_i.ravel(), duty_f.ravel()])
    wage_row = np.hstack([(w*b["labor"]).ravel(order="F"), np.zeros(nfd*nc)])
    rent_row = np.hstack([(r*b["capital"]).ravel(order="F"), np.zeros(nfd*nc)])
    table = np.vstack([b["producer_values"], tax_row, wage_row, rent_row])
    full_table = np.vstack([b["producer_values"], tax_row, tariff_row, wage_row, rent_row])
    purchaser_i = b["Z"]*pv[:, None]*b["ta"].reshape(n, n, order="F")
    purchaser_f = b["F"]*pv[:, None]*b["tf"].reshape(n, nfd*nc, order="F")/(1-b["fd_tax"].ravel(order="F"))[None]
    factor_income = w.ravel()*calib.l_endow.ravel() + r.ravel()*calib.k_endow.ravel()
    gdp = factor_income+b["government"]
    household_gap = purchaser_f.sum(0).reshape(nc, nfd).sum(1)+b["B"].ravel()-b["income"].ravel()
    expenditure_gdp = purchaser_f.sum(0).reshape(nc, nfd).sum(1)+exports-imports
    ledger = {"goods": float(np.max(np.abs(b["goods_gap"]))),
              "foreign_balance": float(np.max(np.abs(b["net_exports_gap"]))),
              "household_budget": float(np.max(np.abs(household_gap))),
              "government_budget": float(np.max(np.abs(b["T"].ravel()-b["government"]))),
              "gdp_income_expenditure": float(np.max(np.abs(gdp-expenditure_gdp))),
              "physical_equations": float(np.max(np.abs(b["physical_residuals"])))}
    residual = b["residuals"]
    meta = dict(accounting="consistent", tariff_revenue_mode="schedule", replicate_matlab_precedence=False,
                intermediate_tariff_multipliers=b["ta"].copy(),
                final_tariff_multipliers=b["tf"].copy(),
                matlab_compat=False, foreign_balance_closure="fixed baseline in numeraire units",
                numeraire="first country/sector producer price = 1", fiscal_closure="lump_sum",
                production_tax_base="output revenue", final_tax_share=b["fd_tax"],
                final_tax_base="actual final expenditure excluding foreign saving",
                producer_values=b["producer_values"], purchaser_intermediate_values=purchaser_i,
                purchaser_final_values=purchaser_f, production_tax_receipts=b["production_tax"],
                final_tax_receipts=b["final_tax"], household_income=b["income"],
                final_expenditure=b["expenditure"], account_residuals=ledger,
                physical_residuals=b["physical_residuals"], nominal_cpi=nominal_cpi,
                cpi_convention="fixed-basket purchaser-price Laspeyres, divided by relative wage",
                demand_feasible=bool(np.all(b["expenditure"] >= 0)),
                table_valuation="producer transaction values plus nominal tax, duty and factor-payment rows")
    return dict(p_sol=p, y_sol=y, r_sol=r, w_sol=w, T_sol=b["T"], XN_sol=b["XN"],
                invforT=b["B"], intermediate_matrix=b["Z"], final_demand_matrix=b["F"],
                intermediate_flows=inter, final_demand_flows=final, p_fd=b["Q"], P_fd=b["P"],
                c=b["c"], c_fd=b["c"], cd=calib.theta*b["income"]/b["P"],
                net_exports=exports-imports, tariffs=b["tariffs"], fiscal_tariffs=b["tariffs"], tariffs_interm=duty_i,
                tariffs_fd=duty_f, bilateral_trade=b["trade"], exports=exports, imports=imports,
                gdp=gdp, gdp_fc=factor_income, cpi=cpi, terms_of_trade=px/pm,
                data_model_vf=table, data_tariff_vf=full_table, qxX0=inter.transpose(0, 2, 1, 3),
                qxFD0=final.transpose(0, 2, 1, 3), residuals=residual,
                diff=float(np.sum(np.abs(residual))), residual_norm=float(np.sum(np.abs(residual))),
                max_residual=float(np.max(np.abs(residual))), accounting_metadata=meta)
