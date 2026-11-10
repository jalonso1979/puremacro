"""Independent algebraic references for HANK state and variance curvature."""
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from puremacro.models._hank_second_order import solve_curvature


def claim_system(rho=.7, beta=.9):
    F0=np.array([[-rho,0.],[0.,1.]])
    F1=np.array([[1.,0.],[0.,-beta]])
    return F0,F1,np.array([[rho]]),np.zeros((1,1)),np.array([[[0.]],[[-2.]]]),np.zeros(2)


def test_squared_ar1_claim_has_exact_state_and_risk_curvature():
    Hxx,Gxx,Hss,Gss,report=solve_curvature(*claim_system(),shock_index=0)
    A=1/(1-.9*.7**2)
    np.testing.assert_allclose(Hxx,0.,atol=1e-14)
    np.testing.assert_allclose(Hss,0.,atol=1e-14)
    np.testing.assert_allclose(Gxx,2*A,rtol=1e-13)
    np.testing.assert_allclose(Gss,2*.9*A/(1-.9),rtol=1e-13)
    assert report['state_equation_error']<1e-12
    assert report['variance_equation_error']<1e-12


def test_rotating_states_and_nonlinear_transition_match_dense_reference():
    H=.6*np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
    G=np.array([[.3,-.2]])
    beta=.8
    F0=np.block([[-H,np.zeros((2,1))],[-G,np.ones((1,1))]])
    F1=np.block([[np.eye(2),np.zeros((2,1))],[beta*G,np.array([[-beta]])]])
    Q=np.array([[2.,.3],[.3,1.]])
    state_curve=np.array([[[.1,.2],[.2,-.1]],[[.4,0.],[0.,.3]]])
    C=-np.concatenate([state_curve,Q[None]])
    Hxx,Gxx,Hss,Gss,report=solve_curvature(F0,F1,H,G,C,np.zeros(3),shock_index=1)
    P=solve_discrete_lyapunov(np.sqrt(beta)*H.T,Q)
    np.testing.assert_allclose(Hxx,state_curve,atol=1e-13)
    np.testing.assert_allclose(Gxx[0],P,atol=1e-13)
    np.testing.assert_allclose(Hss,0.,atol=1e-13)
    np.testing.assert_allclose(Gss,beta*P[1,1]/(1-beta),atol=1e-13)
    A=np.column_stack([F1@np.vstack([np.eye(2),G]),F0[:,2:]])
    B=np.column_stack([np.zeros((3,2)),F1[:,2:]])
    operator=np.kron(A,np.eye(4))+np.kron(B,np.kron(H,H).T)
    dense=np.linalg.solve(operator,-C.reshape(-1)).reshape(3,2,2)
    np.testing.assert_allclose(np.concatenate([Hxx,Gxx]),dense,atol=1e-13)


def test_zero_curvature_nests_first_order_exactly():
    data=list(claim_system());data[4]=np.zeros_like(data[4])
    result=solve_curvature(*data,shock_index=0)
    for array in result[:4]:np.testing.assert_array_equal(array,0.)


@pytest.mark.parametrize('which',range(6))
@pytest.mark.parametrize('bad',[np.nan,np.inf])
def test_nonfinite_coefficient_data_rejected(which,bad):
    data=list(claim_system());data[which]=data[which].copy();data[which].flat[0]=bad
    with pytest.raises(ValueError,match='finite'):
        solve_curvature(*data,shock_index=0)


@pytest.mark.parametrize('index',[-1,1,True,.5])
def test_invalid_shock_index_rejected(index):
    with pytest.raises(ValueError,match='shock_index'):
        solve_curvature(*claim_system(),shock_index=index)


def test_singular_variance_equation_is_not_regularized():
    with pytest.raises(ValueError,match='variance'):
        solve_curvature(*claim_system(beta=1.),shock_index=0)


@pytest.fixture(scope='module')
def full_policy():
    from puremacro.models.one_asset_hank import calibrate_one_asset_hank
    from puremacro.models.hank_reduction import solve_hank_reduced_policy
    ss=calibrate_one_asset_hank(n_a=12,n_e=2,a_max=120.)
    return solve_hank_reduced_policy(ss,distribution_rank=None,marginal_rank=None,shock_std=.001)


@pytest.fixture(scope='module')
def second_order(full_policy):
    from puremacro.models.hank_second_order import solve_hank_second_order
    return solve_hank_second_order(full_policy)


def test_live_hank_second_order_improves_signed_shock_errors_cubically(second_order,full_policy):
    for sign in (-1,1):
        errors=[]
        for amplitude in (.001,.0005,.00025):
            x=full_policy.state(monetary_shock=sign*amplitude)
            report=second_order.audit(x,shock_std=amplitude,tolerance=1e-5)
            assert report.accepted
            assert report.diagnostics['minimum_mass']>=-1e-12
            assert report.diagnostics['mass_error']<1e-10
            assert report.diagnostics['budget_error']<1e-8
            errors.append(report.diagnostics['equation_error'])
        assert 6<errors[0]/errors[1]<10
        assert 6<errors[1]/errors[2]<10


def test_risk_intercept_is_quadratic_in_variance(second_order,full_policy):
    x=full_policy.state()
    a=second_order.controls(x,shock_std=.001)
    b=second_order.controls(x,shock_std=.0005)
    assert np.max(abs(a))>1e-6
    np.testing.assert_allclose(a,4*b,rtol=1e-14)
    np.testing.assert_array_equal(second_order.controls(x,shock_std=0.),0.)


def test_mean_decomposition_and_pruned_path_recurrences(second_order,full_policy):
    moments=second_order.mean_decomposition()
    H=full_policy.transition;G=full_policy.reduced_feedback;sigma=full_policy.shock_std
    covariance=full_policy.covariance()
    risk=.5*second_order.transition_variance*sigma**2
    curve=.5*np.einsum('aij,ij->a',second_order.transition_curvature,covariance)
    np.testing.assert_allclose((np.eye(len(H))-H)@moments['state_mean'],risk+curve,atol=1e-15)
    np.testing.assert_allclose(moments['state_mean'],moments['state_risk']+moments['state_dispersion'],atol=1e-15)
    eps=np.array([.0002,-.0001,.00005,0.,0.])
    sim=second_order.simulate(eps)
    assert sim.converged
    x1=np.zeros(len(H));x2=x1.copy()
    for t,e in enumerate(eps):
        x1[full_policy.shock_index]+=e
        y=G@(x1+x2)+.5*np.einsum('aij,i,j->a',second_order.feedback_curvature,x1,x1)+.5*second_order.feedback_variance*sigma**2
        np.testing.assert_allclose(sim.states[t],x1+x2,atol=1e-15)
        np.testing.assert_allclose(sim.paths.iloc[t].values,y[-3:],atol=1e-15)
        x2=H@x2+.5*np.einsum('aij,i,j->a',second_order.transition_curvature,x1,x1)+risk
        x1=H@x1


def test_second_order_owns_input_policy_and_immutable_arrays(full_policy):
    from copy import deepcopy
    from puremacro.models.hank_second_order import solve_hank_second_order
    parent=deepcopy(full_policy);sol=solve_hank_second_order(parent)
    expected=sol.controls(parent.state(monetary_shock=.0001))
    parent.transition=np.zeros_like(parent.transition)
    parent.reduced_feedback=np.zeros_like(parent.reduced_feedback)
    np.testing.assert_array_equal(sol.controls(full_policy.state(monetary_shock=.0001)),expected)
    for field in ('transition_curvature','feedback_curvature','transition_variance','feedback_variance'):
        value=getattr(sol,field)
        with pytest.raises(ValueError):value.setflags(write=True)


@pytest.mark.parametrize('sigma',[-1.,True,'0.1',[.1],np.inf])
def test_invalid_shock_scales_fail_before_evaluation(second_order,full_policy,sigma):
    with pytest.raises(ValueError):second_order.controls(full_policy.state(),shock_std=sigma)


@pytest.mark.parametrize('kwargs',[{'difference_step':0.},{'difference_step':True},
    {'curvature_tolerance':0.},{'max_states':2},{'max_states':True},{'max_workspace_bytes':1}])
def test_solver_guards_resources_and_controls(full_policy,kwargs):
    from puremacro.models.hank_second_order import solve_hank_second_order
    with pytest.raises(ValueError):solve_hank_second_order(full_policy,**kwargs)


def test_reduced_household_model_is_explicitly_unsupported(full_policy):
    from puremacro.models.hank_second_order import solve_hank_second_order
    from puremacro.models.hank_reduction import solve_hank_reduced_policy
    reduced=solve_hank_reduced_policy(full_policy.steady_state,distribution_rank=8,marginal_rank=12)
    with pytest.raises(ValueError,match='full'):solve_hank_second_order(reduced)


def test_unresolved_curvature_is_rejected_without_regularization(full_policy):
    from puremacro.models.hank_second_order import solve_hank_second_order
    with pytest.raises(ValueError,match='derivative refinement'):
        solve_hank_second_order(full_policy,curvature_tolerance=1e-10)


def test_infeasible_state_is_retained_as_a_failed_audit(second_order):
    x=second_order.state();x[0]=10.
    report=second_order.audit(x)
    assert not report.accepted
    assert report.diagnostics['reason'] in ('outside_local_domain','equations_or_feasibility_failed')


@pytest.mark.parametrize('order',[1,13,True,2.5])
def test_quadrature_limits_are_explicit(second_order,order):
    with pytest.raises(ValueError,match='quadrature_order'):
        second_order.audit(second_order.state(),quadrature_order=order)


def test_overflowing_polynomial_evaluations_fail_explicitly(second_order):
    x=second_order.state();x[-1]=1e308
    for method in (second_order.controls,second_order.next_state):
        with pytest.raises(ValueError,match='nonfinite'):
            method(x)
        with pytest.raises(ValueError,match='variance'):
            method(second_order.state(),shock_std=1e308)


def test_zero_realized_shocks_converge_to_risk_component(second_order):
    mean=second_order.mean_decomposition()
    path=second_order.simulate(np.zeros(400))
    assert path.converged
    np.testing.assert_allclose(path.states[-1],mean['state_risk'],atol=1e-9,rtol=1e-5)
    np.testing.assert_allclose(path.paths.iloc[-1],mean['control_risk'][-3:],atol=1e-9,rtol=1e-5)


@pytest.mark.parametrize('sigma,horizon,depth',[(0.,16,1),(.0005,8,7)])
def test_independent_finite_nonlinear_equilibrium_improves_on_first_order(sigma,horizon,depth):
    from puremacro.models.one_asset_hank import calibrate_one_asset_hank
    from puremacro.models.hank_reduction import solve_hank_reduced_policy
    from puremacro.models.hank_second_order import solve_hank_second_order
    from puremacro.models.hank_nonlinear_continuation import solve_hank_nonlinear_continuation
    ss=calibrate_one_asset_hank(n_a=20,n_e=2,a_max=120.)
    p=solve_hank_reduced_policy(ss,distribution_rank=None,marginal_rank=None,shock_std=sigma)
    second=solve_hank_second_order(p)
    x=p.state(monetary_shock=.0005)
    ref=solve_hank_nonlinear_continuation(p,aggregate_state=np.array([0.,.0005]),horizon=horizon,
        branch_depth=depth,quadrature_order=2,tolerance=1e-9)
    assert ref.converged
    truth=ref.aggregate_coordinates[0]
    first_error=np.max(abs(p.aggregate_feedback@x-truth))
    second_error=np.max(abs(second.controls(x)[-3:]-truth))
    assert second_error<.3*first_error
    # This reference retains a first-order terminal tail. It is a finite
    # stochastic equation check, not a certification of the infinite tail.
    assert ref.tree.metadata['terminal_policy_order']==1


@pytest.mark.parametrize('amplitude',[.001,.0005])
def test_negative_finer_grid_shocks_keep_failed_distribution_audit(amplitude):
    from puremacro.models.one_asset_hank import calibrate_one_asset_hank
    from puremacro.models.hank_reduction import solve_hank_reduced_policy
    from puremacro.models.hank_second_order import solve_hank_second_order
    ss=calibrate_one_asset_hank(n_a=20,n_e=2,a_max=120.)
    linear=solve_hank_reduced_policy(ss,distribution_rank=None,marginal_rank=None,shock_std=amplitude)
    policy=solve_hank_second_order(linear)
    negative=policy.audit(policy.state(monetary_shock=-amplitude))
    positive=policy.audit(policy.state(monetary_shock=amplitude))
    assert not negative.accepted
    assert negative.diagnostics['distribution_error']>1e-5
    assert negative.diagnostics['reason']=='equations_or_feasibility_failed'
    assert positive.accepted
