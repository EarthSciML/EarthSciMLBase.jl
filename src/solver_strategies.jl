export SolverIMEX

"""
SolverStrategy is an abstract type that defines the strategy for running a simulation.
"""
abstract type SolverStrategy end

"""
A solver strategy based on implicit-explicit (IMEX) time integration.
See [here](https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/)
for additional information.

kwargs:
- stiff_scimlop: Whether the stiff ODE function should be implemented as a SciMLOperator.
- stiff_sparse: Whether the stiff ODE function should use a sparse Jacobian.
- stiff_jac: Whether the stiff ODE function should use an analytical Jacobian.
- stiff_jac_scimlop: Whether the stiff ODE function Jacobian should be implemented as a SciMLOperator. (Ignored if `stiff_jac==false`.)
- stiff_tgrad: Whether the stiff ODE function should use an analytical time gradient.

Additional kwargs for ODEProblem constructor:
- u0: initial condtions; if "nothing", default values will be used.
- p: parameters; if "nothing", default values will be used.
- name: name of the model.
"""
struct SolverIMEX <: SolverStrategy
    stiff_scimlop::Bool
    stiff_jac::Bool
    stiff_sparse::Bool
    stiff_tgrad::Bool
    function SolverIMEX(; stiff_scimlop=false, stiff_jac=true, stiff_sparse=true, stiff_tgrad=true)
        new(stiff_scimlop, stiff_jac, stiff_sparse, stiff_tgrad)
    end
end

function ODEProblem(sys::CoupledSystem, st::SolverIMEX; u0=nothing, p=nothing,
    name=:model, kwargs...)

    sys_mtk, obs_eqs = convert(ODESystem, sys; simplify=true, name=name)
    dom = domain(sys)

    u0 = isnothing(u0) ? init_u(sys_mtk, dom) : u0
    p = isnothing(p) ? default_params(sys_mtk) : p

    f1 = mtk_grid_func(sys_mtk, dom, u0, p; jac=st.stiff_jac,
        sparse=st.stiff_sparse, scimlop=st.stiff_scimlop, tgrad=st.stiff_tgrad)

    f2 = nonstiff_ops(sys, sys_mtk, obs_eqs, dom, u0, p)

    cb = get_callbacks(sys, sys_mtk, obs_eqs, dom)
    start, finish = tspan(dom)
    SplitODEProblem(f1, f2, u0[:], (start, finish), p,
        callback=CallbackSet(cb...); kwargs...)
end
