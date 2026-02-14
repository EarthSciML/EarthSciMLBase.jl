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

  - stiff_sparse: Whether the stiff ODE function should use a sparse Jacobian.
  - stiff_tgrad: Whether the stiff ODE function should use an analytical time gradient.

Additional kwargs for ODEProblem constructor:

  - u0: initial conditions; if "nothing", default values will be used.
  - p: parameters; if "nothing", default values will be used.
  - name: name of the model.
"""
struct SolverIMEX{MA, JT} <: SolverStrategy
    alg::MA
    jac_type::JT
    stiff_sparse::Bool
    stiff_tgrad::Bool
    function SolverIMEX(alg::MA = MapBroadcast(), jac_type::JT = BlockDiagonalJacobian();
            stiff_sparse = false, stiff_tgrad = true) where {
            MA <: MapAlgorithm, JT <: JacobianType}
        new{MA, JT}(alg, jac_type, stiff_sparse, stiff_tgrad)
    end
end

function ODEProblem{iip}(sys::CoupledSystem, st::SolverIMEX; u0 = nothing,
        name = :model, extra_vars = [], kwargs...) where {iip}
    sys_mtk = convert(System, sys; name = name, extra_vars = extra_vars)
    dom = domain(sys)

    u0 = isnothing(u0) ? init_u(sys_mtk, dom) : u0
    u0 = reshape(u0, :) # DiffEq state must be a vector.

    f1, sys_mtk,
    coord_args = mtk_grid_func(sys_mtk, dom, u0,
        st.alg; sparse = st.stiff_sparse,
        tgrad = st.stiff_tgrad)

    type_convert_params(sys_mtk, u0)
    p = MTKParameters(sys_mtk, ModelingToolkit.initial_conditions(sys_mtk))

    if st.alg isa MapKernel
        p = bitsify_params(p)
    end

    f2 = nonstiff_ops(sys, sys_mtk, coord_args, dom, u0, p, st.alg)

    cb = []
    event_cb = ModelingToolkit.process_events(sys_mtk)
    if !isnothing(event_cb)
        push!(cb, event_cb)
    end
    push!(cb, get_callbacks(sys, sys_mtk, coord_args, dom, st.alg)...)
    if :callback in keys(kwargs)
        push!(cb, kwargs[:callback])
        kwargs = filter((p -> p.first ≠ :callback), kwargs)
    end

    start, finish = get_tspan(dom)
    SplitODEProblem{iip}(f1, f2, u0, (start, finish), p,
        callback = CallbackSet(cb...); kwargs...)
end
function ODEProblem(sys::CoupledSystem, st::SolverIMEX; kwargs...)
    ODEProblem{true}(sys, st; kwargs...)
end

"""
Convert the floating point parameters in `sys` to the element type of `u`.

This is only needed until https://github.com/SciML/ModelingToolkit.jl/issues/3709
is resolved.
"""
function type_convert_params(sys::System, u::AbstractArray)
    T = eltype(u)
    ics = ModelingToolkit.initial_conditions(sys)
    for p in keys(ics)
        if ics[p] isa AbstractFloat
            ics[p] = T(ics[p])
        end
    end
end

"""
Convert parameters to bitstypes so that they can be used on the GPU.
"""
function bitsify_params(p::MTKParameters, op = tuple)
    tunable = op(p.tunable...)
    initials = op(p.initials...)
    discrete = Tuple(eltype(buf) <: Real ? op(buf...) : copy.(buf) for buf in p.discrete)
    constant = Tuple(eltype(buf) <: Real ? op(buf...) : copy.(buf) for buf in p.constant)
    nonnumeric = isempty(p.nonnumeric) ? p.nonnumeric : copy.(p.nonnumeric)
    caches = isempty(p.caches) ? p.caches : copy.(p.caches)
    return MTKParameters(
        tunable,
        initials,
        discrete,
        constant,
        nonnumeric,
        caches
    )
end
