export SolverStrangThreads, SolverStrangSerial

"""
A simulator strategy based on Strang splitting.
Choose either `SimulatorStrangThreads` or `SimulatorStrangSerial` to run the simulation.

kwargs for ODEProblem constructor:

  - u0: initial conditions; if "nothing", default values will be used.
  - p: parameters; if "nothing", default values will be used.
  - nonstiff_params: parameters for the non-stiff ODE system.
  - name: name of the system.
"""
abstract type SolverStrang <: SolverStrategy end

mutable struct IIP{T1, T2}
    ii::T1
    p::T2
end

# Make IIP transparent for parameter access so MTK-generated affect code (and
# the SciMLStructures / SymbolicIndexingInterface machinery it sits on) can
# reach the underlying MTKParameters without knowing about the IIP wrapper.
# The stiff-RHS path (`f_stiff` in `_strang_ode_func`) explicitly unwraps via
# `p.p`, but callbacks attached for issue #219 don't, and would otherwise hit
# `BoundsError` walking off the end of the IIP wrapper.
@inline function Base.getproperty(iip::IIP, s::Symbol)
    s === :ii && return getfield(iip, :ii)
    s === :p && return getfield(iip, :p)
    return getproperty(getfield(iip, :p), s)
end
@inline function Base.setproperty!(iip::IIP, s::Symbol, v)
    s === :ii && return setfield!(iip, :ii, v)
    s === :p && return setfield!(iip, :p, v)
    return setproperty!(getfield(iip, :p), s, v)
end
@inline Base.getindex(iip::IIP, args...) = getindex(getfield(iip, :p), args...)
@inline Base.setindex!(iip::IIP, v, args...) = setindex!(getfield(iip, :p), v, args...)
@inline Base.length(iip::IIP) = length(getfield(iip, :p))
@inline Base.size(iip::IIP) = size(getfield(iip, :p))
Base.IndexStyle(::Type{<:IIP}) = IndexLinear()

SymbolicIndexingInterface.parameter_values(iip::IIP) = getfield(iip, :p)

SciMLStructures.ismutablescimlstructure(iip::IIP) =
    SciMLStructures.ismutablescimlstructure(getfield(iip, :p))
for Portion in (SciMLStructures.Tunable, SciMLStructures.Initials,
    SciMLStructures.Discrete, SciMLStructures.Constants,
    SciMLStructures.Caches)
    @eval SciMLStructures.canonicalize(portion::$Portion, iip::IIP) =
        SciMLStructures.canonicalize(portion, getfield(iip, :p))
    @eval SciMLStructures.replace(portion::$Portion, iip::IIP, newvals) =
        IIP{typeof(getfield(iip, :ii)), typeof(getfield(iip, :p))}(
            getfield(iip, :ii),
            SciMLStructures.replace(portion, getfield(iip, :p), newvals))
    @eval SciMLStructures.replace!(portion::$Portion, iip::IIP, newvals) =
        SciMLStructures.replace!(portion, getfield(iip, :p), newvals)
end

# MTK's `GeneratedFunctionWrapper` is `@generated`: when it sees a parameter
# argument that isn't `<: Union{Tuple, MTKParameters}` it falls through to the
# "single buffer" branch and rebuilds the call as `f(..., (p, nothing), ...)`
# — so the generated body's `___mtkparameters___[3]` then walks off the 2-tuple
# (PR #220 review). Unwrap `IIP` to the underlying `MTKParameters` at the GFW
# boundary so the type guard sees the right thing — the same `p.p` unwrap
# `f_stiff` already does, lifted to every MTK-generated function (RHS,
# jacobian, observed reader, modified-value reader, …) called with `integ.p`.
@inline function (gfw::ModelingToolkit.GeneratedFunctionWrapper)(u, iip::IIP, t)
    return gfw(u, getfield(iip, :p), t)
end
@inline function (gfw::ModelingToolkit.GeneratedFunctionWrapper)(out, u, iip::IIP, t)
    return gfw(out, u, getfield(iip, :p), t)
end

"""
```julia
# Specify the number of threads and the stiff ODE solver algorithm.
# `timestep` is the length of time for each splitting step.
SimulatorStrangThreads(threads, stiffalg, timestep; stiff_kwargs...)
# Use the default number of threads.
SimulatorStrangThreads(stiffalg, timestep; stiff_kwargs...)
```

Perform a simulation using [Strang splitting](https://en.wikipedia.org/wiki/Strang_splitting),
where the MTK system is assumed to be stiff and the operators are assumed to be non-stiff.
The solution of the stiff ODE system is parallelized across grid cells
using the specified number of threads.

$(FIELDS)
"""
struct SolverStrangThreads <: SolverStrang
    "Number of threads to use"
    threads::Int
    "Stiff solver algorithm to use (see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)"
    stiffalg::Any
    "Length of each splitting time step"
    timestep::AbstractFloat
    "Keyword arguments for the stiff ODEProblem constructor and solver."
    stiff_kwargs::Any
    "Algorithm for performing gridded computations."
    alg::MapAlgorithm

    function SolverStrangThreads(
            nthreads, stiffalg, timestep; alg = MapThreads(), stiff_kwargs...)
        new(nthreads, stiffalg, timestep, stiff_kwargs, alg)
    end
    function SolverStrangThreads(stiffalg, timestep; alg = MapThreads(), stiff_kwargs...)
        new(Threads.nthreads(), stiffalg, timestep, stiff_kwargs, alg)
    end
end

"""
```julia
# Specify the stiff ODE solver algorithm.
# `timestep` is the length of time for each splitting step.
SimulatorStrangSerial(stiffalg, timestep; stiff_kwargs...)
```

Perform a simulation using [Strang splitting](https://en.wikipedia.org/wiki/Strang_splitting),
where the MTK system is assumed to be stiff and the operators are assumed to be non-stiff.
The solution will be calculated in serial.

Additional kwargs for ODEProblem constructor:

  - u0: initial conditions; if "nothing", default values will be used.
  - p: parameters; if "nothing", default values will be used.

$(FIELDS)

WARNING: Is is not possible to change the parameters using this strategy.
"""
struct SolverStrangSerial <: SolverStrang
    "Stiff solver algorithm to use (see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)"
    stiffalg::Any
    "Length of each splitting time step"
    timestep::AbstractFloat
    "Keyword arguments for the stiff ODEProblem constructor and solver."
    stiff_kwargs::Any
    "Algorithm for performing gridded computations."
    alg::MapAlgorithm
    function SolverStrangSerial(
            stiffalg, timestep, alg::MapAlgorithm = MapBroadcast(); stiff_kwargs...)
        new(stiffalg, timestep, stiff_kwargs, alg)
    end
end

nthreads(st::SolverStrangThreads) = st.threads
nthreads(st::SolverStrangSerial) = 1

function _strang_ode_func(sys_mtk, coord_args, tspan, grd; sparse = false)
    mtkf_coord = build_coord_ode_function(sys_mtk, coord_args)
    jac_coord = build_coord_jac_function(sys_mtk, coord_args, sparse = sparse)
    _prob = ODEProblem(sys_mtk, [], tspan; sparse = sparse, build_initializeprob = false)

    function f_stiff(du, u, p, t)
        coords = (g[p.ii[j]] for (j, g) in enumerate(grd))
        mtkf_coord(du, u, p.p, t, coords...)
    end
    function jac_stiff(du, u, p, t)
        coords = (g[p.ii[j]] for (j, g) in enumerate(grd))
        jac_coord(du, u, p.p, t, coords...)
    end
    ode_f = ODEFunction(f_stiff, jac = jac_stiff, jac_prototype = _prob.f.jac_prototype)
    return ode_f, _prob.u0, _prob.p
end

function _strang_integrators(
        st::SolverStrang, dom::DomainInfo, f_ode, u0_single, start, p, event_cb)
    II = CartesianIndices(tuple(size(dom)...))
    IIchunks = collect(Iterators.partition(II, length(II) ÷ nthreads(st)))
    iip = IIP{typeof(II[1]), typeof(p)}(II[1], p)

    # Thread the data-load discrete callback into the inner stiff sub-integrators
    # so its `initialize = affect` populates parameter buffers before `init`'s
    # auto_dt_reset! evaluates the RHS — otherwise NaN propagates from
    # still-empty interpolator buffers (issue #219).
    nt = (; st.stiff_kwargs...)
    inner_kwargs = if isnothing(event_cb)
        nt
    else
        existing_cb = get(nt, :callback, nothing)
        merged_cb = isnothing(existing_cb) ? CallbackSet(event_cb) :
                    CallbackSet(event_cb, existing_cb)
        merge(nt, (; callback = merged_cb))
    end

    prob = ODEProblem(f_ode, u0_single, (start, start + typeof(start)(st.timestep)), iip;
        inner_kwargs...)
    stiff_integrators = [init(
                             remake(prob, u0 = u0_single,
                                 p = IIP{typeof(II[1]), typeof(p)}(II[1], p)),
                             st.stiffalg,
                             save_on = false, save_start = false,
                             save_end = false, initialize_save = false;
                             inner_kwargs...) for _ in 1:length(IIchunks)]
    return IIchunks, stiff_integrators
end

function ODEProblem(s::CoupledSystem, st::SolverStrang; u0 = nothing, tspan = nothing,
        name = :model, extra_vars = [], kwargs...)
    sys_mtk = convert(System, s; name = name, extra_vars = extra_vars)
    dom = domain(s)
    sys_mtk, coord_args = _prepare_coord_sys(sys_mtk, dom)

    start, finish = isnothing(tspan) ? get_tspan(dom) : tspan

    u0 = isnothing(u0) ? init_u(sys_mtk, dom) : u0

    type_convert_params(sys_mtk, u0)

    grd = grid(dom)
    sparse = :sparse in keys(st.stiff_kwargs) ? st.stiff_kwargs[:sparse] : false
    f_ode, u0_single,
    p = _strang_ode_func(sys_mtk, coord_args, (start, finish), grd;
        sparse = sparse)

    event_cb = ModelingToolkit.process_events(sys_mtk)
    IIchunks,
    stiff_integrators = _strang_integrators(
        st, dom, f_ode, u0_single, start, p, event_cb)
    nonstiff_op = nonstiff_ops(s, sys_mtk, coord_args, dom, u0, p, st.alg)

    cb = []
    if !isnothing(event_cb)
        push!(cb, event_cb)
    end
    push!(cb, get_callbacks(s, sys_mtk, coord_args, dom, st.alg)...)
    push!(cb, stiff_callback(reshape(u0, :, size(dom)...), st, IIchunks,
        stiff_integrators))
    if :callback in keys(kwargs)
        push!(cb, kwargs[:callback])
        kwargs = filter((p -> p.first ≠ :callback), kwargs)
    end

    # Attach sys_mtk so that users can query unknowns(prob.f.sys) to get the
    # variable ordering that matches prob.u0 (same ordering as init_u uses).
    nonstiff_fn = ODEFunction(nonstiff_op; sys = sys_mtk)
    ODEProblem(nonstiff_fn, view(u0, :), (start, finish), p; callback = CallbackSet(cb...),
        dt = st.timestep, kwargs...)
end

"""
A callback to periodically run the stiff solver.
"""
function stiff_callback(u0::AbstractArray{T, 4}, st::SolverStrang,
        IIchunks, integrators) where {T}
    sz = size(u0)
    function affect!(integrator)
        u = reshape(integrator.u, sz...)
        ode_step!(st, u, IIchunks, integrators, integrator.t, T(st.timestep))
        @views integrator.u .= u[:]
    end
    PeriodicCallback(
        affect!,
        T(st.timestep),
        initial_affect = true,
        final_affect = false
    )
end

"""
Take a step using the ODE solver.
"""
function ode_step!(::SolverStrangThreads, u, IIchunks, integrators, time, step_length)
    threaded_ode_step!(u, IIchunks, integrators, time, step_length)
end
function ode_step!(::SolverStrangSerial, u, IIchunks, integrators, time, step_length)
    single_ode_step!(u, IIchunks[1], integrators[1], time, step_length)
end

"""
Take a step using the ODE solver.
"""
function threaded_ode_step!(u, IIchunks, integrators, time, step_length)
    tasks = map(1:length(IIchunks)) do ithread
        Threads.@spawn single_ode_step!(u, IIchunks[$ithread], integrators[$ithread],
            time, step_length)
    end::Vector{Task}
    wait.(tasks)
    nothing
end

"""
Take a step using the ODE solver with the given IIchunk (grid cell iterator) and integrator.
"""
function single_ode_step!(u, IIchunk, integrator, time, step_length)
    for ii in IIchunk
        uii = @view u[:, ii]
        # `reset_dt = false`: `reinit!` runs `auto_dt_reset!` BEFORE
        # `initialize_callbacks!` (the opposite order from `__init`), so the
        # data-load discrete callback's `initialize` affect would not yet have
        # populated the interpolator parameter buffer when the inner stiff RHS
        # is evaluated for the initial-dt estimate — tripping the bounds check
        # in `interp_unsafe` against a still-sentinel `DataBufferType` (issue
        # EarthSciData #207).  We disable `reset_dt` here, set the cell index,
        # and then call `auto_dt_reset!` ourselves AFTER `reinit!`'s callback
        # initialization has refreshed the parameter buffer.
        reinit!(integrator, uii, t0 = time, tf = time + step_length,
            erase_sol = false, reset_dt = false)
        integrator.p.ii = ii
        if integrator.opts.adaptive
            auto_dt_reset!(integrator)
        end
        solve!(integrator)
        @assert length(integrator.sol.u) == 0
        uii .= integrator.u
    end
end
