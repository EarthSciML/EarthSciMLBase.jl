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

function _strang_integrators(st::SolverStrang, dom::DomainInfo, f_ode, u0_single, start, p)
    II = CartesianIndices(tuple(size(dom)...))
    IIchunks = collect(Iterators.partition(II, length(II) ÷ nthreads(st)))
    iip = IIP{typeof(II[1]), typeof(p)}(II[1], p)
    prob = ODEProblem(f_ode, u0_single, (start, start + typeof(start)(st.timestep)), iip;
        st.stiff_kwargs...)
    stiff_integrators = [init(
                             remake(prob, u0 = u0_single,
                                 p = IIP{typeof(II[1]), typeof(p)}(II[1], p)),
                             st.stiffalg,
                             save_on = false, save_start = false,
                             save_end = false, initialize_save = false;
                             st.stiff_kwargs...) for _ in 1:length(IIchunks)]
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

    IIchunks, stiff_integrators = _strang_integrators(st, dom, f_ode, u0_single, start, p)
    nonstiff_op = nonstiff_ops(s, sys_mtk, coord_args, dom, u0, p, st.alg)

    cb = []
    event_cb = ModelingToolkit.process_events(sys_mtk)
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

    ODEProblem(nonstiff_op, view(u0, :), (start, finish), p; callback = CallbackSet(cb...),
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
        reinit!(integrator, uii, t0 = time, tf = time + step_length,
            erase_sol = false, reset_dt = true)
        integrator.p.ii = ii
        solve!(integrator)
        @assert length(integrator.sol.u) == 0
        uii .= integrator.u
    end
end
