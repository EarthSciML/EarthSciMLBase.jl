export SolverStrangThreads, SolverStrangSerial

"""
A simulator strategy based on Strang splitting.
Choose either `SimulatorStrangThreads` or `SimulatorStrangSerial` to run the simulation.

kwargs for ODEProblem constructor:
- u0: initial condtions; if "nothing", default values will be used.
- p: parameters; if "nothing", default values will be used.
- nonstiff_params: parameters for the non-stiff ODE system.
- name: name of the system.
"""
abstract type SolverStrang <: SolverStrategy end

"""
```julia
# Specify the number of threads and the stiff ODE solver algorithm.
# `timestep` is the length of time for each splitting step.
SimulatorStrangThreads(threads, stiffalg, timestep; kwargs...)
# Use the default number of threads.
SimulatorStrangThreads(stiffalg, timestep; kwargs...)
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
    stiffalg
    "Length of each splitting time step"
    timestep::AbstractFloat
    "Keyword arguments for the stiff ODEProblem constructor and solver."
    stiff_kwargs::Any


    SolverStrangThreads(nthreads, stiffalg, timestep; stiff_kwargs...) = new(nthreads,
        stiffalg, timestep, stiff_kwargs)
    SolverStrangThreads(stiffalg, timestep; stiff_kwargs...) = new(Threads.nthreads(),
        stiffalg, timestep, stiff_kwargs)
end

"""
```julia
# Specify the stiff ODE solver algorithm.
# `timestep` is the length of time for each splitting step.
SimulatorStrangSerial(stiffalg, timestep; kwargs...)
```

Perform a simulation using [Strang splitting](https://en.wikipedia.org/wiki/Strang_splitting),
where the MTK system is assumed to be stiff and the operators are assumed to be non-stiff.
The solution will be calculated in serial.

Additional kwargs for ODEProblem constructor:
- u0: initial condtions; if "nothing", default values will be used.
- p: parameters; if "nothing", default values will be used.

$(FIELDS)

WARNING: Is is not possible to change the parameters using this strategy..
"""
struct SolverStrangSerial <: SolverStrang
    "Stiff solver algorithm to use (see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)"
    stiffalg
    "Length of each splitting time step"
    timestep::AbstractFloat
    "Keyword arguments for the stiff ODEProblem constructor and solver."
    stiff_kwargs::Any
    SolverStrangSerial(stiffalg, timestep; stiff_kwargs...) = new(stiffalg, timestep, stiff_kwargs)
end

nthreads(st::SolverStrangThreads) = st.threads
nthreads(st::SolverStrangSerial) = 1

function ODEProblem(s::CoupledSystem, st::SolverStrang; u0=nothing, tspan=nothing,
        name=:model, extra_vars=[], kwargs...)

    sys_mtk = convert(ODESystem, s; name=name, extra_vars=extra_vars)

    dom = domain(s)
    u0 = isnothing(u0) ? init_u(sys_mtk, dom) : u0
    stiff_p = default_params(sys_mtk)

    II = CartesianIndices(tuple(size(dom)...))
    IIchunks = collect(Iterators.partition(II, length(II) ÷ nthreads(st)))
    tspan = isnothing(tspan) ? get_tspan(dom) : tspan
    start, finish = tspan
    prob = ODEProblem(sys_mtk, [], (start, start+typeof(start)(st.timestep)), []; st.stiff_kwargs...)
    stiff_integrators = [init(remake(prob, u0=zeros(eltype(u0), length(unknowns(sys_mtk))),
        p=deepcopy(stiff_p)), st.stiffalg, save_on=false, save_start=false, save_end=false,
        initialize_save=false;
        st.stiff_kwargs...) for _ in 1:length(IIchunks)]

    coord_sys, coord_args = _prepare_coord_sys(sys_mtk, dom)
    nonstiff_p = default_params(coord_sys)
    nonstiff_op = nonstiff_ops(s, sys_mtk, coord_args, dom, u0, nonstiff_p)

    setp! = coord_setter(sys_mtk, dom)

    cb = CallbackSet(
        stiff_callback(setp!, u0, st, IIchunks, stiff_integrators),
        get_callbacks(s, sys_mtk, dom)...,
    )
    ODEProblem(nonstiff_op, view(u0, :), (start, finish), nonstiff_p; callback=cb,
        dt=st.timestep, kwargs...)
end

"A callback to periodically run the stiff solver."
function stiff_callback(setp!, u0::AbstractArray{T}, st::SolverStrang, IIchunks, integrators) where T
    sz = size(u0)
    function affect!(integrator)
        u = reshape(integrator.u, sz...)
        ode_step!(setp!, st, u, IIchunks, integrators, integrator.t, T(st.timestep))
        @views integrator.u .= u[:]
    end
    PeriodicCallback(
        affect!,
        T(st.timestep),
        initial_affect=true,
        final_affect=false,
    )
end

"Take a step using the ODE solver."
function ode_step!(setp!, ::SolverStrangThreads, u, IIchunks, integrators, time, step_length)
    threaded_ode_step!(setp!, u, IIchunks, integrators, time, step_length)
end
function ode_step!(setp!, ::SolverStrangSerial, u, IIchunks, integrators, time, step_length)
    single_ode_step!(setp!, u, IIchunks[1], integrators[1], time, step_length)
end

"Take a step using the ODE solver."
function threaded_ode_step!(setp!, u, IIchunks, integrators, time, step_length)
    tasks = map(1:length(IIchunks)) do ithread
        Threads.@spawn single_ode_step!(setp!, u, IIchunks[$ithread], integrators[$ithread], time, step_length)
    end::Vector{Task}
    wait.(tasks)
    nothing
end

"Take a step using the ODE solver with the given IIchunk (grid cell interator) and integrator."
function single_ode_step!(setp!, u, IIchunk, integrator, time, step_length)
    for ii in IIchunk
        uii = @view u[:, ii]
        reinit!(integrator, uii, t0=time, tf=time + step_length,
            erase_sol=false, reset_dt=true)
        setp!(integrator.p, ii)
        solve!(integrator)
        @assert length(integrator.sol.u) == 0
        uii .= integrator.u
    end
end
