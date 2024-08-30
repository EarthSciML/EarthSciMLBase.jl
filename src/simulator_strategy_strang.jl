export SimulatorStrangThreads, SimulatorStrangSerial

"""
A simulator strategy based on Strang splitting.
Choose either `SimulatorStrangThreads` or `SimulatorStrangSerial` to run the simulation.

!!! warning
    `SimulatorStrang` strategies will still work if no operator is included, 
    but any callbacks included in the system are executed together with the operators,
    so if there are no operators in the system, the callbacks will not be executed.
"""
abstract type SimulatorStrang <: SimulatorStrategy end

"""
```julia
# Specify the number of threads and the stiff and nonstiff ODE solver algorithm.
# `timestep` is the length of time for each splitting step.
SimulatorStrangThreads(threads, stiffalg, nonstiffalg, timestep)
# Use the default number of threads.
SimulatorStrangThreads(stiffalg, nonstiffalg, timestep)
```

Perform a simulation using [Strang splitting](https://en.wikipedia.org/wiki/Strang_splitting),
where the MTK system is assumed to be stiff and the operators are assumed to be non-stiff.
The solution of the stiff ODE system is parallelized across grid cells 
using the specified number of threads.

$(FIELDS)
"""
struct SimulatorStrangThreads <: SimulatorStrang
    "Number of threads to use"
    threads::Int
    "Stiff solver algorithm to use (see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)"
    stiffalg
    "Non-stiff solver algorithm to use (see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)"
    nonstiffalg
    "Length of each splitting time step"
    timestep::AbstractFloat
    SimulatorStrangThreads(stiffalg, nonstiffalg, timestep) = new(Threads.nthreads(), stiffalg, nonstiffalg, timestep)
end

"""
```julia
# Specify the stiff and nonstiff ODE solver algorithm.
# `timestep` is the length of time for each splitting step.
SimulatorStrangSerial(stiffalg, nonstiffalg, timestep)
```

Perform a simulation using [Strang splitting](https://en.wikipedia.org/wiki/Strang_splitting),
where the MTK system is assumed to be stiff and the operators are assumed to be non-stiff.
The solution will be calculated in serial.

$(FIELDS)
"""
struct SimulatorStrangSerial <: SimulatorStrang
    "Stiff solver algorithm to use (see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)"
    stiffalg
    "Non-stiff solver algorithm to use (see https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)"
    nonstiffalg
    "Length of each splitting time step"
    timestep::AbstractFloat
end

nthreads(st::SimulatorStrangThreads) = st.threads
nthreads(st::SimulatorStrangSerial) = 1

"""
$(TYPEDSIGNATURES)

Run the simualation.
`kwargs` are passed to the ODEProblem and ODE solver constructors.
"""
function run!(s::Simulator{T}, st::SimulatorStrang, u=init_u(s); kwargs...) where {T}
    II = CartesianIndices(size(u)[2:4])
    IIchunks = collect(Iterators.partition(II, length(II) ÷ nthreads(st)))
    start, finish = time_range(s.domaininfo)
    prob = ODEProblem(s.sys_mtk, [], (start, finish), []; kwargs...)
    stiff_integrators = [init(remake(prob, u0=similar(s.u_init), p=deepcopy(s.p)), st.stiffalg, save_on=false,
        save_start=false, save_end=false, initialize_save=false; kwargs...)
                         for _ in 1:length(IIchunks)]

    # Combine the non-stiff operators into a single operator.
    # This works because SciMLOperators can be added together.
    nonstiff_op = length(s.sys.ops) > 0 ? sum([get_scimlop(op, s) for op ∈ s.sys.ops]) : NullOperator(length(u))
    nonstiff_op = cache_operator(nonstiff_op, u)

    cb = CallbackSet(
        stiff_callback(s, st, IIchunks, stiff_integrators),
        get_callbacks(s)...,
    )
    @views nonstiff_prob = ODEProblem(nonstiff_op, u[:], (start, finish), s.p, callback=cb; kwargs...)
    solve(nonstiff_prob, st.nonstiffalg, dt=st.timestep; kwargs...)
end

"A callback to periodically run the stiff solver."
function stiff_callback(s::Simulator{T}, st::SimulatorStrang, IIchunks, integrators) where T
    function affect!(integrator)
        u = reshape(integrator.u, length(unknowns(s.sys_mtk)), [length(g) for g in s.grid]...)
        ode_step!(s, st, u, IIchunks, integrators, T(integrator.t), T(st.timestep))
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
function ode_step!(s::Simulator, st::SimulatorStrangThreads, u, IIchunks, integrators, time, step_length)
    threaded_ode_step!(s, u, IIchunks, integrators, time, step_length)
end
function ode_step!(s::Simulator, st::SimulatorStrangSerial, u, IIchunks, integrators, time, step_length)
    single_ode_step!(s, u, IIchunks[1], integrators[1], time, step_length)
end

"Take a step using the ODE solver."
function threaded_ode_step!(s::Simulator{T}, u, IIchunks, integrators, time::T, step_length::T) where {T}
    tasks = map(1:length(IIchunks)) do ithread
        Threads.@spawn single_ode_step!(s, u, IIchunks[$ithread], integrators[$ithread], time, step_length)
    end::Vector{Task}
    wait.(tasks)
    nothing
end

"Take a step using the ODE solver with the given IIchunk (grid cell interator) and integrator."
function single_ode_step!(s::Simulator{T}, u, IIchunk, integrator, time::T, step_length::T) where {T}
    for ii in IIchunk
        uii = @view u[:, ii]
        reinit!(integrator, uii, t0=time, tf=time + step_length,
            erase_sol=false, reset_dt=true)
        for (jj, g) ∈ enumerate(s.grid) # Set the coordinates of this grid cell.
            integrator.p[s.pvidx[jj]] = g[ii[jj]]
        end
        solve!(integrator)
        @assert length(integrator.sol.u) == 0
        uii .= integrator.u
    end
end