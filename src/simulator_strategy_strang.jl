export SimulatorStrangThreads, SimulatorStrangSerial

"""
A simulator strategy based on Strang splitting.
Choose either `SimulatorStrangThreads` or `SimulatorStrangSerial` to run the simulation.
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
function run!(s::Simulator{T}, st::SimulatorStrang; kwargs...) where {T}
    II = CartesianIndices(size(s.u)[2:4])
    IIchunks = collect(Iterators.partition(II, length(II) ÷ nthreads(st)))
    start, finish = time_range(s.domaininfo)
    prob = ODEProblem(s.sys_mtk, [], (start, finish), []; kwargs...)
    integrators = [init(remake(prob, u0=similar(s.u_init), p=deepcopy(s.p)), st.stiffalg, save_on=false,
        save_start=false, save_end=false, initialize_save=false; kwargs...)
                   for _ in 1:length(IIchunks)]

    # Combine the non-stiff operators into a single operator.
    # This works because SciMLOperators can be added together.
    nonstiff_op = length(s.sys.ops) > 0 ? sum([get_scimlop(op, s) for op ∈ s.sys.ops]) : nothing

    init_u!(s)

    times = start:T(st.timestep):finish

    @progress name = String(nameof(s.sys_mtk)) for time in times
        strang_step!(s, st, IIchunks, integrators, nonstiff_op, time)
    end
    nothing
end

"Take a step using the ODE solver."
function threaded_ode_step!(s::Simulator{T}, IIchunks, integrators, time::T, step_length::T) where {T}
    tasks = map(1:length(IIchunks)) do ithread
        Threads.@spawn single_ode_step!(s, IIchunks[$ithread], integrators[$ithread], time, step_length)
    end::Vector{Task}
    wait.(tasks)
    nothing
end

"Take a step using the ODE solver with the given IIchunk (grid cell interator) and integrator."
function single_ode_step!(s::Simulator{T}, IIchunk, integrator, time::T, step_length::T) where {T}
    for ii in IIchunk
        uii = @view s.u[:, ii]
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

"Take a step using the nonstiff operator."
function operator_step!(s::Simulator{T}, nonstiff_op, time::T, step_length::T) where {T}
    if !isnothing(nonstiff_op)
        @info "xxx"
        @views nonstiff_op(s.du[:], s.u[:], s.p, time)
        s.u .+= s.du .* step_length
    end
end

"Take a step using Strang splitting, first with the ODE solver, then with the operators."
function strang_step!(s::Simulator{T}, st::SimulatorStrangThreads, IIchunks, integrators, nonstiff_op, time::T) where {T}
    threaded_ode_step!(s, IIchunks, integrators, time, T(st.timestep))
    operator_step!(s, nonstiff_op, time, T(st.timestep))
    nothing
end

"Take a step using Strang splitting, first with the ODE solver, then with the operators."
function strang_step!(s::Simulator{T}, st::SimulatorStrangSerial, IIchunks, integrators, nonstiff_op, time::T) where {T}
    single_ode_step!(s, IIchunks[1], integrators[1], time, st.timestep)
    operator_step!(s, nonstiff_op, time, st.timestep)
    nothing
end