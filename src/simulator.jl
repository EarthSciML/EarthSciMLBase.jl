export Simulator, run!

"""
$(TYPEDSIGNATURES)

Specify a simulator for large-scale model runs. 
`Δs` represent the grid cell spacing in each dimension; for example `Δs = [0.1, 0.1, 1]` 
would represent a grid with 0.1 spacing in the first two dimensions and 1 in the third,
in whatever units the grid is natively in.
The grid spacings should be given in the same order as the partial independent variables
are in the provided `DomainInfo`.
`algorithm` should be a [DifferentialEquations.jl ODE solver](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/).
`kwargs` are passed on to the DifferentialEquations.jl integrator initialization.

$(TYPEDFIELDS)
"""
struct Simulator{T,IT,FT1,FT2,TG}
    "The system to be integrated"
    sys::CoupledSystem
    "The ModelingToolkit version of the system"
    sys_mtk::ODESystem
    "Information about the spatiotemporal simulation domain"
    domaininfo::DomainInfo{T}
    "The system state"
    u::Array{T,4}
    "The system state derivative"
    du::Array{T,4}
    "The system parameter values"
    p::Vector{T}
    "The initial values of the system state variables"
    u_init::Vector{T}
    "The indexes of the partial independent variables in the system parameter value vector"
    pvidx::Vector{Int}
    "The discretized values of the partial independent variables"
    grid::TG
    "Functions to get the current values of the observed variables with input arguments of time and the partial independent variables"
    obs_fs::FT1
    "Indexes for the obs_fs functions"
    obs_fs_idx::Dict{Num,Int}
    "Functions to get the current values of the coordinate transforms with input arguments of time and the partial independent variables"
    tf_fs::FT2

    "Internal integrators"
    integrators::Vector{IT}
    "Internal chunks of grid cells for each integrator"
    IIchunks::Vector{SubArray{CartesianIndex{3},1,Base.ReshapedArray{CartesianIndex{3},1,CartesianIndices{3,Tuple{Base.OneTo{Int64},Base.OneTo{Int64},Base.OneTo{Int64}}},Tuple{Base.SignedMultiplicativeInverse{Int64},Base.SignedMultiplicativeInverse{Int64}}},Tuple{UnitRange{Int64}},false}}

    function Simulator(sys::CoupledSystem, Δs::AbstractVector{T2}, algorithm; kwargs...) where {T2<:AbstractFloat}
        @assert !isnothing(sys.domaininfo) "The system must have a domain specified; see documentation for EarthSciMLBase.DomainInfo."
        mtk_sys = structural_simplify(get_mtk_ode(sys; name=:model))
        start, finish = time_range(sys.domaininfo)
        prob = ODEProblem(mtk_sys, [], (start, finish), []; kwargs...)
        vars = states(mtk_sys)
        ps = parameters(mtk_sys)

        dflts = ModelingToolkit.get_defaults(mtk_sys)
        pvals = [dflts[p] for p ∈ ps]
        uvals = [dflts[u] for u ∈ vars]

        iv = ivar(sys.domaininfo)
        pv = pvars(sys.domaininfo)
        @assert length(pv) == 3 "Currently only 3D simulations are supported."
        pvidx = [findfirst(isequal(p), parameters(mtk_sys)) for p in pv]

        # Get functions for observed variables
        obs_fs_idx = Dict()
        obs_fs = []
        for (i, x) ∈ enumerate([eq.lhs for eq ∈ observed(mtk_sys)])
            obs_fs_idx[x] = i
            push!(obs_fs, observed_function(mtk_sys, x, [iv, pv...]))
        end
        obs_fs = Tuple(obs_fs)

        # Get functions for coordinate transforms
        tf_fs = []
        @variables 🌈🐉🏒 # Dummy variable.
        for tf ∈ partialderivative_transforms(sys.domaininfo)
            push!(tf_fs, observed_function(mtk_sys, 🌈🐉🏒, [iv, pv...], extra_eqs=[🌈🐉🏒 ~ tf]))
        end
        tf_fs = Tuple(tf_fs)

        T = utype(sys.domaininfo)

        grd = grid(sys.domaininfo, Δs)
        TG = typeof(grd)

        u = Array{T}(undef, length(uvals), length(grd[1]), length(grd[2]), length(grd[3]))
        du = similar(u)

        II = CartesianIndices(size(u)[2:4])
        IIchunks = collect(Iterators.partition(II, length(II) ÷ Threads.nthreads()))
        integrators = [init(remake(prob, u0=similar(uvals), p=deepcopy(pvals)), algorithm, save_on=false,
            save_start=false, save_end=false, initialize_save=false; kwargs...)
                       for _ in 1:length(IIchunks)]

        new{T,typeof(integrators[1]),typeof(obs_fs),typeof(tf_fs), TG}(sys, mtk_sys, sys.domaininfo, u, du, pvals, uvals, pvidx, grd, obs_fs, obs_fs_idx, tf_fs, integrators, IIchunks)
    end
end

"Take a step using the ODE solver."
function ode_step!(s::Simulator{T,IT,FT,FT2,TG}, time::T, step_length::T) where {T,IT,FT,FT2,TG}
    tasks = map(1:length(s.IIchunks)) do ithread
        Threads.@spawn single_ode_step!(s, $ithread, time, step_length)
    end::Vector{Task}
    wait.(tasks)
    nothing
end

"Take a step using the ODE solver in the thread specified by `ithread`."
function single_ode_step!(s::Simulator{T,IT,FT,FT2,TG}, ithread, time::T, step_length::T) where {T,IT,FT,FT2,TG}
    IIchunk = s.IIchunks[ithread]
    integrator = s.integrators[ithread]
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

"Take a step using the operator functions."
function operator_step!(s::Simulator{T,IT,FT,FT2,TG}, time::T, step_length::T) where {T,IT,FT,FT2,TG}
    for op in s.sys.ops
        s.du .= zero(eltype(s.du))
        run!(op, s, time)
        @. s.u += s.du * step_length
    end
    nothing
end

"Take a step using Strang splitting, first with the ODE solver, then with the operators."
function strang_step!(s::Simulator{T,IT,FT,FT2,TG}, time::T, step_length::T) where {T,IT,FT,FT2,TG}
    ode_step!(s, time, step_length)
    operator_step!(s, time, step_length)
    nothing
end

"Initialize the state variables."
function init_u!(s::Simulator)
    # Set initial conditions
    for i ∈ eachindex(s.u_init)
        for j ∈ eachindex(s.grid[1])
            for k ∈ eachindex(s.grid[2])
                for l ∈ eachindex(s.grid[3])
                    s.u[i, j, k, l] = s.u_init[i]
                end
            end
        end
    end
    nothing
end

"""
$(TYPEDSIGNATURES)

Run the simulation
"""
function run!(s::Simulator{T,IT,FT,FT2,TG}) where {T,IT,FT,FT2,TG}
    start, finish = time_range(s.domaininfo)
    if length(s.sys.ops) > 0
        optimes = [start:T(timestep(op)):finish for op ∈ s.sys.ops]
        steps = timesteps(optimes...)
        step_length = steplength(steps)
    else
        steps = [start,finish]
        step_length = finish - start
    end
    init_u!(s)

    for op ∈ s.sys.ops
        initialize!(op, s)
    end        
    for time in steps
        strang_step!(s, time, step_length)
    end
    for op ∈ s.sys.ops
        finalize!(op, s)
    end
    nothing
end