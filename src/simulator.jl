
"""
$(TYPEDSIGNATURES)

Specify a simulator for large-scale model runs. 
`kwargs` are passed on to the DifferentialEquations.jl integrator initialization.

$(TYPEDFIELDS)
"""
struct Simulator{T,IT,FT1,FT2}
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
    grid::Vector{StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},Int64}}
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

        u = Array{T}(undef, length(uvals), length(grd[1]), length(grd[2]), length(grd[3]))
        du = similar(u)

        II = CartesianIndices(size(u)[2:4])
        IIchunks = collect(Iterators.partition(II, length(II) ÷ Threads.nthreads()))
        integrators = [init(remake(prob, u0=similar(uvals), p=deepcopy(pvals)), algorithm, save_on=false,
            save_start=false, save_end=false, initialize_save=false; kwargs...)
                       for _ in 1:length(IIchunks)]

        new{T,typeof(integrators[1]),typeof(obs_fs),typeof(tf_fs)}(sys, mtk_sys, sys.domaininfo, u, du, pvals, uvals, pvidx, grd, obs_fs, obs_fs_idx, tf_fs, integrators, IIchunks)
    end
end

function ode_step!(s::Simulator{T,IT,FT,FT2}, time::T, step_length::T) where {T,IT,FT,FT2}
    tasks = map(1:length(s.IIchunks)) do ithread
        Threads.@spawn single_ode_step!(s, $ithread, time, step_length)
    end::Vector{Task}
    wait.(tasks)
    nothing
end

function single_ode_step!(s::Simulator{T,IT,FT,FT2}, ithread, time::T, step_length::T) where {T,IT,FT,FT2}
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

function operator_step!(s::Simulator{T,IT,FT,FT2}, time::T, step_length::T) where {T,IT,FT,FT2}
    for op in s.sys.ops
        s.du .= zero(eltype(s.du))
        run!(op, s, time)
        @. s.u += s.du * step_length
    end
    nothing
end

function strang_step!(s::Simulator{T,IT,FT,FT2}, time::T, step_length::T) where {T,IT,FT,FT2}
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

function run!(s::Simulator)
    start, finish = time_range(s.domaininfo)
    optimes = [start:timestep(op):finish for op ∈ s.sys.ops]
    steps = timesteps(optimes...)
    step_length = steplength(steps)
    init_u!(s)

    #@assert all([x ∈ steps for x ∈ write_times]) "output times must be a subset of time steps"
    #write_step(r.writer, r.u, start)
    #@showprogress for time in steps
    for time in steps
        strang_step!(s, time, step_length)
        #   if time ∈ r.writer.output_times
        #      write_step(r.writer, r.u, time)
        # end
    end
    #    close(o)
    nothing
end