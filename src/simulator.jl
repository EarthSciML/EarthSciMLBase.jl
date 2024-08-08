export Simulator, init_u

"""
$(TYPEDSIGNATURES)

Specify a simulator for large-scale model runs. 
`Δs` represent the grid cell spacing in each dimension; for example `Δs = [0.1, 0.1, 1]` 
would represent a grid with 0.1 spacing in the first two dimensions and 1 in the third,
in whatever units the grid is natively in.
The grid spacings should be given in the same order as the partial independent variables
are in the provided `DomainInfo`.

$(TYPEDFIELDS)
"""
struct Simulator{T,FT1,FT2,TG}
    "The system to be integrated"
    sys::CoupledSystem
    "The ModelingToolkit version of the system"
    sys_mtk::ODESystem
    "Information about the spatiotemporal simulation domain"
    domaininfo::DomainInfo{T}
    "The system parameter values"
    p::Vector{T}
    "The initial values of the system state variables"
    u_init::Vector{T}
    "The indexes of the partial independent variables in the system parameter value vector"
    pvidx::Vector{Int}
    "The discretized values of the partial independent variables"
    grid::TG
    "The spacings for each grid dimension"
    Δs::Tuple{T,T,T}
    "Functions to get the current values of the observed variables with input arguments of time and the partial independent variables"
    obs_fs::FT1
    "Indexes for the obs_fs functions"
    obs_fs_idx::Dict{Num,Int}
    "Functions to get the current values of the coordinate transforms with input arguments of time and the partial independent variables"
    tf_fs::FT2

    function Simulator(sys::CoupledSystem, Δs::AbstractVector{T2}) where {T2<:AbstractFloat}
        @assert !isnothing(sys.domaininfo) "The system must have a domain specified; see documentation for EarthSciMLBase.DomainInfo."
        mtk_sys = structural_simplify(get_mtk_ode(sys; name=:model))

        mtk_sys, obs_eqs = prune_observed(mtk_sys) # Remove unused variables to speed up computation.

        vars = states(mtk_sys)
        ps = parameters(mtk_sys)

        dflts = ModelingToolkit.get_defaults(mtk_sys)
        pvals = [dflts[p] for p ∈ ps]
        uvals = [dflts[u] for u ∈ vars]

        iv = ivar(sys.domaininfo)
        pv = pvars(sys.domaininfo)
        @assert length(pv) == 3 "Currently only 3D simulations are supported."
        pvidx = [findfirst(v -> split(String(Symbol(v)), "₊")[end] == String(Symbol(p)), parameters(mtk_sys)) for p ∈ pv]

        # Get functions for observed variables
        obs_fs_idx = Dict()
        obs_fs = []
        for (i, x) ∈ enumerate([eq.lhs for eq ∈ obs_eqs])
            obs_fs_idx[x] = i
            push!(obs_fs, observed_function(obs_eqs, x, [iv, pv...]))
        end
        obs_fs = Tuple(obs_fs)

        # Get functions for coordinate transforms
        tf_fs = []
        @variables 🌈🐉🏒 # Dummy variable.
        for tf ∈ partialderivative_transforms(sys.domaininfo)
            push!(tf_fs, observed_function([obs_eqs..., 🌈🐉🏒 ~ tf], 🌈🐉🏒, [iv, pv...]))
        end
        tf_fs = Tuple(tf_fs)

        T = utype(sys.domaininfo)

        grd = grid(sys.domaininfo, Δs)
        TG = typeof(grd)

        new{T,typeof(obs_fs),typeof(tf_fs),TG}(sys, mtk_sys, sys.domaininfo, pvals, uvals, pvidx, grd, tuple(Δs...), obs_fs, obs_fs_idx, tf_fs)
    end
end

function Base.show(io::IO, s::Simulator)
    print(io, "Simulator{$(utype(s.domaininfo))} with $(length(equations(s.sys_mtk))) equation(s), $(length(s.sys.ops)) operator(s), and $(*([length(g) for g in s.grid]...)) grid cells.")
end

"Initialize the state variables."
function init_u(s::Simulator{T}) where T
    u = Array{T}(undef, size(s)...)
    # Set initial conditions
    for i ∈ eachindex(s.u_init), j ∈ eachindex(s.grid[1]), k ∈ eachindex(s.grid[2]), l ∈ eachindex(s.grid[3])
        u[i, j, k, l] = s.u_init[i]
    end
    u
end

function get_callbacks(s::Simulator)
    extra_cb = [init_callback(c, s) for c ∈ s.sys.init_callbacks]
    [s.sys.callbacks; extra_cb]
end

Base.size(s::Simulator) = (length(states(s.sys_mtk)), [length(g) for g ∈ s.grid]...)
Base.length(s::Simulator) = *(size(s)...)