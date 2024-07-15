
"""
Operators are objects that modify the current state of a `Simulator` system.
Each operator should be define a `run` function with the signature:

    `run!(op::Operator, s::Simulator, time)`

which modifies the `s.du` field in place. It should also implement:

    `timestep(op::Operator)`

which returns the timestep length for the operator.
"""
abstract type Operator end

"""
$(TYPEDSIGNATURES)

Specify a simulator for large-scale model runs.

$(TYPEDFIELDS)
"""
struct Simulator{T}
    "The system to be integrated"
    sys::CoupledSystem
    "The ModelingToolkit version of the system"
    sys_mtk::ODESystem
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
    grid::Vector{AbstractRange{T}}
    "Functions to get the current values of the observed variables with input arguments of time and the partial independent variables"
    obs_fs::Dict{Any,Function}
    "Functions to get the current values of the coordinate transforms with input arguments of time and the partial independent variables"
    tf_fs::Vector{Function}

    "Internal integrators"
    integrators::Vector{OrdinaryDiffEq.ODEIntegrator}
    "Internal chunks of grid cells for each integrator"
    IIchunks

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
        obs_fs = Dict()
        for x ∈ [eq.lhs for eq ∈ observed(mtk_sys)]
            obs_fs[x] = observed_function(mtk_sys, x, [iv, pv...])
        end

        # Get functions for coordinate transforms
        tf_fs = []
        @variables 🌈🐉🏒 # Dummy variable.
        for tf ∈ partialderivative_transforms(sys.domaininfo)
            push!(tf_fs, observed_function(mtk_sys, 🌈🐉🏒, [iv, pv...], extra_eqs=[🌈🐉🏒 ~ tf]))
        end

        T = utype(sys.domaininfo)

        grd = grid(sys.domaininfo, Δs)

        u = Array{T}(undef, length(uvals), length(grd[1]), length(grd[2]), length(grd[3]))
        du = similar(u)

        II = CartesianIndices(size(u)[2:4])
        IIchunks = Iterators.partition(II, length(II) ÷ Threads.nthreads())
        integrators = [init(prob, algorithm, save_on=false,
            save_start=false, save_end=false, initialize_save=false)
                       for _ in 1:length(IIchunks)]

        new{T}(sys, mtk_sys, u, du, pvals, uvals, pvidx, grd, obs_fs, tf_fs, integrators, IIchunks)
    end
end

function ode_step!(s::Simulator{T}, time::T, step_length::T) where {T}
    tasks = map(1:length(s.IIchunks)) do ithread
        Threads.@spawn begin
            IIchunk = collect(s.IIchunks)[$ithread]
            integrator = s.integrators[$ithread]
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
    end
    wait.(tasks)
    nothing
end

function operator_step!(s::Simulator{T}, time::T, step_length::T) where {T}
    for op in s.sys.ops
        s.du .= zero(eltype(s.du))
        run!(op, s, time)
        @. s.u += s.du * step_length
    end
    nothing
end

function strang_step!(s::Simulator{T}, time::T, step_length::T) where {T}
    ode_step!(s, time, step_length)
    operator_step!(s, time, step_length)
    nothing
end

function steplength(timesteps)
    Δs = [timesteps[i] - timesteps[i-1] for i ∈ 2:length(timesteps)]
    @assert all(Δs[1] .≈ Δs) "Not all time steps are the same."
    return Δs[1]
end

@test steplength([0, 0.1, 0.2]) == 0.1

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

function run!(s::Simulator{T}) where {T}
    start, finish = time_range(s.sys.domaininfo)
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

using Test

@parameters x y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0
)
Dt = Differential(t)

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat,
    x ~ 2α + p + y,
    y ~ 4α - 2p
]


@named sys = ODESystem(eqs)
sys = structural_simplify(sys)
observed(sys)

"""
$(SIGNATURES)

Return an expression for the observed value of a variable `x` after
substituting in the constants observed values of other variables.
`extra_eqs` is a list of additional equations to use in the substitution.
"""
function observed_expression(sys::ODESystem, x; extra_eqs=[])
    expr = nothing
    eqs = observed(sys)
    push!(eqs, extra_eqs...)
    for eq ∈ eqs
        if isequal(eq.lhs, x)
            expr = eq.rhs
        end
    end
    if isnothing(expr)
        return nothing
    end
    expr = ModelingToolkit.subs_constants(expr)
    for v ∈ Symbolics.get_variables(expr)
        v_expr = observed_expression(sys, v)
        if !isnothing(v_expr)
            expr = Symbolics.replace(expr, v => v_expr)
        end
    end
    # Do it again to catch extra variables TODO(CT): Theoretically this could recurse forever; when to stop?
    for v ∈ Symbolics.get_variables(expr)
        v_expr = observed_expression(sys, v, extra_eqs=extra_eqs)
        if !isnothing(v_expr)
            expr = Symbolics.replace(expr, v => v_expr)
        end
    end
    expr
end

"""
$(SIGNATURES)

Return a function to  for the observed value of a variable `x` based
on the input arguments in `coords`.
`extra_eqs` is a list of additional equations to use to determine 
the value of `x`.
"""
function observed_function(sys::ODESystem, x, coords; extra_eqs=[])
    expr = observed_expression(sys, x, extra_eqs=extra_eqs)
    vars = Symbolics.get_variables(expr)
    @assert (length(vars) <= length(coords)) "Extra variables: $(vars) != $(coords)"
    @assert all(Bool.([sum(isequal.((v,), coords)) for v ∈ vars])) "Incorrect variables: $(vars) != $(coords)"
    return Symbolics.build_function(expr, coords...; expression=Val{false})
end

xx = observed_expression(sys, x)

@test isequal(xx, -1.0 + 6α)

coords = [α]
xf = observed_function(sys, x, coords)

@test isequal(xf(α), -1.0 + 6α)
@test isequal(xf(2), -1.0 + 6 * 2)


@variables uu, vv
extra_eqs = [uu ~ x + 3, vv ~ uu * 4]

xx2 = observed_expression(sys, vv, extra_eqs=extra_eqs)

xf2 = observed_function(sys, vv, coords, extra_eqs=extra_eqs)

@test isequal(xf2(α), 4 * (2 + 6α))



prob = ODEProblem(sys, tspan=(0.0, 1.0))

sol = solve(prob, Tsit5(), tspan=(0.0, 11.5))

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...))

vars = states(sys)

bcs = icbc(domain, vars)


"""
$(SIGNATURES)

Return the data type of the state variables for this domain,
based on the data types of the boundary conditions domain intervals.
"""
function utype(d::DomainInfo)
    T = []
    for icbc ∈ d.icbc
        if icbc isa BCcomponent
            for pd ∈ icbc.partialdomains
                push!(T, eltype(pd.domain))
            end
        end
    end
    @assert length(T) > 0 "This domain does not include any boundary conditions"
    return promote_type(T)[1]
end


@test utype(domain) == Float64
@test utype(DomainInfo(constIC(0, t ∈ Interval(0, 1)), constBC(16.0f0, lon ∈ Interval(0.0f0, 1.0f0)))) == Float32


"""
$(SIGNATURES)

Return the ranges representing the discretization of the partial independent 
variables for this domain, based on the discretization intervals given in `Δs`
"""
function grid(d::DomainInfo, Δs::AbstractVector{T})::Vector{AbstractRange{T}} where {T<:AbstractFloat}
    i = 1
    rngs = []
    for icbc ∈ d.icbc
        if icbc isa BCcomponent
            for pd ∈ icbc.partialdomains
                rng = DomainSets.infimum(pd.domain):Δs[i]:DomainSets.supremum(pd.domain)
                push!(rngs, rng)
                i += 1
            end
        end
    end
    @assert length(rngs) == length(Δs) "The number of partial independent variables ($(length(rng))) must equal the number of Δs provided ($(length(Δs)))"
    return rngs
end

@test grid(domain, [0.1π, 0.01π]) ≈ [-π:0.1π:π, -0.45π:0.01π:0.45π]


"""
$(SIGNATURES)

Return the time range associated with this domain.
"""
function time_range(d::DomainInfo)
    for icbc ∈ d.icbc
        if icbc isa ICcomponent
            return DomainSets.infimum(icbc.indepdomain.domain), DomainSets.supremum(icbc.indepdomain.domain)
        end
    end
    ArgumentError("Could not find a time range for this domain.")
end

"""
$(SIGNATURES)

Return the time points during which integration should be stopped to run the operators.
"""
function timesteps(tsteps...)
    allt = sort(union(vcat(tsteps...)))
    allt2 = [allt[1]]
    for i ∈ 2:length(allt) # Remove nearly duplicate times.
        if allt[i] ≉ allt[i-1]
            push!(allt2, allt[i])
        end
    end
    allt2
end

@test timesteps(0:0.1:1, 0:0.15:1) == [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

@test timesteps(0:0.1:0.3, 0:0.1000000000001:0.3) == [0.0, 0.1, 0.2, 0.3]

mutable struct ExampleOp <: Operator
    α # Multiplier from ODESystem
end

function run!(op::ExampleOp, s::Simulator, t)
    f = s.obs_fs[op.α]
    for ix ∈ 1:size(s.u, 1)
        for (i, c1) ∈ enumerate(s.grid[1])
            for (j, c2) ∈ enumerate(s.grid[2])
                for (k, c3) ∈ enumerate(s.grid[3])
                    # Demonstrate coordinate transforms
                    t1 = s.tf_fs[1](t, c1, c2, c3)
                    t2 = s.tf_fs[2](t, c1, c2, c3)
                    t3 = s.tf_fs[3](t, c1, c2, c3)
                    # Demonstrate calculating observed value.
                    fv = f(t, c1, c2, c3)
                    # Set derivative value.
                    s.du[ix, i, j, k] = (t1 + t2 + t3) * fv
                end
            end
        end
    end
end

timestep(op::ExampleOp) = 1.0

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...))

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0
lat = GlobalScope(lat)
lon = GlobalScope(lon)
lev = GlobalScope(lev)
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0, windspeed(t) = 1.0
)
Dt = Differential(t)

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat,
    windspeed ~ lat + lon + lev,
]
@named sys = ODESystem(eqs, t)

op = ExampleOp(sys.windspeed)

csys = couple(sys, op, domain)


sim = Simulator(csys, [0.1, 0.1, 1], Tsit5())

sim.pvidx

@test 1 / (sim.tf_fs[1](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111319.44444444445
@test 1 / (sim.tf_fs[2](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111320.00000000001
@test sim.tf_fs[3](0.0, 0.0, 0.0, 0.0) == 1.0

@test sim.obs_fs[sys.windspeed](0.0, 1.0, 3.0, 2.0) == 6.0
@test sim.obs_fs[op.α](0.0, 1.0, 3.0, 2.0) == 6.0

run!(op, sim, 0.0)

@test sum(abs.(sim.du)) ≈ 26094.203039436292

operator_step!(sim, 0.0, 1.0)

@test sum(abs.(sim.du)) ≈ 26094.203039436292

init_u!(sim)

ode_step!(sim, 0.0, 1.0)

@test sum(abs.(sim.u)) ≈ 182879.67315985882

run!(sim)

@test sum(abs.(sim.u)) ≈ 3.776280791609253e7
