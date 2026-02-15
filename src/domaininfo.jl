export DomainInfo, ICBCcomponent, constIC, constBC, zerogradBC, periodicBC,
       partialderivatives, get_tref, get_tspan, get_tspan_datetime

"""
Initial and boundary condition components that can be combined to
create an DomainInfo object.

$(METHODLIST)
"""
abstract type ICBCcomponent end
abstract type ICcomponent <: ICBCcomponent end
abstract type BCcomponent <: ICBCcomponent end

"""
$(SIGNATURES)

Domain information for a ModelingToolkit.jl PDESystem.
It can be used with the `+` operator to add initial and boundary conditions
and coordinate transforms to a
ModelingToolkit.jl System or Catalyst.jl ReactionSystem.

**NOTE**: The independent variable (usually time) must be first in the list of initial and boundary conditions.

$(FIELDS)
"""
struct DomainInfo{ET, AT}
    """
    Function that returns spatial derivatives of the partially-independent variables,
    optionally performing a coordinate transformation first.

    Current function options in this package are:
    - `partialderivatives_δxyδlonlat`: Returns partial derivatives after transforming any variables named `lat` and `lon`
    from degrees to cartesian meters, assuming a spherical Earth.

    Other packages may implement additional functions. They are encouraged to use function names starting
    with `partialderivatives_`.
    """
    partial_derivative_funcs::Vector{Function}

    "The discretization intervals for the partial independent variables."
    grid_spacing::Vector{Float64} # Use Float64 grid spacing to avoid precision issues.

    "The sets of initial and/or boundary conditions."
    icbc::Vector{ICBCcomponent}

    "The spatial reference system for the domain."
    spatial_ref::Any

    """
    The prototype state array for the domain.
    """
    u_proto::AT

    """
    The reference time for the domain, relative to which the simulation time
    period will be calculated.
    """
    tref::ET

    function DomainInfo(pdfs, gs, icbc, sr, t_ref::ET) where {ET}
        new{ET, Vector{ET}}(pdfs, gs, icbc, sr, zeros(ET, 0), t_ref)
    end
    function DomainInfo(
            icbc::ICBCcomponent...; u_proto::AT = zeros(0), grid_spacing = nothing,
            spatial_ref = "+proj=longlat +datum=WGS84 +no_defs") where {AT <: AbstractArray}
        @assert length(icbc)>0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        et = eltype(u_proto)
        grid_spacing = isnothing(grid_spacing) ? defaultgridspacing(et, icbc) : grid_spacing
        new{et, AT}([], grid_spacing, ICBCcomponent[icbc...], spatial_ref, u_proto, 0)
    end
    function DomainInfo(fdx::Function, icbc::ICBCcomponent...; grid_spacing = nothing,
            u_proto::AT = zeros(0),
            spatial_ref = "+proj=longlat +datum=WGS84 +no_defs") where {AT <: AbstractArray}
        @assert length(icbc)>0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        et = eltype(u_proto)
        grid_spacing = isnothing(grid_spacing) ? defaultgridspacing(et, icbc) : grid_spacing
        new{et, AT}([fdx], grid_spacing, ICBCcomponent[icbc...], spatial_ref, u_proto, 0)
    end
    function DomainInfo(fdxs::Vector{Function}, icbc::ICBCcomponent...;
            u_proto::AT = zeros(0), grid_spacing = nothing,
            spatial_ref = "+proj=longlat +datum=WGS84 +no_defs") where {AT <: AbstractArray}
        @assert length(icbc)>0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        et = eltype(u_proto)
        grid_spacing = isnothing(grid_spacing) ? defaultgridspacing(et, icbc) : grid_spacing
        new{et, AT}(fdxs, grid_spacing, ICBCcomponent[icbc...], spatial_ref, u_proto, 0)
    end
    function DomainInfo(starttime::DateTime, endtime::DateTime;
            xrange = nothing, yrange = nothing, levrange = nothing,
            latrange = nothing, lonrange = nothing, u_proto::AT = zeros(0),
            level_trans = nothing, tref = starttime,
            spatial_ref = "+proj=longlat +datum=WGS84 +no_defs") where {AT <: AbstractArray}
        et = eltype(u_proto)
        @assert et(datetime2unix(starttime))<et(datetime2unix(endtime)) "starttime must be before endtime when represented as $et."
        @assert (!isnothing(xrange) &&
                 !isnothing(yrange)) ||
                (!isnothing(latrange) && !isnothing(lonrange)) "Either x and y ranges or lat and lon ranges must be provided."
        @assert (isnothing(xrange) &&
                 isnothing(yrange)) ||
                (isnothing(latrange) && isnothing(lonrange)) "Either x and y ranges or lat and lon ranges must be provided, not both."

        # Coordinate transforms
        fdxs = Vector{Function}()
        if !isnothing(latrange) # Convert lat/lon to meters.
            push!(fdxs, partialderivatives_δxyδlonlat)
        end
        !isnothing(level_trans) ? push!(fdxs, level_trans) : nothing

        tref = tref isa DateTime ? datetime2unix(tref) : tref
        ic = constIC(et(0.0),
            ModelingToolkit.t ∈
            Interval(et.(datetime2unix.([starttime, endtime]) .- tref)...))

        boundaries = [] # Boundary conditions
        grid_spacing = []
        gridT = Float64 # Grid spacing is always Float64 to avoid precision issues.
        if !isnothing(latrange)
            @assert maximum(abs.(latrange))<=π "Latitude must be in radians."
            @assert maximum(abs.(lonrange))<=2π "Longitude must be in radians."
            lon = only(@parameters lon=mean(lonrange) [
                unit = u"rad", description = "Longitude"])
            lat = only(@parameters lat=mean(latrange) [
                unit = u"rad", description = "Latitude"])
            push!(boundaries, lon ∈ Interval(et.([first(lonrange), last(lonrange)])...))
            push!(boundaries, lat ∈ Interval(et.([first(latrange), last(latrange)])...))
            push!(grid_spacing, gridT.([step(lonrange), step(latrange)])...)
        else
            x = only(@parameters x=mean(xrange) [
                unit = u"m", description = "East-West Distance"])
            y = only(@parameters y=mean(yrange) [
                unit = u"m", description = "North-South Distance"])
            push!(boundaries, x ∈ Interval(et.([first(xrange), last(xrange)])...))
            push!(boundaries, y ∈ Interval(et.([first(yrange), last(yrange)])...))
            push!(grid_spacing, gridT.([step(xrange), step(yrange)])...)
        end
        if !isnothing(levrange)
            lev = only(@parameters lev=mean(levrange) [description = "Level Index"])
            push!(boundaries, lev ∈ Interval(et.([first(levrange), last(levrange)])...))
            push!(grid_spacing, gridT(step(levrange)))
        end
        bcs = constBC(et(0.0), boundaries...)
        new{et, AT}(fdxs, grid_spacing, ICBCcomponent[ic, bcs], spatial_ref, u_proto, tref)
    end
end

function defaultgridspacing(et, icbc)
    ndims = length(filter(icbc -> icbc isa BCcomponent, icbc))
    return ones(et, ndims)
end

Base.size(d::DomainInfo) = tuple((length(g) for g in grid(d))...)
function Base.size(d::DomainInfo, staggering::NTuple{3, Bool})
    tuple((length(g) for g in grid(d, staggering))...)
end
Base.size(d::DomainInfo, i) = length(grid(d)[i])
Base.size(d::DomainInfo, staggering::NTuple{3, Bool}, i) = length(grid(d, staggering)[i])

"""
$(TYPEDSIGNATURES)

Return the scalar data type of the state variable elements for this domain.
"""
Base.eltype(_::DomainInfo{ET}) where {ET} = ET
Base.@deprecate dtype(d) eltype(d)

"""
$(SIGNATURES)

Return the ranges representing the discretization of the partial independent
variables for this domain, based on the discretization intervals given in `Δs`.
"""
function grid(d::DomainInfo{T}) where {T}
    endpts = endpoints(d)
    [s:d:e for ((s, e), d) in zip(endpts, d.grid_spacing)]
end
function grid(d::DomainInfo{T}, staggering) where {T}
    endpts = endpoints(d)
    @assert length(staggering)==length(endpts) "The number of staggering values $(length(staggering)) must match the number of partial independent variables $(length(endpts))."
    @assert all(isa.(staggering, (Bool,))) "Staggering must be a vector of booleans."
    [stag ? range(start = s - d / 2, step = d, length = length(s:d:e) + 1) : s:d:e
     for (stag, (s, e), d) in zip(staggering, endpts, d.grid_spacing)]
end

"""
$(SIGNATURES)

Return the concrete grid representation for this domain, as a Vector including the grid
points for the entire 3D domain.
"""
function concrete_grid(domain::DomainInfo{ET, AT}) where {ET, AT}
    g = grid(domain)
    II = CartesianIndices(tuple(size(domain)...))
    map(enumerate(g)) do (j, c)
        # Collect the grid points and convert them to the correct array type.
        _grd = [c[II[i][j]] for i in 1:length(II)]
        grd = similar(domain.u_proto, length(_grd))
        copyto!(grd, _grd)
    end
end

"""
$(SIGNATURES)

Return the endpoints of the partial independent
variables for this domain.
"""
function endpoints(d::DomainInfo)
    T = Float64 # Endpoints are always Float64 to avoid rounding issues.
    bcs = filter((icbc) -> icbc isa BCcomponent, d.icbc)
    rngs = NTuple{2, T}[]
    for bc in bcs
        for pd in bc.partialdomains
            rng = (T(DomainSets.infimum(pd.domain)), T(DomainSets.supremum(pd.domain)))
            push!(rngs, rng)
        end
    end
    return rngs
end

"""
$(TYPEDSIGNATURES)

Return the time range associated with this domain, returning the values as Unix times
relative to the reference time `tref`.
"""
function get_tspan(d::DomainInfo{T})::Tuple{T, T} where {T <: AbstractFloat}
    for icbc in d.icbc
        if icbc isa ICcomponent
            return DomainSets.infimum(icbc.indepdomain.domain),
            DomainSets.supremum(icbc.indepdomain.domain)
        end
    end
    throw(ArgumentError("Could not find a time range for this domain."))
end

"""
$(TYPEDSIGNATURES)

Return the reference time for this simulation.
"""
get_tref(d::DomainInfo) = d.tref

"""
$(SIGNATURES)

Return the time range associated with this domain, returning the values as DateTimes.
"""
function get_tspan_datetime(d::DomainInfo)
    (Float64.(get_tspan(d)) .+ Float64(get_tref(d))) .|> unix2datetime
end

"""
$(SIGNATURES)

Return a vector of equations that define the initial and boundary conditions for the
given state variables.
"""
function icbc(di::DomainInfo, states::AbstractVector)::Vector{Equation}
    ic = di.icbc[findall(icbc -> isa(icbc, ICcomponent), di.icbc)]
    @assert length(ic)==1 "Only one independent domain is allowed."

    bcs = di.icbc[findall(icbc -> isa(icbc, BCcomponent), di.icbc)]
    partialdomains = vcat([bc.partialdomains for bc in bcs]...)
    @assert length(partialdomains)>0 "At least one partial domain is required."
    @assert length(unique(partialdomains))==length(partialdomains) "Each partial domain must have only one set of boundary conditions."
    o = [icbc(states, ic[1].indepdomain, partialdomains) for icbc in di.icbc]
    vcat(o...)
end

"""
$(TYPEDSIGNATURES)

Return the independent variable associated with these
initial and boundary conditions.
"""
function ivar(di::DomainInfo)
    ic = di.icbc[findall(icbc -> isa(icbc, ICcomponent), di.icbc)]
    @assert length(ic)==1 "Only one independent domain is allowed."
    return ic[1].indepdomain.variables
end

"""
$(TYPEDSIGNATURES)

Return the partial independent variables associated with these
initial and boundary conditions.
"""
function pvars(di::DomainInfo)
    bcs = di.icbc[findall(icbc -> isa(icbc, BCcomponent), di.icbc)]
    partialdomains = vcat([bc.partialdomains for bc in bcs]...)
    @assert length(partialdomains)>0 "At least one partial domain is required."
    @assert length(unique(partialdomains))==length(partialdomains) "Each partial domain must have only one set of boundary conditions."
    return [domain.variables for domain in partialdomains]
end

"""
$(TYPEDSIGNATURES)

Return transform factor to multiply each partial derivative operator by,
for example to convert from degrees to meters.
"""
function partialderivative_transforms(mtk_sys::System, di::DomainInfo)
    xs = coord_params(mtk_sys, di)
    partialderivative_transforms(xs, di)
end
function partialderivative_transforms(di::DomainInfo)
    xs = pvars(di)
    partialderivative_transforms(xs, di)
end
function partialderivative_transforms(xs, di::DomainInfo)
    fs = Dict()
    for f in di.partial_derivative_funcs
        for (k, v) in f(xs)
            @assert k∉keys(fs) "Multiple transforms were specified for $(xs[k])."
            fs[k] = v
        end
    end
    ts = []
    for i in eachindex(xs)
        if i in keys(fs)
            push!(ts, fs[i])
        else
            push!(ts, 1)
        end
    end
    ts
end

function partialderivative_transform_vars(mtk_sys, di::DomainInfo)
    xs = coord_params(mtk_sys, di)
    iv = ivar(di)
    ts = partialderivative_transforms(mtk_sys, di)
    vs = []
    for (i, x) in enumerate(xs)
        n = Symbol("δ$(x)_transform")
        v = only(@variables $n(iv) [unit = ModelingToolkit.get_unit(ts[i]),
            description = "Transform factor for $(x)"])
        push!(vs, v)
    end
    vs
end

function partialderivative_transform_eqs(mtk_sys, di::DomainInfo)
    vs = partialderivative_transform_vars(mtk_sys, di)
    ts = partialderivative_transforms(mtk_sys, di)
    eqs = vs .~ ts
    System(eqs, ivar(di); name = :transforms)
end

"""
$(TYPEDSIGNATURES)

Return the partial derivative operators for the given domain.
"""
function partialderivatives(di::DomainInfo)
    xs = pvars(di)
    δs = Differential.(xs)
    ts = partialderivative_transforms(di)
    [(x) -> (δs[i](x) * ts[i]) for i in eachindex(xs)]
end

"""
$(TYPEDSIGNATURES)

Construct constant initial conditions equal to the value
specified by `val`.

$(FIELDS)
"""
struct constIC <: ICcomponent
    "The value of the constant initial conditions."
    val::Number
    "The independent domain, e.g. `t ∈ Interval(t_min, t_max)`."
    indepdomain::Symbolics.VarDomainPairing
end

function (ic::constIC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing,
        allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])

    bcs = Equation[]

    for state in statevars
        push!(bcs, state.val.f(indepdomain.domain.left, dims...) ~ ic.val)
    end

    bcs
end

"""
$(TYPEDSIGNATURES)

Construct constant boundary conditions equal to the value
specified by `val`.

$(FIELDS)
"""
struct constBC <: BCcomponent
    "The value of the constant boundary conditions."
    val::Number
    "The partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}

    function constBC(val::Number, partialdomains::Symbolics.VarDomainPairing...)
        new(val, [partialdomains...])
    end
end

function (bc::constBC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing,
        allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])

    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)

    for state in statevars
        for (j, i) in enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            for edge in [domain.domain.left, domain.domain.right]
                args = Any[indepdomain.variables, dims...]
                args[i + 1] = edge
                push!(bcs, state.val.f(args...) ~ bc.val)
            end
        end
    end

    bcs
end

"""
$(TYPEDSIGNATURES)

Construct zero-gradient boundary conditions for the given `partialdomains`.

$(FIELDS)
"""
struct zerogradBC <: BCcomponent
    "The partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}

    zerogradBC(partialdomains::Symbolics.VarDomainPairing...) = new([partialdomains...])
end

function (bc::zerogradBC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing,
        allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])

    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)

    for state in statevars
        for (j, i) in enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            for edge in [domain.domain.left, domain.domain.right]
                args = Any[indepdomain.variables, dims...]
                args[i + 1] = edge
                D = Differential(dims[i])
                push!(bcs, D(state.val.f(args...)) ~ 0.0)
            end
            j += 1
        end
    end

    bcs
end

"""
$(TYPEDSIGNATURES)

Construct periodic boundary conditions for the given `partialdomains`.
Periodic boundary conditions are defined as when the value at one
side of the domain is set equal to the value at the other side, so
that the domain "wraps around" from one side to the other.

$(FIELDS)
"""
struct periodicBC <: BCcomponent
    "The partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}

    periodicBC(partialdomains::Symbolics.VarDomainPairing...) = new([partialdomains...])
end

function (bc::periodicBC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing,
        allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])

    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)

    for state in statevars
        for (j, i) in enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            argsleft = Any[indepdomain.variables, dims...]
            argsleft[i + 1] = domain.domain.left
            argsright = Any[indepdomain.variables, dims...]
            argsright[i + 1] = domain.domain.right
            push!(bcs, state.val.f(argsleft...) ~ state.val.f(argsright...))
        end
    end

    bcs
end

"""
$(TYPEDSIGNATURES)

Returns the dimensions of the independent and partial domains associated with these
initial or boundary conditions.
"""
dims(icbc::ICcomponent) = Num[icbc.indepdomain.variables]
dims(icbc::BCcomponent) = Num[domain.variables for domain in icbc.partialdomains]
dims(di::DomainInfo) = unique(vcat(dims.(di.icbc)...))

"""
$(TYPEDSIGNATURES)

Returns the domains associated with these initial or boundary conditions.
"""
domains(icbc::ICcomponent) = [icbc.indepdomain]
domains(icbc::BCcomponent) = icbc.partialdomains
domains(di::DomainInfo) = unique(vcat(domains.(di.icbc)...))

function Base.:(+)(
        sys::ModelingToolkit.System, di::DomainInfo)::ModelingToolkit.PDESystem
    dimensions = dims(di)
    allvars = unknowns(sys)
    statevars = unknowns(sys)
    ps = parameters(sys)
    toreplace, replacements = replacement_params(ps, pvars(di))
    dvs = add_dims(allvars, dimensions) # Add new dimensions to dependent variables.
    eqs = substitute(equations(sys), Dict(zip(toreplace, replacements))) # Substitute local coordinate parameters for global ones.
    eqs = Vector{Equation}([add_dims(eq, allvars, dimensions) for eq in eqs]) # Add new dimensions to equations.
    PDESystem(
        eqs, icbc(di, statevars), domains(di), dimensions, dvs, ps, name = nameof(sys))
end

Base.:(+)(
    di::DomainInfo, sys::ModelingToolkit.System)::ModelingToolkit.PDESystem = sys + di

# Match local parameters with the global parameters of the same name.
function replacement_params(localcoords::AbstractVector, globalcoords::AbstractVector)
    gcstr = string.(globalcoords)
    lcstr = string.(localcoords)
    toreplace = []
    replacements = []
    for (i, gc) in enumerate(gcstr)
        for (j, lc) in enumerate(lcstr)
            if endswith(lc, "₊" * gc)
                push!(toreplace, localcoords[j])
                push!(replacements, globalcoords[i])
            end
        end
    end
    toreplace, replacements
end

function add_partial_derivative_func(di::DomainInfo, f::Function)
    push!(di.partial_derivative_funcs, f)
    di
end
