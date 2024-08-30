export DomainInfo, ICBCcomponent, constIC, constBC, zerogradBC, periodicBC, partialderivatives

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
ModelingToolkit.jl ODESystem or Catalyst.jl ReactionSystem.

**NOTE**: The independent variable (usually time) must be first in the list of initial and boundary conditions.

$(FIELDS)

"""
struct DomainInfo{T}
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

    "The sets of initial and/or boundary conditions."
    icbc::Vector{ICBCcomponent}

    function DomainInfo(icbc::ICBCcomponent...; dtype=Float64)
        @assert length(icbc) > 0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        new{dtype}([], ICBCcomponent[icbc...])
    end
    function DomainInfo(fdx::Function, icbc::ICBCcomponent...; dtype=Float64) 
        @assert length(icbc) > 0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        new{dtype}([fdx], ICBCcomponent[icbc...])
    end
    function DomainInfo(fdxs::Vector{Function}, icbc::ICBCcomponent...; dtype=Float64) 
        @assert length(icbc) > 0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        new{dtype}(fdxs, ICBCcomponent[icbc...])
    end
end

"""
$(SIGNATURES)

Return a vector of equations that define the initial and boundary conditions for the 
given state variables.
"""
function icbc(di::DomainInfo, states::AbstractVector)::Vector{Equation}
    ic = di.icbc[findall(icbc -> isa(icbc, ICcomponent), di.icbc)]
    @assert length(ic) == 1 "Only one independent domain is allowed."

    bcs = di.icbc[findall(icbc -> isa(icbc, BCcomponent), di.icbc)]
    partialdomains = vcat([bc.partialdomains for bc ∈ bcs]...)
    @assert length(partialdomains) > 0 "At least one partial domain is required."
    @assert length(unique(partialdomains)) == length(partialdomains) "Each partial domain must have only one set of boundary conditions."
    o = [icbc(states, ic[1].indepdomain, partialdomains) for icbc ∈ di.icbc]
    vcat(o...)
end

"""
$(TYPEDSIGNATURES)

Return the independent variable associated with these 
initial and boundary conditions.
"""
function ivar(di::DomainInfo)
    ic = di.icbc[findall(icbc -> isa(icbc, ICcomponent), di.icbc)]
    @assert length(ic) == 1 "Only one independent domain is allowed."
    return ic[1].indepdomain.variables
end

"""
$(TYPEDSIGNATURES)

Return the partial independent variables associated with these 
initial and boundary conditions.
"""
function pvars(di::DomainInfo)
    bcs = di.icbc[findall(icbc -> isa(icbc, BCcomponent), di.icbc)]
    partialdomains = vcat([bc.partialdomains for bc ∈ bcs]...)
    @assert length(partialdomains) > 0 "At least one partial domain is required."
    @assert length(unique(partialdomains)) == length(partialdomains) "Each partial domain must have only one set of boundary conditions."
    return [domain.variables for domain in partialdomains]
end

"""
$(TYPEDSIGNATURES)

Return transform factor to multiply each partial derivative operator by,
for example to convert from degrees to meters.
"""
function partialderivative_transforms(di::DomainInfo)
    xs = pvars(di)
    fs = Dict()
    for f in di.partial_derivative_funcs
        for (k, v) ∈ f(xs)
            @assert k ∉ keys(fs) "Multiple transforms were specified for $(xs[k])."
            fs[k] = v
        end
    end
    ts = []
    for i ∈ eachindex(xs)
        if i in keys(fs)
            push!(ts, fs[i])
        else
            push!(ts, 1.0)
        end
    end
    ts
end


"""
$(TYPEDSIGNATURES)

Return the partial derivative operators for the given domain.
"""
function partialderivatives(di::DomainInfo)
    xs = pvars(di)
    δs = Differential.(xs)
    ts = partialderivative_transforms(di)
    [(x)->(δs[i](x) * ts[i]) for i ∈ eachindex(xs)]
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

function (ic::constIC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])
    
    bcs = Equation[]
    
    for state ∈ statevars
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

    constBC(val::Number, partialdomains::Symbolics.VarDomainPairing...) = new(val, [partialdomains...])
end

function (bc::constBC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])
    
    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)
    
    for state ∈ statevars
        for (j, i) ∈ enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[indepdomain.variables, dims...]
                args[i+1] = edge
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

function (bc::zerogradBC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])
    
    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)

    for state ∈ statevars
        for (j, i) ∈ enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[indepdomain.variables, dims...]
                args[i+1] = edge
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

function (bc::periodicBC)(states::AbstractVector, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states, [indepdomain.variables, dims...])
    
    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)

    for state ∈ statevars
        for (j, i) ∈ enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            argsleft = Any[indepdomain.variables, dims...]
            argsleft[i+1] = domain.domain.left
            argsright = Any[indepdomain.variables, dims...]
            argsright[i+1] = domain.domain.right
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

function Base.:(+)(sys::ModelingToolkit.ODESystem, di::DomainInfo)::ModelingToolkit.PDESystem
    dimensions = dims(di)
    allvars = unknowns(sys)
    statevars = unknowns(structural_simplify(sys))
    ps = parameters(sys)
    toreplace, replacements = replacement_params(ps, pvars(di))
    dvs = add_dims(allvars, dimensions) # Add new dimensions to dependent variables.
    eqs = substitute(equations(sys), Dict(zip(toreplace, replacements))) # Substitute local coordinate parameters for global ones.
    eqs = Vector{Equation}([add_dims(eq, allvars, dimensions) for eq in eqs]) # Add new dimensions to equations.
    PDESystem(eqs, icbc(di, statevars), domains(di), dimensions, dvs, ps, name=nameof(sys))
end

Base.:(+)(di::DomainInfo, sys::ModelingToolkit.ODESystem)::ModelingToolkit.PDESystem = sys + di

function Base.:(+)(sys::Catalyst.ReactionSystem, di::DomainInfo)::ModelingToolkit.PDESystem
    convert(ODESystem, sys; combinatoric_ratelaws=false) + di
end

Base.:(+)(di::DomainInfo, sys::Catalyst.ReactionSystem)::ModelingToolkit.PDESystem = sys + di

# Match local parameters with the global parameters of the same name.
function replacement_params(localcoords::AbstractVector, globalcoords::AbstractVector)
    gcstr = string.(globalcoords)
    lcstr = string.(localcoords)
    toreplace = []
    replacements = []
    for (i, gc) in enumerate(gcstr)
        for (j, lc) in enumerate(lcstr)
            if endswith(lc, "₊"*gc)
                push!(toreplace, localcoords[j])
                push!(replacements, globalcoords[i])
            end
        end
    end
    toreplace, replacements
end
