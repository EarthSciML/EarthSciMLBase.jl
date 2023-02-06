export DomainInfo, ICBCcomponent, constIC, constBC, zerogradBC, periodicBC

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
struct DomainInfo
    """
    Function that returns spatial derivatives of the partially-independent variables,
    optionally performing a coordinate transformation first. 

    Current function options are:
    - `partialderivatives_identity` (the default): Returns partial derivatives without performing any coordinate transforms.
    - `partialderivatives_lonlat2xymeters`: Returns partial derivatives after transforming any variables named `lat` and `lon` 
    from degrees to cartesian meters, assuming a spherical Earth.    
    """
    partial_derivative_func::Function

    "The sets of initial and/or boundary conditions."
    icbc::Vector{ICBCcomponent}

    function DomainInfo(icbc::ICBCcomponent...) 
        @assert length(icbc) > 0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        new(partialderivatives_identity, ICBCcomponent[icbc...])
    end
    function DomainInfo(fdx::Function, icbc::ICBCcomponent...) 
        @assert length(icbc) > 0 "At least one initial or boundary condition is required."
        @assert icbc[1] isa ICcomponent "The first initial or boundary condition must be the initial condition for the independent variable."
        new(fdx, ICBCcomponent[icbc...])
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
    allvars = states(sys)
    statevars = states(structural_simplify(sys))
    # TODO(CT): Update once the MTK get_defaults function can get defaults for composed system.
    # defaults = ModelingToolkit.get_defaults(sys)
    toreplace, replacements = replacement_params(parameters(sys), pvars(di))
    defaults = get_defaults_all(sys)
    ps = [k => v for (k,v) in defaults] # Add parameters and their default values
    parameterstokeep = setdiff(parameters(sys), toreplace)
    if !all([p ∈ keys(defaults) for p in parameterstokeep])
        error("All parameters in the system of equations must have default values, but these ones don't: $(setdiff(parameterstokeep, keys(defaults))).")
    end
    ivs = dims(di) # New dimensions are the independent variables.
    dvs = add_dims(allvars, dimensions) # Add new dimensions to dependent variables.
    eqs = substitute(equations(sys), Dict(zip(toreplace, replacements))) # Substitute local coordinate parameters for global ones.
    eqs = Vector{Equation}([add_dims(eq, allvars, dimensions) for eq in eqs]) # Add new dimensions to equations.
    PDESystem(eqs, icbc(di, statevars), domains(di), ivs, dvs, ps, name=nameof(sys), defaults=defaults)
end

Base.:(+)(di::DomainInfo, sys::ModelingToolkit.ODESystem)::ModelingToolkit.PDESystem = sys + di

function Base.:(+)(sys::Catalyst.ReactionSystem, di::DomainInfo)::ModelingToolkit.PDESystem
    convert(ODESystem, sys; combinatoric_ratelaws=false) + di
end

Base.:(+)(di::DomainInfo, sys::Catalyst.ReactionSystem)::ModelingToolkit.PDESystem = sys + di

# TODO(CT): Delete once the MTK get_defaults function can get defaults for composed system.
get_defaults_all(sys) = get_defaults_all(sys, "", 0)
function get_defaults_all(sys, prefix, depth)
    dmap = Dict()
    if depth == 1
        prefix = Symbol("$(nameof(sys))₊")
    elseif depth > 1
        prefix = Symbol("$(prefix)₊$(nameof(sys))₊")
    end
    for (p, d) ∈ ModelingToolkit.get_defaults(sys)
        n = Symbol("$(prefix)$(p)")
        pp = (@parameters $(n))[1]
        dmap[pp] = d
    end
    for child ∈ ModelingToolkit.get_systems(sys)
        dmap = merge(dmap, get_defaults_all(child, prefix, depth+1))
    end
    dmap
end

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
