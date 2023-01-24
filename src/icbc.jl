export ICBC, ICBCcomponent, constIC, constBC, zerogradBC, periodicBC

"""
Initial and boundary condition components that can be combined to 
create an ICBC object.

$(METHODLIST)
"""
abstract type ICBCcomponent end
abstract type ICcomponent <: ICBCcomponent end
abstract type BCcomponent <: ICBCcomponent end


"""
```julia
ICBC(icbc::ICBCcomponent...)
```

Initial and boundary conditions for a ModelingToolkit.jl PDESystem. 
It can be used with the `+` operator to add initial and boundary conditions to a
ModelingToolkit.jl ODESystem or Catalyst.jl ReactionSystem.

$(FIELDS)

"""
struct ICBC
    "The sets of initial and/or boundary conditions."
    icbc::Vector{ICBCcomponent}

    ICBC(icbc::ICBCcomponent...) = new(ICBCcomponent[icbc...])
end

function (icbc::ICBC)(states::AbstractVector)::Vector{Equation}
    ic = icbc.icbc[findall(icbc -> isa(icbc, ICcomponent), icbc.icbc)]
    @assert length(ic) == 1 "Only one independent domain is allowed."

    bcs = icbc.icbc[findall(icbc -> isa(icbc, BCcomponent), icbc.icbc)]
    partialdomains = vcat([bc.partialdomains for bc ∈ bcs]...)
    @assert length(partialdomains) > 0 "At least one partial domain is required."
    @assert length(unique(partialdomains)) == length(partialdomains) "Each partial domain must have only one set of boundary conditions."
    o = [icbc(states, ic[1].indepdomain, partialdomains) for icbc ∈ icbc.icbc]
    vcat(o...)
end

"""
$(TYPEDSIGNATURES)

Return the independent variable associated with these 
initial and boundary conditions.
"""
function ivar(icbc::ICBC)
    ic = icbc.icbc[findall(icbc -> isa(icbc, ICcomponent), icbc.icbc)]
    @assert length(ic) == 1 "Only one independent domain is allowed."
    return ic[1].indepdomain.variables
end

"""
$(TYPEDSIGNATURES)

Return the partial independent variables associated with these 
initial and boundary conditions.
"""
function pvars(icbc::ICBC)
    bcs = icbc.icbc[findall(icbc -> isa(icbc, BCcomponent), icbc.icbc)]
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
    statevars = add_dims(states, [dims...; indepdomain.variables])
    
    bcs = Equation[]
    
    for state ∈ statevars
        push!(bcs, state.val.f(dims..., indepdomain.domain.left) ~ ic.val)
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
    statevars = add_dims(states, [dims...; indepdomain.variables])
    
    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)
    
    for state ∈ statevars
        for (j, i) ∈ enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[dims..., indepdomain.variables]
                args[i] = edge
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
    statevars = add_dims(states, [dims...; indepdomain.variables])
    
    bcs = Equation[]

    D = Differential(indepdomain.variables)

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)

    for state ∈ statevars
        for (j, i) ∈ enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[dims..., indepdomain.variables]
                args[i] = edge
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
    statevars = add_dims(states, [dims...; indepdomain.variables])
    
    bcs = Equation[]

    activepartialdomainindex = vcat((y -> findall(x -> x == y, allpartialdomains)).(bc.partialdomains)...)

    for state ∈ statevars
        for (j, i) ∈ enumerate(activepartialdomainindex)
            domain = bc.partialdomains[j]
            argsleft = Any[dims..., indepdomain.variables]
            argsleft[i] = domain.domain.left
            argsright = Any[dims..., indepdomain.variables]
            argsright[i] = domain.domain.right
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
dims(icbc::ICBC) = unique(vcat(dims.(icbc.icbc)...))

"""
$(TYPEDSIGNATURES)

Returns the domains associated with these initial or boundary conditions.
"""
domains(icbc::ICcomponent) = [icbc.indepdomain]
domains(icbc::BCcomponent) = icbc.partialdomains
domains(icbc::ICBC) = unique(vcat(domains.(icbc.icbc)...))

function Base.:(+)(sys::ModelingToolkit.ODESystem, icbc::ICBC)::ModelingToolkit.PDESystem
    dimensions = dims(icbc)
    allvars = states(sys)
    statevars = states(structural_simplify(sys))
    # TODO(CT): Update once the MTK get_defaults function can get defaults for composed system.
    # defaults = ModelingToolkit.get_defaults(sys)
    defaults = get_defaults_all(sys)
    ps = [k => v for (k,v) in defaults] # Add parameters and their default values
    if !all([p ∈ keys(defaults) for p in parameters(sys)])
        error("All parameters in the system of equations must have default values.")
    end
    ivs = dims(icbc) # New dimensions are the independent variables.
    dvs = add_dims(allvars, dimensions) # Add new dimensions to dependent variables.
    eqs = Vector{Equation}([add_dims(eq, allvars, dimensions) for eq in equations(sys)]) # Add new dimensions to equations.
    PDESystem(eqs, icbc(statevars), domains(icbc), ivs, dvs, ps, name=nameof(sys), defaults=defaults)
end

Base.:(+)(icbc::ICBC, sys::ModelingToolkit.ODESystem)::ModelingToolkit.PDESystem = sys + icbc

function Base.:(+)(sys::Catalyst.ReactionSystem, icbc::ICBC)::ModelingToolkit.PDESystem
    convert(ODESystem, sys; combinatoric_ratelaws=false) + icbc
end

Base.:(+)(icbc::ICBC, sys::Catalyst.ReactionSystem)::ModelingToolkit.PDESystem = sys + icbc

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