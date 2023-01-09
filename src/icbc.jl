export ICBC, ICBCcomponent, constIC, constBC, zerogradBC, periodicBC

"""
Initial and boundary condition components that can be combined to 
create an ICBC object.

$(METHODLIST)
"""
abstract type ICBCcomponent end


"""
```julia
ICBC(icbc::ICBCcomponent...)
```

Initial and boundary conditions for a ModelingToolkit.jl PDESystem. 
It can be used with the `+` operator to add initial and boundary conditions to a
ModelingToolkit.jl ODESystem or Catalyst.jl ReactionSystem.

$(FIELDS)

# Example:
```jldoctest
using EarthSciMLBase
using ModelingToolkit, DomainSets

# Set up ODE system
@parameters x y t
@variables u(t) v(t)
Dt = Differential(t)

x_min = y_min = t_min = 0.0
x_max = y_max = 1.0
t_max = 11.5

eqs = [
    Dt(u) ~ √abs(v),
    Dt(v) ~ √abs(u),
]

@named sys = ODESystem(eqs)

# Create domains.
indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [
    x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max)
]

# Create constant initial and boundary conditions = 16.0.
icbc = ICBC(constIC(16.0, indepdomain, partialdomains), constBC(16.0, indepdomain, partialdomains))

# Convert to PDESystem and add constant initial and boundary conditions.
pdesys = sys + icbc

pdesys.bcs

# output
10-element Vector{Equation}:
 u(x, y, 0.0) ~ 16.0
 v(x, y, 0.0) ~ 16.0
 u(0.0, y, t) ~ 16.0
 u(1.0, y, t) ~ 16.0
 u(x, 0.0, t) ~ 16.0
 u(x, 1.0, t) ~ 16.0
 v(0.0, y, t) ~ 16.0
 v(1.0, y, t) ~ 16.0
 v(x, 0.0, t) ~ 16.0
 v(x, 1.0, t) ~ 16.0
```

"""
struct ICBC
    "The sets of initial and/or boundary conditions."
    icbc::Vector{ICBCcomponent}

    ICBC(icbc::ICBCcomponent...) = new(ICBCcomponent[icbc...])
end

function (icbc::ICBC)(sys::ModelingToolkit.ODESystem)::Vector{Equation}
    o = [icbc(sys) for icbc ∈ icbc.icbc]
    vcat(o...)
end

"""
$(TYPEDSIGNATURES)

Construct constant initial conditions equal to the value 
specified by `val`.

$(FIELDS)

"""
struct constIC <: ICBCcomponent
    "The value of the constant initial conditions."
    val
    "The independent domain, e.g. `t ∈ Interval(t_min, t_max)`."
    indepdomain::Symbolics.VarDomainPairing
    "The partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}
end

function (ic::constIC)(sys::ModelingToolkit.ODESystem)
    dims = [domain.variables for domain in ic.partialdomains]
    statevars = add_dims(states(sys), [dims...; ic.indepdomain.variables])
    
    bcs = Equation[]
    
    for state ∈ statevars
        push!(bcs, state.val.f(dims..., ic.indepdomain.domain.left) ~ ic.val)
    end

    bcs
end

"""
$(TYPEDSIGNATURES)

Construct constant boundary conditions equal to the value 
specified by `val`.

$(FIELDS)

"""
struct constBC <: ICBCcomponent
    "The value of the constant boundary conditions."
    val
    "The independent domain, e.g. `t ∈ Interval(t_min, t_max)`."
    indepdomain::Symbolics.VarDomainPairing
    "The partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}
    "The indexes of the domains that these boundary conditons are applied to, e.g. `1,2`. Leave empty if you want to apply to all partial domains."
    activepartialdomainindex

    constBC(val, indepdomain::Symbolics.VarDomainPairing, partialdomains::Vector{Symbolics.VarDomainPairing}) = new(val, indepdomain, partialdomains, 1:length(partialdomains))
    constBC(val, indepdomain::Symbolics.VarDomainPairing, partialdomains::Vector{Symbolics.VarDomainPairing}, activepartialdomainindex...) = new(val, indepdomain, partialdomains, activepartialdomainindex)
end

function (bc::constBC)(sys::ModelingToolkit.ODESystem)
    dims = [domain.variables for domain in bc.partialdomains]
    statevars = add_dims(states(sys), [dims...; bc.indepdomain.variables])
    
    bcs = Equation[]
    
    for state ∈ statevars
        for (i, domain) ∈ enumerate(bc.partialdomains)
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[dims..., bc.indepdomain.variables]
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
struct zerogradBC <: ICBCcomponent
    "The independent domain, e.g. `t ∈ Interval(t_min, t_max)`."
    indepdomain::Symbolics.VarDomainPairing
    "All of partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}
    "The indexes of the domains that these boundary conditons are applied to, e.g. `1,2`. Leave empty if you want to apply to all partial domains."
    activepartialdomainindex
    zerogradBC(indepdomain::Symbolics.VarDomainPairing, partialdomains::Vector{Symbolics.VarDomainPairing}) = new(indepdomain, partialdomains, 1:length(partialdomains))
    zerogradBC(indepdomain::Symbolics.VarDomainPairing, partialdomains::Vector{Symbolics.VarDomainPairing}, activepartialdomainindex...) = new(indepdomain, partialdomains, activepartialdomainindex)
end

function (bc::zerogradBC)(sys::ModelingToolkit.ODESystem)
    dims = [domain.variables for domain in bc.partialdomains]
    statevars = add_dims(states(sys), [dims...; bc.indepdomain.variables])
    
    bcs = Equation[]

    D = Differential(bc.indepdomain.variables)

    for state ∈ statevars
        for i ∈ bc.activepartialdomainindex
            domain = bc.partialdomains[i]
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[dims..., bc.indepdomain.variables]
                args[i] = edge
                push!(bcs, D(state.val.f(args...)) ~ 0.0)
            end
        end
    end

    bcs
end

"""
$(TYPEDSIGNATURES)

Construct period boundary conditions for the given `partialdomains`.
Periodic boundary conditions are defined as when the value at one
side of the domain is set equal to the value at the other side, so 
that the domain "wraps around" from one side to the other.

$(FIELDS)

"""
struct periodicBC <: ICBCcomponent
    "The independent domain, e.g. `t ∈ Interval(t_min, t_max)`."
    indepdomain::Symbolics.VarDomainPairing
    "All of partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}
    "The indexes of the domains that these boundary conditons are applied to, e.g. `1,2`. Leave empty if you want to apply to all partial domains."
    activepartialdomainindex

    periodicBC(indepdomain::Symbolics.VarDomainPairing, partialdomains::Vector{Symbolics.VarDomainPairing}) = new(indepdomain, partialdomains, 1:length(partialdomains))
    periodicBC(indepdomain::Symbolics.VarDomainPairing, partialdomains::Vector{Symbolics.VarDomainPairing}, activepartialdomainindex...) = new(indepdomain, partialdomains, activepartialdomainindex)
end

function (bc::periodicBC)(sys::ModelingToolkit.ODESystem)
    dims = [domain.variables for domain in bc.partialdomains]
    statevars = add_dims(states(sys), [dims...; bc.indepdomain.variables])
    
    bcs = Equation[]

    for state ∈ statevars
        for i ∈ bc.activepartialdomainindex
            domain = bc.partialdomains[i]
            argsleft = Any[dims..., bc.indepdomain.variables]
            argsleft[i] = domain.domain.left
            argsright = Any[dims..., bc.indepdomain.variables]
            argsright[i] = domain.domain.right
            push!(bcs, state.val.f(argsleft...) ~ state.val.f(argsright...))
        end
    end

    bcs
end

"""
$(METHODLIST)

Returns the dimensions of the independent and partial domains associated with these 
initial or boundary conditions.
"""
dims(icbc::ICBCcomponent) = [[domain.variables for domain in icbc.partialdomains]..., icbc.indepdomain.variables]
dims(icbc::ICBC) = unique(vcat(dims.(icbc.icbc)...))

"""
$(METHODLIST)

Returns the domains associated with these initial or boundary conditions.
"""
domains(icbc::ICBCcomponent) = [icbc.partialdomains..., icbc.indepdomain]
domains(icbc::ICBC) = unique(vcat(domains.(icbc.icbc)...))

function Base.:(+)(sys::ModelingToolkit.ODESystem, icbc::ICBC)::ModelingToolkit.PDESystem
    dimensions = dims(icbc)
    statevars = states(sys)
    defaults = getfield(sys, :defaults)
    ps = [k => v for (k,v) in defaults] # Add parameters and their default values
    if !all([p ∈ keys(defaults) for p in parameters(sys)])
        error("All parameters in the system of equations must have default values.")
    end
    ivs = dims(icbc) # New dimensions are the independent variables.
    dvs = add_dims(statevars, dimensions) # Add new dimensions to dependent variables.
    eqs = Vector{Equation}([add_dims(eq, statevars, dimensions) for eq in equations(sys)]) # Add new dimensions to equations.
    PDESystem(eqs, icbc(sys), domains(icbc), ivs, dvs, ps, name=nameof(sys), defaults=defaults)
end

Base.:(+)(icbc::ICBC, sys::ModelingToolkit.ODESystem)::ModelingToolkit.PDESystem = sys + icbc

function Base.:(+)(sys::Catalyst.ReactionSystem, icbc::ICBC)::ModelingToolkit.PDESystem
    convert(ODESystem, sys; combinatoric_ratelaws=false) + icbc
end

Base.:(+)(icbc::ICBC, sys::Catalyst.ReactionSystem)::ModelingToolkit.PDESystem = sys + icbc
