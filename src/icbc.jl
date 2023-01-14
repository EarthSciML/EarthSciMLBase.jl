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

# Create constant initial and boundary conditions = 16.0.
icbc = ICBC(
    constBC(16.0, 
        x ∈ Interval(x_min, x_max),
        y ∈ Interval(y_min, y_max),
    ),
    constIC(4.0, t ∈ Interval(t_min, t_max)),
)

# Convert to PDESystem and add constant initial and boundary conditions.
pdesys = sys + icbc

pdesys.bcs

# output
10-element Vector{Equation}:
 u(0.0, y, t) ~ 16.0
 u(1.0, y, t) ~ 16.0
 u(x, 0.0, t) ~ 16.0
 u(x, 1.0, t) ~ 16.0
 v(0.0, y, t) ~ 16.0
 v(1.0, y, t) ~ 16.0
 v(x, 0.0, t) ~ 16.0
 v(x, 1.0, t) ~ 16.0
 u(x, y, 0.0) ~ 4.0
 v(x, y, 0.0) ~ 4.0
```

"""
struct ICBC
    "The sets of initial and/or boundary conditions."
    icbc::Vector{ICBCcomponent}

    ICBC(icbc::ICBCcomponent...) = new(ICBCcomponent[icbc...])
end

function (icbc::ICBC)(sys::ModelingToolkit.ODESystem)::Vector{Equation}
    ic = icbc.icbc[findall(icbc -> isa(icbc, ICcomponent), icbc.icbc)]
    @assert length(ic) == 1 "Only one independent domain is allowed."

    bcs = icbc.icbc[findall(icbc -> isa(icbc, BCcomponent), icbc.icbc)]
    partialdomains = vcat([bc.partialdomains for bc ∈ bcs]...)
    @assert length(partialdomains) > 0 "At least one partial domain is required."
    @assert length(unique(partialdomains)) == length(partialdomains) "Each partial domain must have only one set of boundary conditions."
    o = [icbc(sys, ic[1].indepdomain, partialdomains) for icbc ∈ icbc.icbc]
    vcat(o...)
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

function (ic::constIC)(sys::ModelingToolkit.ODESystem, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states(sys), [dims...; indepdomain.variables])
    
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

function (bc::constBC)(sys::ModelingToolkit.ODESystem, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states(sys), [dims...; indepdomain.variables])
    
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

function (bc::zerogradBC)(sys::ModelingToolkit.ODESystem, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states(sys), [dims...; indepdomain.variables])
    
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

function (bc::periodicBC)(sys::ModelingToolkit.ODESystem, indepdomain::Symbolics.VarDomainPairing, allpartialdomains::Vector{Symbolics.VarDomainPairing})
    dims = [domain.variables for domain in allpartialdomains]
    statevars = add_dims(states(sys), [dims...; indepdomain.variables])
    
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
$(METHODLIST)

Returns the dimensions of the independent and partial domains associated with these 
initial or boundary conditions.
"""
dims(icbc::ICcomponent) = [icbc.indepdomain.variables]
dims(icbc::BCcomponent) = [domain.variables for domain in icbc.partialdomains]
dims(icbc::ICBC) = unique(vcat(dims.(icbc.icbc)...))

"""
$(METHODLIST)

Returns the domains associated with these initial or boundary conditions.
"""
domains(icbc::ICcomponent) = [icbc.indepdomain]
domains(icbc::BCcomponent) = icbc.partialdomains
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
