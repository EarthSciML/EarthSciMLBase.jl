export ICBC, constICBC, dims, domains

"""
Initial and boundary conditions for a ModelingToolkit.jl PDESystem. 
It can be used with the `+` operator to add initial and boundary conditions to a
ModelingToolkit.jl ODESystem or Catalyst.jl ReactionSystem. Refer
to the documentation for the concrete implementations for examples.
"""
abstract type ICBC end

"""
```julia
constICBC(val, indepdomain::Symbolics.VarDomainPairing, partialdomains::Vector{Symbolics.VarDomainPairing})
```

Construct an object that can convert a ModelingToolkit.jl ODESystem or Catalyst.jl ReactionSystem
to a PDESystem and add constant initial and boundary conditions equal to the value 
specified by `val`. The object can be used using the `+` operator.

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
icbc = constICBC(16.0, indepdomain, partialdomains)

# Convert to PDESystem and add constant initial and boundary conditions.
sys + icbc

# output
PDESystem
Equations: Symbolics.Equation[Differential(t)(u(x, y, t)) ~ sqrt(abs(v(x, y, t))), Differential(t)(v(x, y, t)) ~ sqrt(abs(u(x, y, t)))]
Boundary Conditions: Symbolics.Equation[u(x, y, 0.0) ~ 16.0, u(0.0, y, t) ~ 16.0, u(1.0, y, t) ~ 16.0, u(x, 0.0, t) ~ 16.0, u(x, 1.0, t) ~ 16.0, v(x, y, 0.0) ~ 16.0, v(0.0, y, t) ~ 16.0, v(1.0, y, t) ~ 16.0, v(x, 0.0, t) ~ 16.0, v(x, 1.0, t) ~ 16.0]
Domain: Symbolics.VarDomainPairing[Symbolics.VarDomainPairing(x, 0.0..1.0), Symbolics.VarDomainPairing(y, 0.0..1.0), Symbolics.VarDomainPairing(t, 0.0..11.5)]
Dependent Variables: Symbolics.Num[u(x, y, t), v(x, y, t)]
Independent Variables: SymbolicUtils.Sym{Real, Base.ImmutableDict{DataType, Any}}[x, y, t]
Parameters: Any[]
Default Parameter ValuesDict{Any, Any}()
```
"""
struct constICBC <: ICBC
    "The value of the constant initial and boundary conditions."
    val
    "The independent domain, e.g. `t ∈ Interval(t_min, t_max)`."
    indepdomain::Symbolics.VarDomainPairing
    "The partial domains, e.g. `[x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max)]`."
    partialdomains::Vector{Symbolics.VarDomainPairing}
end

function (icbc::constICBC)(sys::ModelingToolkit.ODESystem)
    dims = [domain.variables for domain in icbc.partialdomains]
    statevars = add_dims(states(sys), [dims...; icbc.indepdomain.variables])
    
    bcs = Equation[]
    
    for state ∈ statevars
        push!(bcs, state.val.f(dims..., icbc.indepdomain.domain.left) ~ icbc.val)
        for (i, domain) ∈ enumerate(icbc.partialdomains)
            for edge ∈ [domain.domain.left, domain.domain.right]
                args = Any[dims..., icbc.indepdomain.variables]
                args[i] = edge
                push!(bcs, state.val.f(args...) ~ icbc.val)
            end
        end
    end

    bcs
end

"""
Returns the dimensions of the independent and partial domains associated with these 
initial and boundary conditions.
"""
dims(icbc::constICBC) = [[domain.variables for domain in icbc.partialdomains]..., icbc.indepdomain.variables]

"""
Returns the domains associated with these initial and boundary conditions.
"""
domains(icbc::constICBC) = [icbc.partialdomains..., icbc.indepdomain]

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
    PDESystem(eqs, icbc(sys), domains(icbc), ivs, dvs, ps, name=getfield(sys, :name), defaults=defaults)
end

Base.:(+)(icbc::ICBC, sys::ModelingToolkit.ODESystem)::ModelingToolkit.PDESystem = sys + icbc

function Base.:(+)(sys::Catalyst.ReactionSystem, icbc::ICBC)::ModelingToolkit.PDESystem
    convert(ODESystem, sys; combinatoric_ratelaws=false) + icbc
end

Base.:(+)(icbc::ICBC, sys::Catalyst.ReactionSystem)::ModelingToolkit.PDESystem = sys + icbc
