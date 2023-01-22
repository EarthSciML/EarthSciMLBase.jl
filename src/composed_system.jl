export AbstractEarthSciMLSystem, EarthSciMLODESystem, ComposedEarthSciMLSystem, ConnectorSystem, get_mtk 

"""
One or more ModelingToolkit systems of equations. EarthSciML uses custom types to allow 
automatic composition of different systems together.
"""
abstract type AbstractEarthSciMLSystem end

"""
A type for actual implementations of ODE systems.
"""
abstract type EarthSciMLODESystem <: AbstractEarthSciMLSystem end

"""
```julia
    get_mtk(sys::AbstractEarthSciMLSystem)
```
Return the ModelingToolkit version of this system.
"""
get_mtk(sys::AbstractEarthSciMLSystem)::ModelingToolkit.AbstractSystem = sys.sys


"""
$(TYPEDSIGNATURES)

A system for composing together other systems using the `+` operator.

## Example
The easiest way to explain how this works and why we might want to do it is with an example.
The following example is modified from the ModelingToolkit.jl documentation 
[here](https://docs.sciml.ai/ModelingToolkit/dev/basics/Composition/#Inheritance-and-Combine):

```jldoctest
using EarthSciMLBase
using ModelingToolkit

# Set up our independent variable time, which will be shared by all systems.
@parameters t

# Create three systems which we will compose together.
struct SEqn <: EarthSciMLODESystem
    sys::ODESystem

    function SEqn(t) 
        @variables S(t), I(t), R(t)
        D = Differential(t)
        N = S + I + R
        @parameters β
        @named seqn = ODESystem([D(S) ~ -β*S*I/N])
        new(seqn)
    end
end

struct IEqn <: EarthSciMLODESystem
    sys::ODESystem

    function IEqn(t) 
        @variables S(t), I(t), R(t)
        D = Differential(t)
        N = S + I + R
        @parameters β,γ
        @named ieqn = ODESystem([D(I) ~ β*S*I/N-γ*I])
        new(ieqn)
    end
end

struct REqn <: EarthSciMLODESystem
    sys::ODESystem

    function REqn(t) 
        @variables I(t), R(t)
        D = Differential(t)
        @parameters γ
        @named reqn = ODESystem([D(R) ~ γ*I])
        new(reqn)
    end
end


# Create functions to allow us to compose the systems together using the `+` operator.
function Base.:(+)(s::SEqn, i::IEqn)::ComposedEarthSciMLSystem
    seqn = s.sys
    ieqn = i.sys
    ComposedEarthSciMLSystem(
        ConnectorSystem([
            ieqn.S ~ seqn.S,
            seqn.I ~ ieqn.I], s, i), 
        s, i,
    )
end

function Base.:(+)(s::SEqn, r::REqn)::ComposedEarthSciMLSystem
    seqn = s.sys
    reqn = r.sys
    ComposedEarthSciMLSystem(
        ConnectorSystem([seqn.R ~ reqn.R], s, r), 
        s, r,
    )
end

function Base.:(+)(i::IEqn, r::REqn)::ComposedEarthSciMLSystem
    ieqn = i.sys
    reqn = r.sys
    ComposedEarthSciMLSystem(
        ConnectorSystem([
            ieqn.R ~ reqn.R,
            reqn.I ~ ieqn.I], i, r), 
        i, r,
    )
end

# Instantiate our three systems.
seqn, ieqn, reqn = SEqn(t), IEqn(t), REqn(t)

# Compose the systems together using the `+` operator. This is the fancy part!
sir = seqn + ieqn + reqn

# Finalize the system for solving.
sirfinal = get_mtk(sir)

# Show the equations in our combined system.
equations(structural_simplify(sirfinal))

# output
3-element Vector{Equation}:
 Differential(t)(reqn₊R(t)) ~ reqn₊γ*ieqn₊I(t)
 Differential(t)(seqn₊S(t)) ~ (-seqn₊β*ieqn₊I(t)*seqn₊S(t)) / (ieqn₊I(t) + reqn₊R(t) + seqn₊S(t))
 Differential(t)(ieqn₊I(t)) ~ (ieqn₊β*ieqn₊I(t)*seqn₊S(t)) / (ieqn₊I(t) + reqn₊R(t) + seqn₊S(t)) - ieqn₊γ*ieqn₊I(t)
```
"""
struct ComposedEarthSciMLSystem <: AbstractEarthSciMLSystem
    "Model components to be composed together"
    systems::Vector{AbstractEarthSciMLSystem}
    "Initial and boundary conditions"
    icbc
    """
    A vector of functions where each function takes as an argument the resulting PDESystem after ICBCs are 
    added to this sytem, and returns a transformed PDESystem.
    """
    pdefunctions::AbstractVector

    ComposedEarthSciMLSystem(systems::Vector{AbstractEarthSciMLSystem}, icbc, f::AbstractVector) = new(systems, icbc, f)
    ComposedEarthSciMLSystem(systems::AbstractEarthSciMLSystem...) = new([systems...], nothing, [])
end

function Base.:(+)(composed::ComposedEarthSciMLSystem, sys::AbstractEarthSciMLSystem)::ComposedEarthSciMLSystem
    o = AbstractEarthSciMLSystem[sys]
    for s in composed.systems
        push!(o, s)
        if applicable(+, s, sys)
            push!(o, (s + sys).systems...)
        end
    end
    ComposedEarthSciMLSystem(unique(o), composed.icbc, composed.pdefunctions)
end

function Base.:(+)(sys::AbstractEarthSciMLSystem, composed::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem
    composed + sys
end

function Base.:(+)(composed::ComposedEarthSciMLSystem, icbc::ICBC)::ComposedEarthSciMLSystem
    @assert composed.icbc === nothing "Cannot add two sets ICBCs to a system."
    ComposedEarthSciMLSystem(composed.systems, icbc, composed.pdefunctions)
end
Base.:(+)(icbc::ICBC, composed::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = composed + icbc

function Base.:(+)(sys::EarthSciMLODESystem, icbc::ICBC)::ComposedEarthSciMLSystem
    ComposedEarthSciMLSystem(sys, icbc)
end

function get_mtk(sys::ComposedEarthSciMLSystem)::ModelingToolkit.AbstractSystem
    # Separate the connector systems from the concrete systems.
    systems = []
    connectorsystems = []
    for s ∈ sys.systems
        if isa(s, ConnectorSystem)
            push!(connectorsystems, s)
        elseif isa(s, EarthSciMLODESystem)
            push!(systems, s)
        else
            error("Cannot compose system of type $(typeof(s))")
        end
    end

    # Create the connector system of equations.
    @named connectors = ODESystem(vcat([s.eqs for s ∈ connectorsystems]...))

    # Finalize the concrete systems.
    mtksys = []
    for s ∈ systems
        if applicable(get_mtk, s)
            push!(mtksys, get_mtk(s))
        else
            push!(mtksys, s)
        end
    end
    # Compose everything together.
    o = compose(connectors, mtksys...)

    if sys.icbc !== nothing
        o += sys.icbc
        for f ∈ sys.pdefunctions
            o = f(o)
        end    
    end
    return o
end

"""
A connector for two systems.

$(FIELDS)
"""
struct ConnectorSystem <: AbstractEarthSciMLSystem
    eqs::Vector{Equation}
    from::AbstractEarthSciMLSystem
    to::AbstractEarthSciMLSystem
end

get_mtk(_::ConnectorSystem) = error("Cannot convert a connector system to ModelingToolkit")