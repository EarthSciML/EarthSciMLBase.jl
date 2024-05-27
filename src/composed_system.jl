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
$(TYPEDSIGNATURES)
```
Return the ModelingToolkit version of this system.
"""
get_mtk(sys::AbstractEarthSciMLSystem)::ModelingToolkit.AbstractSystem = sys.sys


"""
A system for composing together other systems using the `+` operator.

$(FIELDS)

"""
struct ComposedEarthSciMLSystem <: AbstractEarthSciMLSystem
    "Model components to be composed together"
    systems::Vector{EarthSciMLODESystem}
    "Initial and boundary conditions and other domain information"
    domaininfo
    """
    A vector of functions where each function takes as an argument the resulting PDESystem after DomainInfo is 
    added to this system, and returns a transformed PDESystem.
    """
    pdefunctions::AbstractVector

    ComposedEarthSciMLSystem(systems::Vector{EarthSciMLODESystem}, domaininfo, f::AbstractVector) = new(systems, domaininfo, f)
    ComposedEarthSciMLSystem(systems::EarthSciMLODESystem...) = new([systems...], nothing, [])
end

function Base.:(+)(composed::ComposedEarthSciMLSystem, sys::EarthSciMLODESystem)::ComposedEarthSciMLSystem
    push!(composed.systems, sys)
    composed
end

Base.:(+)(sys::EarthSciMLODESystem, composed::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = composed + sys
Base.:(+)(systems::EarthSciMLODESystem...)::ComposedEarthSciMLSystem = ComposedEarthSciMLSystem(systems...)

function Base.:(+)(composed::ComposedEarthSciMLSystem, domaininfo::DomainInfo)::ComposedEarthSciMLSystem
    @assert composed.domaininfo === nothing "Cannot add two sets of DomainInfo to a system."
    ComposedEarthSciMLSystem(composed.systems, domaininfo, composed.pdefunctions)
end
Base.:(+)(di::DomainInfo, composed::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = composed + di

function Base.:(+)(sys::EarthSciMLODESystem, di::DomainInfo)::ComposedEarthSciMLSystem
    ComposedEarthSciMLSystem(EarthSciMLODESystem[sys], di, [])
end
Base.:(+)(di::DomainInfo, sys::EarthSciMLODESystem)::ComposedEarthSciMLSystem = sys + di

"""
Couple two systems together. This function should be overloaded for each pair of 
systems that can be coupled together.

In the `get_mtk` method of `ComposedEarthSciMLSystem`, this function is called to
make any edits needed to the two systems before they are composed together,
and also to return a `ConnectorSystem` that represents the coupling of the two systems.
"""
couple() = error("not implemented")

function get_mtk(sys::ComposedEarthSciMLSystem; name=:model)::ModelingToolkit.AbstractSystem
    connector_eqs = []
    for (i, a) ∈ enumerate(sys.systems)
        for (j, b) ∈ enumerate(sys.systems)
            if applicable(couple, a, b)
                cs = couple(a, b)
                @assert cs isa ConnectorSystem "The result of coupling two systems together with must be a ConnectorSystem. " *
                                               "This is not the case for $(typeof(a)) and $(typeof(b)); it is instead a $(typeof(cs))."
                sys.systems[i], a = cs.from, cs.from
                sys.systems[j], b = cs.to, cs.to
                append!(connector_eqs, cs.eqs)
            end
        end
    end
    iv = ModelingToolkit.get_iv(get_mtk(first(sys.systems)))
    connectors = ODESystem(connector_eqs, iv; name=name)

    # Finalize the concrete systems.
    mtksys = [get_mtk(s) for s ∈ sys.systems]
    # Compose everything together.
    o = compose(connectors, mtksys...)

    if sys.domaininfo !== nothing
        o += sys.domaininfo
    end

    if length(sys.pdefunctions) > 0
        @assert sys.domaininfo !== nothing "Cannot apply PDE functions to a system without domain information."
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
struct ConnectorSystem
    eqs::Vector{Equation}
    from::AbstractEarthSciMLSystem
    to::AbstractEarthSciMLSystem
end