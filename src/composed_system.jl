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
Base.:(+)(a::EarthSciMLODESystem, b::EarthSciMLODESystem)::ComposedEarthSciMLSystem = ComposedEarthSciMLSystem(a, b)

function Base.:(+)(composed::ComposedEarthSciMLSystem, domaininfo::DomainInfo)::ComposedEarthSciMLSystem
    @assert composed.domaininfo === nothing "Cannot add two sets of DomainInfo to a system."
    ComposedEarthSciMLSystem(composed.systems, domaininfo, composed.pdefunctions)
end
Base.:(+)(di::DomainInfo, composed::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = composed + di

function Base.:(+)(sys::EarthSciMLODESystem, di::DomainInfo)::ComposedEarthSciMLSystem
    ComposedEarthSciMLSystem(EarthSciMLODESystem[sys], di, [])
end
Base.:(+)(di::DomainInfo, sys::EarthSciMLODESystem)::ComposedEarthSciMLSystem = sys + di

function get_mtk(sys::ComposedEarthSciMLSystem; name=:model)::ModelingToolkit.AbstractSystem
    # Separate the connector systems from the concrete systems.
    connector_eqs = []
    for (i, a) ∈ enumerate(sys.systems)
        for (j, b) ∈ enumerate(sys.systems)
            if applicable(couple, a, b)
                cs = couple(a, b)
                @assert cs isa ConnectorSystem "The result of coupling two systems together with must be a ConnectorSystem. "*
                                "This is not the case for $(typeof(a)) and $(typeof(b)); it is instead a $(typeof(cs))."
                sys.systems[i], a = cs.from, cs.from
                sys.systems[j], b = cs.to, cs.to
                append!(connector_eqs, cs.eqs)
            end
        end
    end
    # systems = []
    # connectorsystems = []
    # for s ∈ sys.systems
    #     if isa(s, ConnectorSystem)
    #         push!(connectorsystems, s)
    #     elseif isa(s, EarthSciMLODESystem)
    #         push!(systems, s)
    #     else
    #         error("Cannot compose system of type $(typeof(s))")
    #     end
    # end

    # if length(systems) == 0 && length(connectorsystems) > 0
    #     error("Cannot compose only connector systems")
    # end

    # Create the connector system of equations.
    #connector_eqs = vcat([s.eqs for s ∈ connectorsystems]...)
    #if length(connectors) > 0
        iv = ModelingToolkit.get_iv(get_mtk(first(sys.systems)))
        connectors = ODESystem(connector_eqs, iv; name=name)
    #end

    # Finalize the concrete systems.
    mtksys = [get_mtk(s) for s ∈ sys.systems]
    # Compose everything together.
    o = compose(connectors, mtksys...)

    if sys.domaininfo !== nothing
        o += sys.domaininfo
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