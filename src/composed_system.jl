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
    systems::Vector{AbstractEarthSciMLSystem}
    "Initial and boundary conditions and other domain information"
    domaininfo
    """
    A vector of functions where each function takes as an argument the resulting PDESystem after DomainInfo is 
    added to this sytem, and returns a transformed PDESystem.
    """
    pdefunctions::AbstractVector

    ComposedEarthSciMLSystem(systems::Vector{AbstractEarthSciMLSystem}, domaininfo, f::AbstractVector) = new(systems, domaininfo, f)
    ComposedEarthSciMLSystem(systems::AbstractEarthSciMLSystem...) = new([systems...], nothing, [])
end

function Base.:(+)(composed::ComposedEarthSciMLSystem, sys::AbstractEarthSciMLSystem)::ComposedEarthSciMLSystem
    o = AbstractEarthSciMLSystem[sys]
    for s in composed.systems
        push!(o, s)
        if applicable(+, s, sys)
            c = s + sys
            @assert c isa ComposedEarthSciMLSystem "The result of adding two systems together with `+` must be a ComposedEarthSciMLSystem. "*
                            "This is not the case for $(typeof(s)) and $(typeof(sys)); it is instead a $(typeof(c))."
            push!(o, c.systems...)
            if c.domaininfo !== nothing
                @assert composed.domaininfo === nothing "Cannot add two sets of DomainInfo to a system."
                composed.domaininfo = c.domaininfo
            end
            push!(composed.pdefunctions, c.pdefunctions...)
        end
    end
    ComposedEarthSciMLSystem(unique(o), composed.domaininfo, composed.pdefunctions)
end

Base.:(+)(sys::AbstractEarthSciMLSystem, composed::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = composed + sys


function Base.:(+)(composed::ComposedEarthSciMLSystem, domaininfo::DomainInfo)::ComposedEarthSciMLSystem
    @assert composed.domaininfo === nothing "Cannot add two sets of DomainInfo to a system."
    ComposedEarthSciMLSystem(composed.systems, domaininfo, composed.pdefunctions)
end
Base.:(+)(di::DomainInfo, composed::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = composed + di

function Base.:(+)(sys::EarthSciMLODESystem, di::DomainInfo)::ComposedEarthSciMLSystem
    ComposedEarthSciMLSystem(AbstractEarthSciMLSystem[sys], di, [])
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

    if length(systems) == 0 && length(connectorsystems) > 0
        error("Cannot compose only connector systems")
    end

    # Create the connector system of equations.
    connector_eqs = vcat([s.eqs for s ∈ connectorsystems]...)
    if length(connector_eqs) > 0
        iv = ModelingToolkit.get_iv(get_mtk(systems[1]))
        @named connectors = ODESystem(connector_eqs, iv)
    end

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
    o = nothing
    if length(connector_eqs) > 0
        o = compose(connectors, mtksys...)
    else
        o = compose(mtksys...)
    end

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