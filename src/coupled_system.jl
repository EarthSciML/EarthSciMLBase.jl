export CoupledSystem, ConnectorSystem, get_mtk, register_coupling, couple, systemname

""" Return a unique identifier for a system. """
function systemhash(sys::ModelingToolkit.AbstractSystem)
    name = nameof(sys)
    if !occursin("₊", string(name))
        @warn "The name of each system should be formatted like `Module₊Name`. " *
              "The name of the system `$name` does not meet this criteria. " *
              "If the system is part of the `MyModule` module, the proper name would be `MyModule₊$name`."
    end
    name
end

"""
A system for composing together other systems using the [`couple`](@ref) function.

$(FIELDS)

"""
mutable struct CoupledSystem
    "Model components to be composed together"
    systems::Vector{ModelingToolkit.AbstractSystem}
    "Initial and boundary conditions and other domain information"
    domaininfo
    """
    A vector of functions where each function takes as an argument the resulting PDESystem after DomainInfo is 
    added to this system, and returns a transformed PDESystem.
    """
    pdefunctions::AbstractVector

    """
    A vector of operators to run during simulations.
    """
    ops::Vector{Operator}
end

function Base.show(io::IO, cs::CoupledSystem)
    print(io, "CoupledSystem containing $(length(cs.systems)) system(s) and $(length(cs.ops)) operator(s).")
end

"""    
    $(TYPEDSIGNATURES)

Couple multiple ModelingToolkit systems together.

The systems that are arguments to this system can be of type `ModelingToolkit.AbstractSystem`, 
[`CoupledSystem`](@ref), [`DomainInfo`](@ref), 
or any type `T` that has a method `couple(::CoupledSystem, ::T)::CoupledSystem` or a method 
`couple(::T, ::CoupledSystem)::CoupledSystem` defined for it.
"""
function couple(systems...)::CoupledSystem
    o = CoupledSystem([], nothing, [], [])
    for sys ∈ systems
        if sys isa DomainInfo # Add domain information to the system.
            if o.domaininfo !== nothing
                error("Cannot add two sets of DomainInfo to a system.")
            end
            o.domaininfo = sys
        elseif sys isa Operator
            push!(o.ops, sys)
        elseif sys isa ModelingToolkit.AbstractSystem # Add a system to the coupled system.
            push!(o.systems, sys)
        elseif sys isa CoupledSystem # Add a coupled system to the coupled system.
            o.systems = vcat(o.systems, sys.systems)
            o.pdefunctions = vcat(o.pdefunctions, sys.pdefunctions)
            if sys.domaininfo !== nothing
                if o.domaininfo !== nothing
                    error("Cannot add two sets of DomainInfo to a system.")
                end
                o.domaininfo = sys.domaininfo
            end
        elseif applicable(couple, o, sys)
            o = couple(o, sys)
        elseif applicable(couple, sys, o)
            o = couple(sys, o)
        else
            error("Cannot couple a $(typeof(sys)).")
        end
    end
    o
end

"A registery for functions to couple systems together, defined by their [system hashes](@ref systemhash)."
const coupling_registry = Dict{Tuple{Symbol,Symbol},Function}()

"""
    $(SIGNATURES)

Register a coupling function for two systems.

In the `get_mtk` method of `CoupledSystem`, the function `f` is called to
make any edits needed to the two systems before they are composed together,
and also to return a `ConnectorSystem` that represents the coupling of the two systems.

The function `f` should take as two `ODESystem`s and return a `ConnectorSystem`, i.e. 
`f(a::ODESystem, b::ODESystem)::ConnectorSystem`.
"""
function register_coupling(f::Function, a::ModelingToolkit.AbstractSystem, b::ModelingToolkit.AbstractSystem)
    ah, bh = systemhash(a), systemhash(b)
    if (ah, bh) in keys(coupling_registry)
        error("Coupling between $(nameof(a)) and $(nameof(b)) already registered.")
    end
    coupling_registry[(ah, bh)] = f
end

"""
    $(SIGNATURES)

Get the ODE ModelingToolkit representation of a [`CoupledSystem`](@ref).
"""
function get_mtk_ode(sys::CoupledSystem; name=:model)::ModelingToolkit.AbstractSystem
    connector_eqs = []
    systems = copy(sys.systems)
    hashes = systemhash.(systems)
    for (i, a) ∈ enumerate(systems)
        for (j, b) ∈ enumerate(systems)
            if (hashes[i], hashes[j]) ∈ keys(coupling_registry)
                f = coupling_registry[hashes[i], hashes[j]]
                cs = f(deepcopy(a), deepcopy(b))
                @assert cs isa ConnectorSystem "The result of coupling two systems together with must be a ConnectorSystem. " *
                                               "This is not the case for $(nameof(a)) and $(nameof(b)); it is instead a $(typeof(cs))."
                systems[i], a = cs.from, cs.from
                systems[j], b = cs.to, cs.to
                for eq ∈ cs.eqs
                    @assert ModelingToolkit.validate(eq) "invalid units in coupling equation: $eq. See warnings for details."
                end
                append!(connector_eqs, cs.eqs)
            end
        end
    end
    iv = ModelingToolkit.get_iv(first(systems))
    connectors = ODESystem(connector_eqs, iv; name=name)

    # Compose everything together.
    compose(connectors, systems...)
end

"""
    $(SIGNATURES)

Get the ModelingToolkit representation of a [`CoupledSystem`](@ref).
"""
function get_mtk(sys::CoupledSystem; name=:model)::ModelingToolkit.AbstractSystem
    o = get_mtk_ode(sys; name)

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
    from::ModelingToolkit.AbstractSystem
    to::ModelingToolkit.AbstractSystem
end