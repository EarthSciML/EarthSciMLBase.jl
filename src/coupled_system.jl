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
Types that implement an:

`init_callback(x, Simulator)::DECallback`

method can also be coupled into a `CoupledSystem`.
The `init_callback` function will be run before the simulator is run
to get the callback.
"""
init_callback() = error("Not implemented")

"""
A system for composing together other systems using the [`couple`](@ref) function.

$(FIELDS)

Things that can be added to a `CoupledSystem`:
    * `ModelingToolkit.ODESystem`s
    * [`Operator`](@ref)s
    * [`DomainInfo`](@ref)s
    * [Callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/)
    * Types `X` that implement a `EarthSciMLBase.get_callback(::X, ::Simulator)::DECallback` method
    * Other `CoupledSystem`s
    * Types `X` that implement a `EarthSciMLBase.couple(::X, ::CoupledSystem)` or `EarthSciMLBase.couple(::CoupledSystem, ::X)` method.
    * `Tuple`s or `AbstractVector`s of any of the things above.
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

    "A vector of callbacks to run during simulations."
    callbacks::Vector{DECallback}

    "Objects `x` with an `init_callback(x, Simulator)::DECallback` method."
    init_callbacks::Vector
end

function Base.show(io::IO, cs::CoupledSystem)
    print(io, "CoupledSystem containing $(length(cs.systems)) system(s), $(length(cs.ops)) operator(s), and $(length(cs.callbacks) + length(cs.init_callbacks)) callback(s).")
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
    o = CoupledSystem([], nothing, [], [], [], [])
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
            o.ops = vcat(o.ops, sys.ops)
            o.callbacks = vcat(o.callbacks, sys.callbacks)
            o.init_callbacks = vcat(o.init_callbacks, sys.init_callbacks)
            if sys.domaininfo !== nothing
                if o.domaininfo !== nothing
                    error("Cannot add two sets of DomainInfo to a system.")
                end
                o.domaininfo = sys.domaininfo
            end
        elseif sys isa DECallback
            push!(o.callbacks, sys)
        elseif hasmethod(init_callback, (typeof(sys), Simulator))
            push!(o.init_callbacks, sys)
        elseif applicable(couple, o, sys)
            o = couple(o, sys)
        elseif applicable(couple, sys, o)
            o = couple(sys, o)
        elseif (sys isa Tuple) || (sys isa AbstractVector)
            o = couple(o, sys...)
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