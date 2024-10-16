export CoupledSystem, ConnectorSystem, couple

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
    * `ModelingToolkit.ODESystem`s. If the ODESystem has a field in the metadata called
        `:coupletype` (e.g. `ModelingToolkit.get_metadata(sys)[:coupletype]` returns a struct type
        with a single field called `sys`)
        then that type will be used to check for methods of `EarthSciMLBase.couple` that use that type.
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
        elseif (sys isa Tuple) || (sys isa AbstractVector)
            o = couple(o, sys...)
        elseif hasmethod(init_callback, (typeof(sys), Simulator))
            push!(o.init_callbacks, sys)
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

"Return the coupling type associated with the given system."
function get_coupletype(sys::ModelingToolkit.AbstractSystem)
    md = ModelingToolkit.get_metadata(sys)
    if (!isa(md, Dict)) || (:coupletype ∉ keys(md))
        return Nothing
    end
    T = md[:coupletype]
    @assert ((length(fieldnames(T)) == 1) && (only(fieldnames(T)) == :sys))
        "The `couple_type` $T must have a single field named `:sys` and no other fields"
    T
end

"""
$(SIGNATURES)

Perform bi-directional coupling for two
equation systems.

To specify couplings for system pairs, create
methods for this function with the signature:

```julia
EarthSciMLBase.couple2(a::ACoupler, b::BCoupler)::ConnectorSystem
```

where `ACoupler` and `BCoupler` are `:coupletype`s defined like this:

```julia
struct ACoupler sys end
@named asys = ODESystem([], t, metadata=Dict(:coupletype=>ACoupler))
```
"""
couple2() = nothing

"""
$(SIGNATURES)

Get the ODE ModelingToolkit ODESystem representation of a [`CoupledSystem`](@ref).
"""
function Base.convert(::Type{<:ODESystem}, sys::CoupledSystem; name=:model, kwargs...)::ModelingToolkit.AbstractSystem
    connector_eqs = []
    systems = copy(sys.systems)
    for (i, a) ∈ enumerate(systems)
        for (j, b) ∈ enumerate(systems)
            a_t, b_t = get_coupletype(a), get_coupletype(b)
            if hasmethod(couple2, (a_t, b_t))
                cs = couple2(a_t(a), b_t(b))
                @assert cs isa ConnectorSystem "The result of coupling two systems together with must be a ConnectorSystem. " *
                                               "This is not the case for $(nameof(a)) ($a_t) and $(nameof(b)) ($b_t); it is instead a $(typeof(cs))."
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
    connectors = ODESystem(connector_eqs, iv; name=name, kwargs...)

    # Compose everything together.
    o = compose(connectors, systems...)
    remove_extra_defaults(o)
end

"""
    $(SIGNATURES)

Get the ModelingToolkit PDESystem representation of a [`CoupledSystem`](@ref).
"""
function Base.convert(::Type{<:PDESystem}, sys::CoupledSystem; name=:model, kwargs...)::ModelingToolkit.AbstractSystem
    o = convert(ODESystem, sys; name, kwargs...)

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
