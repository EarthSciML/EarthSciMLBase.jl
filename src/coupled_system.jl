export CoupledSystem, ConnectorSystem, couple, CoupleType, SysDiscreteEvent

"""
A system for composing together other systems using the [`couple`](@ref) function.

$(FIELDS)

Things that can be added to a `CoupledSystem`:

  - `ModelingToolkit.System`s. If the System has a field in the metadata called
    `:coupletype` (e.g. `ModelingToolkit.get_metadata(sys)[:coupletype]` returns a struct type
    with a single field called `sys`)
    then that type will be used to check for methods of `EarthSciMLBase.couple` that use that type.
  - [`Operator`](@ref)s
  - [`DomainInfo`](@ref)s
  - [Callbacks](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/)
  - Types `X` that implement a `EarthSciMLBase.init_callback(::X, ::CoupledSystem, sys_mtk, ::DomainInfo, ::MapAlgorithm)::DECallback` method
  - Other `CoupledSystem`s
  - Types `X` that implement a `EarthSciMLBase.couple2(::X, ::CoupledSystem)` or `EarthSciMLBase.couple2(::CoupledSystem, ::X)` method.
  - `Tuple`s or `AbstractVector`s of any of the things above.
"""
mutable struct CoupledSystem
    "Model components to be composed together"
    systems::Vector{ModelingToolkit.AbstractSystem}
    "Initial and boundary conditions and other domain information"
    domaininfo::Union{Nothing, DomainInfo}
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
    print(io,
        "CoupledSystem containing $(length(cs.systems)) system(s), $(length(cs.ops)) operator(s), and $(length(cs.callbacks) + length(cs.init_callbacks)) callback(s).")
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
    for sys in systems
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
        elseif hasmethod(
            init_callback, (typeof(sys), CoupledSystem, Any, Any, DomainInfo,
                MapAlgorithm))
            push!(o.init_callbacks, sys)
        elseif hasmethod(couple2, (CoupledSystem, typeof(sys)))
            o = couple2(o, sys)
        elseif hasmethod(couple2, (typeof(sys), CoupledSystem)) # TODO(CT): Mismatch between couple and couple2 here?
            o = couple(sys, o)
        else
            error("Cannot couple a $(typeof(sys)).")
        end
    end
    o
end

"""
The DataType that should be used in the ModelingToolkit System
metadata for specifying a system's coupling behavior.
"""
struct CoupleType end

"""
Return the coupling type associated with the given system.
"""
function get_coupletype(sys::ModelingToolkit.AbstractSystem)
    T = getmetadata(sys, CoupleType, nothing)
    if isnothing(T)
        return Nothing
    end
    @assert ((length(fieldnames(T)) == 1) && (only(fieldnames(T)) == :sys))
    "The `couple_type` $T must have a single field named `:sys` and no other fields"
    T
end

"""
The DataType that should be used in the ModelingToolkit System
metadata for specifying a discrete system event.
"""
struct SysDiscreteEvent end

"""
Returns the `sys_discrete_event` function associated with the given system, which
is meant to be a function that takes the fully coupled ModelingToolkit System and returns
a discrete event that should be applied to it.
"""
function get_sys_discrete_event(sys::ModelingToolkit.System)
    f = getmetadata(sys, SysDiscreteEvent, nothing)
    if isnothing(f)
        return f
    end
    @assert f isa Function "The `sys_discrete_event` for $(nameof(sys)) must be a function."
    @assert hasmethod(f, (AbstractSystem,))
    """The `sys_discrete_event` for $(nameof(sys)) must be a function that takes a
    ModelingToolkit.AbstractSystem as an argument and returns a ModelingToolkit event."""
    f
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
struct ACoupler
    sys
end
@named asys = System([], t, metadata = Dict(:coupletype=>ACoupler))
```
"""
couple2() = nothing

"""
$(SIGNATURES)

Get the ODE ModelingToolkit System representation of a [`CoupledSystem`](@ref).

kwargs:

  - name: The desired name for the resulting System
  - compile: Whether to run `mtkcompile` on the resulting System
  - prune: Whether to prune the extra observed equations to improve performance

Return values:

  - The ModelingToolkit System representation of the CoupledSystem
"""
function Base.convert(::Type{<:System}, sys::CoupledSystem; name = :model, compile = true,
        prune = false, extra_vars = [], kwargs...)
    connector_eqs = Equation[]
    discrete_event_fs = []
    systems = copy(sys.systems)
    for (i, a) in enumerate(systems)
        for (j, b) in enumerate(systems)
            a_t, b_t = get_coupletype(a), get_coupletype(b)
            if hasmethod(couple2, (a_t, b_t))
                cs = couple2(a_t(a), b_t(b))
                @assert cs isa ConnectorSystem "The result of coupling two systems together must be a EarthSciMLBase.ConnectorSystem. " *
                                               "This is not the case for $(nameof(a)) ($a_t) and $(nameof(b)) ($b_t); it is instead a $(typeof(cs))."
                systems[i], a = cs.from, cs.from
                systems[j], b = cs.to, cs.to
                for eq in cs.eqs
                    @assert ModelingToolkit.validate(eq) "invalid units in coupling equation: $eq. See warnings for details."
                end
                append!(connector_eqs, cs.eqs)
            end
        end
        de = get_sys_discrete_event(a)
        (!isnothing(de)) && push!(discrete_event_fs, de)
    end

    iv = ModelingToolkit.get_iv(first(systems))

    # Create temporary coupled system and use it to get system events.
    ics = ModelingToolkit.initial_conditions(ModelingToolkit.flatten(
        System(Equation[], iv; name = :temp, systems = systems)))
    if length(discrete_event_fs) > 0
        temp_connectors = System(connector_eqs, iv; name = name,
            initial_conditions = ics, kwargs...)
        temp_sys = mtkcompile(ModelingToolkit.flatten(compose(
            temp_connectors, systems...)))
        de = filter(!isnothing, [f(temp_sys) for f in discrete_event_fs])

        # Create system of connectors and events.
        connectors = System(connector_eqs, iv; name = name,
            discrete_events = de, initial_conditions = ics, kwargs...)
    else
        # Create system of connectors.
        connectors = System(connector_eqs, iv; name = name,
            initial_conditions = ics, kwargs...)
    end

    # Compose everything together.
    o = compose(connectors, systems...)

    if !isnothing(sys.domaininfo) # Add coordinate transform equations.
        o = extend(o, partialderivative_transform_eqs(o, sys.domaininfo))
    end
    o = ModelingToolkit.flatten(o)
    if prune
        o_simplified = mtkcompile(o)
        extra_vars2 = []
        if !isnothing(sys.domaininfo)
            extra_vars2 = operator_vars(sys, o_simplified, sys.domaininfo)
        end
        o = prune_observed(o, o_simplified, vcat(extra_vars, extra_vars2))
    end
    #o_simplified = mtkcompile(o)
    #o = remove_extra_defaults(o, o_simplified)
    if compile
        o = mtkcompile(o)
    end
    return o
end

"""
    $(SIGNATURES)

Get the ModelingToolkit PDESystem representation of a [`CoupledSystem`](@ref).
"""
function Base.convert(::Type{<:PDESystem}, sys::CoupledSystem; name = :model,
        kwargs...)::ModelingToolkit.AbstractSystem
    o = convert(System, sys; name = name, compile = false, prune = false, kwargs...)

    if sys.domaininfo !== nothing
        o += sys.domaininfo
    end

    if length(sys.pdefunctions) > 0
        @assert sys.domaininfo!==nothing "Cannot apply PDE functions to a system without domain information."
        for f in sys.pdefunctions
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

# Combine the non-stiff operators into a single operator.
# This works because SciMLOperators can be added together.
function nonstiff_ops(sys::CoupledSystem, sys_mtk, coord_args, domain, u0, p, alg)
    fs = [get_odefunction(op, sys, sys_mtk, coord_args, domain, u0, p, alg)
          for op in sys.ops]
    if length(fs) == 0
        return let
            f(u, p, t) = zeros(size(u))
            f(du, u, p, t) = du .= zero(eltype(u))
            f
        end
    elseif length(fs) == 1
        return only(fs)
    else
        fs = tuple(fs...)
        return let
            function f(u, p, t)
                du = zeros(size(u))
                for op in fs
                    du .+= op(u, p, t)
                end
                du
            end
            dusum = zeros(size(u0))
            function f(du, u, p, t)
                dusum .= 0.0
                du .= 0.0
                for op in fs
                    op(du, u, p, t)
                    dusum .+= du
                end
                du
            end
            f
        end
    end
end

function operator_vars(sys::CoupledSystem, mtk_sys, domain::DomainInfo)
    unique(vcat([get_needed_vars(op, sys, mtk_sys, domain) for op in sys.ops]...))
end

"""
Types that implement an:

`init_callback(x, sys::CoupledSystem, sys_mtk, coord_args, domain::DomainInfo, alg::MapAlgorithm)::DECallback`

method can also be coupled into a `CoupledSystem`.
The `init_callback` function will be run before the simulator is run
to get the callback.
"""
init_callback() = error("Not implemented")

function get_callbacks(sys::CoupledSystem, sys_mtk, coord_args, domain::DomainInfo, alg)
    extra_cb = [init_callback(c, sys, sys_mtk, coord_args, domain::DomainInfo, alg)
                for c in sys.init_callbacks]
    [sys.callbacks; extra_cb]
end

function domain(s::CoupledSystem)
    @assert !isnothing(s.domaininfo) "The system must have a domain specified; see documentation for EarthSciMLBase.DomainInfo."
    s.domaininfo
end
