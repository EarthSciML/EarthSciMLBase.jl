export CoupledSystem, ConnectorSystem, couple

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
    * Types `X` that implement a `EarthSciMLBase.init_callback(::X, sys::CoupledSystem, sys_mtk, domain::DomainInfo)::DECallback` method
    * Other `CoupledSystem`s
    * Types `X` that implement a `EarthSciMLBase.couple2(::X, ::CoupledSystem)` or `EarthSciMLBase.couple2(::CoupledSystem, ::X)` method.
    * `Tuple`s or `AbstractVector`s of any of the things above.
"""
mutable struct CoupledSystem
    "Model components to be composed together"
    systems::Vector{ModelingToolkit.AbstractSystem}
    "Initial and boundary conditions and other domain information"
    domaininfo::Union{Nothing,DomainInfo}
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
        elseif hasmethod(init_callback, (typeof(sys), CoupledSystem, Any, DomainInfo))
            push!(o.init_callbacks, sys)
        elseif hasmethod(couple2, (CoupledSystem, typeof(sys)))
            o = couple2(o, sys)
        elseif hasmethod(couple2, (typeof(sys), CoupledSystem))
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

kwargs:
- name: The desired name for the resulting ODESystem
- simplify: Whether to run `structural_simplify` on the resulting ODESystem
- prune: Whether to prune the extra observed equations to improve performance

Return values:
- The ODESystem representation of the CoupledSystem
- The extra observed equations which have been pruned to improve performance
"""
function Base.convert(::Type{<:ODESystem}, sys::CoupledSystem; name=:model, simplify=true,
    prune=true, extra_vars=[], kwargs...)
    connector_eqs = Equation[]
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
    if !isnothing(sys.domaininfo) # Add coordinate transform equations.
        o = extend(o, partialderivative_transform_eqs(o, sys.domaininfo))
    end
    o = ModelingToolkit.flatten(o)
    o_simplified = structural_simplify(o)
    if prune
        extra_vars2 = []
        if !isnothing(sys.domaininfo)
            extra_vars2 = operator_vars(sys, o_simplified, sys.domaininfo)
        end
        o = prune_observed(o, o_simplified, vcat(extra_vars, extra_vars2))
    end
    o_simplified = structural_simplify(o)
    o = remove_extra_defaults(o, o_simplified)
    if simplify
        o = structural_simplify(o)
    end
    return o
end

"""
    $(SIGNATURES)

Get the ModelingToolkit PDESystem representation of a [`CoupledSystem`](@ref).
"""
function Base.convert(::Type{<:PDESystem}, sys::CoupledSystem; name=:model, kwargs...)::ModelingToolkit.AbstractSystem
    o = convert(ODESystem, sys; name=name, simplify=false, prune=false, kwargs...)

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

# Combine the non-stiff operators into a single operator.
# This works because SciMLOperators can be added together.
function nonstiff_ops(sys::CoupledSystem, sys_mtk, coord_args, domain, u0, p)
    nonstiff_op = length(sys.ops) > 0 ?
                  sum([get_scimlop(op, sys, sys_mtk, coord_args, domain, u0, p) for op ∈ sys.ops]) :
                  NullOperator(length(u0))
    nonstiff_op = cache_operator(nonstiff_op, u0)
end

function operator_vars(sys::CoupledSystem, mtk_sys, domain::DomainInfo)
    unique(vcat([get_needed_vars(op, sys, mtk_sys, domain) for op in sys.ops]...))
end

"""
Types that implement an:

`init_callback(x, sys::CoupledSystem, sys_mtk, domain::DomainInfo)::DECallback`

method can also be coupled into a `CoupledSystem`.
The `init_callback` function will be run before the simulator is run
to get the callback.
"""
init_callback() = error("Not implemented")

function get_callbacks(sys::CoupledSystem, sys_mtk, domain::DomainInfo)
    extra_cb = [init_callback(c, sys, sys_mtk, domain::DomainInfo) for c ∈ sys.init_callbacks]
    [sys.callbacks; extra_cb]
end

function domain(s::CoupledSystem)
    @assert !isnothing(s.domaininfo) "The system must have a domain specified; see documentation for EarthSciMLBase.DomainInfo."
    s.domaininfo
end
