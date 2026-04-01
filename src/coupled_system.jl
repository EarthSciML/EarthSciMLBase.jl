export CoupledSystem, ConnectorSystem, couple, CoupleType, SysDiscreteEvent, SysDomainInfo, merge_pdesystems, slice_variable

"""
A system for composing together other systems using the [`couple`](@ref) function.

$(FIELDS)

Things that can be added to a `CoupledSystem`:

  - `ModelingToolkit.System`s. If the System has a field in the metadata called
    `:coupletype` (e.g. `ModelingToolkit.get_metadata(sys)[:coupletype]` returns a struct type
    with a single field called `sys`)
    then that type will be used to check for methods of `EarthSciMLBase.couple` that use that type.
  - `ModelingToolkit.PDESystem`s, which will be stored separately and merged when converting to a PDESystem.
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
    "PDESystem components to be merged together"
    pdesystems::Vector{ModelingToolkit.PDESystem}
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
        "CoupledSystem containing $(length(cs.systems)) system(s), $(length(cs.pdesystems)) PDESystem(s), $(length(cs.ops)) operator(s), and $(length(cs.callbacks) + length(cs.init_callbacks)) callback(s).")
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
    o = CoupledSystem([], [], nothing, [], [], [], [])
    for sys in systems
        if sys isa DomainInfo # Add domain information to the system.
            if o.domaininfo !== nothing
                error("Cannot add two sets of DomainInfo to a system.")
            end
            o.domaininfo = sys
        elseif sys isa Operator
            push!(o.ops, sys)
        elseif sys isa ModelingToolkit.PDESystem # Add a PDESystem to the coupled system.
            push!(o.pdesystems, sys)
        elseif sys isa ModelingToolkit.AbstractSystem # Add a system to the coupled system.
            push!(o.systems, sys)
        elseif sys isa CoupledSystem # Add a coupled system to the coupled system.
            o.systems = vcat(o.systems, sys.systems)
            o.pdesystems = vcat(o.pdesystems, sys.pdesystems)
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
    meta = ModelingToolkit.get_metadata(sys)
    if isnothing(meta)
        return Nothing
    end
    T = get(meta, CoupleType, nothing)
    if isnothing(T)
        return Nothing
    end
    @assert ((length(fieldnames(T)) == 1) && (only(fieldnames(T)) == :sys))
    "The `couple_type` $T must have a single field named `:sys` and no other fields"
    T
end

"""
Return all coupling types associated with the given system as a vector.

Unlike `get_coupletype` (which returns a single type or `Nothing`), this
handles systems whose metadata stores multiple CoupleTypes in a vector,
such as promoted ODE PDESystems that carry the CoupleTypes of their
constituent ODE components.
"""
function get_coupletypes(sys::ModelingToolkit.AbstractSystem)
    meta = ModelingToolkit.get_metadata(sys)
    if isnothing(meta)
        return DataType[]
    end
    val = get(meta, CoupleType, nothing)
    if isnothing(val)
        return DataType[]
    end
    if val isa AbstractVector
        for T in val
            @assert ((length(fieldnames(T)) == 1) && (only(fieldnames(T)) == :sys))
            "The `couple_type` $T must have a single field named `:sys` and no other fields"
        end
        return val
    end
    @assert ((length(fieldnames(val)) == 1) && (only(fieldnames(val)) == :sys))
    "The `couple_type` $val must have a single field named `:sys` and no other fields"
    return DataType[val]
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
The DataType that should be used in the ModelingToolkit System
metadata for specifying a system's own [`DomainInfo`](@ref).

When a system has a `SysDomainInfo` entry in its metadata, that `DomainInfo`
will be used for ODE→PDE promotion instead of the [`CoupledSystem`](@ref)'s
`DomainInfo`. This allows systems with different spatial dimensionality
(e.g., a 3D data source like ERA5 and a 2D surface model) to coexist in the
same coupled system.

See also: [`DomainInfo`](@ref), [`CoupleType`](@ref)
"""
struct SysDomainInfo end

"""
Returns the [`DomainInfo`](@ref) stored in the given system's metadata under
the [`SysDomainInfo`](@ref) key, or `nothing` if none is set.
"""
function get_sys_domaininfo(sys::ModelingToolkit.AbstractSystem)
    meta = ModelingToolkit.get_metadata(sys)
    if isnothing(meta)
        return nothing
    end
    di = get(meta, SysDomainInfo, nothing)
    if !isnothing(di)
        @assert di isa DomainInfo "SysDomainInfo metadata must be a DomainInfo, got $(typeof(di))."
    end
    return di
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
    if !isempty(sys.pdesystems)
        error("Cannot convert a CoupledSystem containing PDESystems to an ODE System. " *
              "Use `convert(PDESystem, ...)` instead.")
    end
    connector_eqs = Equation[]
    discrete_event_fs = []
    systems = copy(sys.systems)
    for (i, a) in enumerate(systems)
        for (j, b) in enumerate(systems)
            i >= j && continue  # Each unordered pair only once, try both directions below.
            for (x, y, xi, yi) in ((a, b, i, j), (b, a, j, i))
                x_t, y_t = get_coupletype(x), get_coupletype(y)
                if hasmethod(couple2, (x_t, y_t))
                    cs = couple2(x_t(x), y_t(y))
                    @assert cs isa ConnectorSystem "The result of coupling two systems together must be a EarthSciMLBase.ConnectorSystem. " *
                                                   "This is not the case for $(nameof(x)) ($x_t) and $(nameof(y)) ($y_t); it is instead a $(typeof(cs))."
                    x_name = nameof(x)
                    if nameof(cs.from) == x_name
                        systems[xi] = cs.from
                        systems[yi] = cs.to
                    elseif nameof(cs.to) == x_name
                        systems[xi] = cs.to
                        systems[yi] = cs.from
                    else
                        error("ConnectorSystem from/to system names ($(nameof(cs.from)), $(nameof(cs.to))) " *
                              "don't match input system names ($x_name, $(nameof(y)))")
                    end
                    for eq in cs.eqs
                        @assert ModelingToolkit.validate(eq) "invalid units in coupling equation: $eq. See warnings for details."
                    end
                    append!(connector_eqs, cs.eqs)
                end
            end
            a = systems[i]  # Re-sync after coupling (from/to may not match i/j order).
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

If the CoupledSystem contains PDESystems, they are merged together into a single
flat PDESystem. Any ODE Systems are first coupled together and promoted to a
PDESystem using the provided DomainInfo, then merged with the existing PDESystems.

ODE Systems may carry their own [`DomainInfo`](@ref) via [`SysDomainInfo`](@ref)
metadata. When present, that system-specific DomainInfo is used for promotion
instead of the CoupledSystem's DomainInfo. This enables coupling of systems with
different spatial dimensionality (e.g., a 2D surface PDE with a 3D data source).
Systems are grouped by DomainInfo, coupled within each group at the ODE level,
then promoted and merged. Cross-group coupling is handled at the PDE level via
[`couple2`](@ref) methods on the promoted PDESystems.
"""
function Base.convert(::Type{<:PDESystem}, sys::CoupledSystem; name = :model,
        kwargs...)::ModelingToolkit.AbstractSystem
    has_pde = !isempty(sys.pdesystems)
    has_ode = !isempty(sys.systems)

    if !has_pde
        # Existing path: ODE Systems only
        o = convert(System, sys; name = name, compile = false, prune = false, kwargs...)
        if sys.domaininfo !== nothing
            o += sys.domaininfo
        end
        if length(sys.pdefunctions) > 0
            @assert sys.domaininfo !== nothing "Cannot apply PDE functions to a system without domain information."
            for f in sys.pdefunctions
                o = f(o)
            end
        end
        return o
    end

    # New path: PDESystem merging
    all_pdesystems = copy(sys.pdesystems)
    coupling_eqs = Equation[]
    handled_cross_pairs = Set{Tuple{DataType, DataType}}()

    # If there are ODE Systems, couple them together and promote to PDESystem.
    # Systems may have their own DomainInfo (via SysDomainInfo metadata) that
    # differs from the CoupledSystem's DomainInfo. In that case, we group systems
    # by DomainInfo, couple within each group, and promote each group separately
    # so that each system gets only the spatial dimensions it needs.
    if has_ode
        groups = _group_by_domaininfo(sys.systems, sys.domaininfo)

        # Precompute system index → group index lookup for O(1) access.
        group_of = Vector{Int}(undef, length(sys.systems))
        for (k, (_, indices)) in enumerate(groups)
            for idx in indices
                group_of[idx] = k
            end
        end

        # Phase 1: Run couple2 between ODE systems in the SAME DomainInfo group.
        # Cross-group coupling is deferred to the PDE-level couple2 loop below.
        systems = copy(sys.systems)
        group_connector_eqs = [Equation[] for _ in groups]

        for (i, a) in enumerate(systems)
            gi = group_of[i]
            for (j, b) in enumerate(systems)
                gi != group_of[j] && continue
                i >= j && continue  # Each unordered pair only once, try both directions below.
                for (x, y, xi, yi) in ((a, b, i, j), (b, a, j, i))
                    x_t, y_t = get_coupletype(x), get_coupletype(y)
                    if hasmethod(couple2, (x_t, y_t))
                        cs = couple2(x_t(x), y_t(y))
                        @assert cs isa ConnectorSystem "The result of coupling two systems together must be a EarthSciMLBase.ConnectorSystem. " *
                                                       "This is not the case for $(nameof(x)) ($x_t) and $(nameof(y)) ($y_t); it is instead a $(typeof(cs))."
                        x_name = nameof(x)
                        if nameof(cs.from) == x_name
                            systems[xi] = cs.from
                            systems[yi] = cs.to
                        elseif nameof(cs.to) == x_name
                            systems[xi] = cs.to
                            systems[yi] = cs.from
                        else
                            error("ConnectorSystem from/to system names ($(nameof(cs.from)), $(nameof(cs.to))) " *
                                  "don't match input system names ($x_name, $(nameof(y)))")
                        end
                        for eq in cs.eqs
                            @assert ModelingToolkit.validate(eq) "invalid units in coupling equation: $eq. See warnings for details."
                        end
                        append!(group_connector_eqs[gi], cs.eqs)
                    end
                end
                a = systems[i]  # Re-sync after coupling (from/to may not match i/j order).
            end
        end

        # Phase 1.5: Run cross-type couple2 between individual ODE systems
        # and PDESystems BEFORE composition/promotion. At this point each
        # ODE system is still individual (sys.varname works), and each
        # PDESystem has its dedicated param_to_var overload.
        # Store coupling equations per group for later transformation.
        cross_coupling_eqs = [Equation[] for _ in groups]

        for (i, ode_sys) in enumerate(systems)
            gi = group_of[i]
            ode_t = get_coupletype(ode_sys)
            ode_t === Nothing && continue
            for (j, pde_sys) in enumerate(all_pdesystems)
                for pde_t in get_coupletypes(pde_sys)
                    # Try both orderings.
                    for (x_t, y_t, x_sys, y_sys) in [
                        (ode_t, pde_t, ode_sys, pde_sys),
                        (pde_t, ode_t, pde_sys, ode_sys),
                    ]
                        hasmethod(couple2, (x_t, y_t)) || continue
                        # Try running couple2 with the individual ODE
                        # System. If the method expects a promoted
                        # PDESystem (e.g. accesses .dvs), it will error
                        # here and we defer to PDE-phase dispatch.
                        local cs
                        try
                            cs = couple2(x_t(x_sys), y_t(y_sys))
                        catch
                            continue
                        end
                        @assert cs isa ConnectorSystem "The result of coupling two systems together must be a EarthSciMLBase.ConnectorSystem. " *
                                                       "This is not the case for ($x_t) and ($y_t); it is instead a $(typeof(cs))."
                        # Update systems: determine which is ODE and which is PDE
                        # based on type, since couple2 controls from/to assignment.
                        ode_name, pde_name = nameof(ode_sys), nameof(pde_sys)
                        for sys_out in (cs.from, cs.to)
                            out_name = nameof(sys_out)
                            if sys_out isa ModelingToolkit.PDESystem
                                @assert out_name == pde_name "ConnectorSystem returned PDESystem named $out_name but expected $pde_name"
                                all_pdesystems[j] = sys_out
                            else
                                @assert out_name == ode_name "ConnectorSystem returned ODE System named $out_name but expected $ode_name"
                                systems[i], ode_sys = sys_out, sys_out
                            end
                        end
                        for eq in cs.eqs
                            @assert ModelingToolkit.validate(eq) "invalid units in coupling equation: $eq. See warnings for details."
                        end
                        append!(cross_coupling_eqs[gi], cs.eqs)
                        push!(handled_cross_pairs, (ode_t, pde_t))
                        push!(handled_cross_pairs, (pde_t, ode_t))
                    end
                end
            end
        end

        # Phase 2: For each group, compose into a flat System and promote
        # to PDESystem with the group's DomainInfo.
        for (k, (di, indices)) in enumerate(groups)
            @assert di !== nothing "DomainInfo is required when coupling ODE Systems with PDESystems."
            group_systems = systems[indices]
            connector_eqs_k = group_connector_eqs[k]

            # Build a pre-composition variable registry so we can later
            # map bare ODE variables in cross-type coupling equations to
            # their promoted (namespaced + spatially-expanded) counterparts.
            pre_comp_vars = Dict{Symbol, Tuple{Symbol, Any}}()
            if !isempty(cross_coupling_eqs[k])
                for s in group_systems
                    sname = nameof(s)
                    for var in unknowns(s)
                        vname = Symbolics.tosymbol(var, escape = false)
                        pre_comp_vars[vname] = (sname, var)
                    end
                end
            end

            # Collect metadata from all systems in this group.
            # Use special handling for CoupleType to avoid overwriting:
            # collect all CoupleTypes into a vector so that the promoted
            # PDE carries the coupling identity of every constituent ODE.
            group_meta = Dict{Any, Any}()
            all_couple_types = DataType[]
            for s in group_systems
                m = ModelingToolkit.get_metadata(s)
                isnothing(m) && continue
                for (k, v) in m
                    if k == CoupleType
                        if v isa AbstractVector
                            append!(all_couple_types, v)
                        else
                            push!(all_couple_types, v)
                        end
                    else
                        group_meta[k] = v
                    end
                end
            end
            if !isempty(all_couple_types)
                group_meta[CoupleType] = all_couple_types
            end
            # Remove SysDomainInfo from the merged metadata since it is
            # consumed during promotion and should not leak into the PDE.
            delete!(group_meta, SysDomainInfo)
            meta_kwarg = isempty(group_meta) ? nothing : group_meta

            iv = ModelingToolkit.get_iv(first(group_systems))
            ics = ModelingToolkit.initial_conditions(ModelingToolkit.flatten(
                System(Equation[], iv; name = :temp, systems = group_systems)))
            connectors = System(connector_eqs_k, iv;
                name = Symbol("ode_group_", k),
                initial_conditions = ics,
                metadata = meta_kwarg, kwargs...)
            o = compose(connectors, group_systems...)
            if !isempty(di.partial_derivative_funcs)
                o = extend(o, partialderivative_transform_eqs(o, di))
            end
            o = ModelingToolkit.flatten(o)
            promoted = o + di

            # Transform cross-type coupling equations: replace bare ODE
            # variables (e.g. R(t)) with their promoted counterparts
            # (e.g. ode_group_1₊rothermel₊R(t,x,y)).
            if !isempty(cross_coupling_eqs[k])
                promoted_dvs = getfield(promoted, :dvs)
                @variables 🔥_cross_couple_temp
                for (eq_idx, eq) in enumerate(cross_coupling_eqs[k])
                    new_lhs = eq.lhs
                    new_rhs = eq.rhs
                    for var in Symbolics.get_variables(eq)
                        varname = Symbolics.tosymbol(var, escape = false)
                        # Try exact match first, then suffix match for namespaced variables
                        # (e.g. rothermel₊R_ct from sys.R_ct dot-notation access).
                        matched_key = if haskey(pre_comp_vars, varname)
                            varname
                        else
                            found = nothing
                            varname_str = string(varname)
                            for (k2, _) in pre_comp_vars
                                if endswith(varname_str, string("₊", k2))
                                    found = k2
                                    break
                                end
                            end
                            found
                        end
                        matched_key === nothing && continue
                        sysname, _ = pre_comp_vars[matched_key]
                        # Match the promoted DV whose name ends with
                        # "sysname₊varname".  Use the form WITHOUT a
                        # leading ₊ so that single-system groups (where
                        # the DV name IS "sysname₊varname" with no outer
                        # prefix) are also matched.
                        target = string(sysname, "₊", matched_key)
                        pidx = findfirst(promoted_dvs) do dv
                            dvname = string(Symbolics.tosymbol(dv, escape = false))
                            endswith(dvname, target)
                        end
                        if pidx === nothing
                            @warn "Cross-coupling equation transformation: could not find promoted counterpart for variable $(varname) (system $(sysname)) in group $k"
                            continue
                        end
                        promoted_var = promoted_dvs[pidx]
                        # Use the two-step substitute_in_deriv pattern
                        # (same as add_dims) to handle derivatives.
                        new_lhs = Symbolics.substitute_in_deriv(new_lhs, Dict(var => 🔥_cross_couple_temp))
                        new_lhs = Symbolics.substitute_in_deriv(new_lhs, Dict(🔥_cross_couple_temp => promoted_var))
                        new_rhs = Symbolics.substitute_in_deriv(new_rhs, Dict(var => 🔥_cross_couple_temp))
                        new_rhs = Symbolics.substitute_in_deriv(new_rhs, Dict(🔥_cross_couple_temp => promoted_var))
                    end
                    cross_coupling_eqs[k][eq_idx] = new_lhs ~ new_rhs
                end
                append!(coupling_eqs, cross_coupling_eqs[k])
            end

            push!(all_pdesystems, promoted)
        end
    end

    # Run couple2 between all PDESystem pairs (PDE-PDE coupling).
    # Skip cross-type pairs already handled in Phase 1.5.
    for (i, a) in enumerate(all_pdesystems)
        for (j, b) in enumerate(all_pdesystems)
            i == j && continue
            for a_t in get_coupletypes(a)
                for b_t in get_coupletypes(b)
                    (a_t, b_t) in handled_cross_pairs && continue
                    if hasmethod(couple2, (a_t, b_t))
                        cs = couple2(a_t(a), b_t(b))
                        @assert cs isa ConnectorSystem "The result of coupling two PDESystems together must be a EarthSciMLBase.ConnectorSystem. " *
                                                       "This is not the case for $(nameof(a)) ($a_t) and $(nameof(b)) ($b_t); it is instead a $(typeof(cs))."
                        a_name = nameof(a)
                        if nameof(cs.from) == a_name
                            all_pdesystems[i], a = cs.from, cs.from
                            all_pdesystems[j], b = cs.to, cs.to
                        elseif nameof(cs.to) == a_name
                            all_pdesystems[i], a = cs.to, cs.to
                            all_pdesystems[j], b = cs.from, cs.from
                        else
                            error("ConnectorSystem from/to system names ($(nameof(cs.from)), $(nameof(cs.to))) " *
                                  "don't match input system names ($a_name, $(nameof(b)))")
                        end
                        append!(coupling_eqs, cs.eqs)
                    end
                end
            end
        end
    end

    # Merge all PDESystems
    merged = merge_pdesystems(all_pdesystems, coupling_eqs; name = name)

    # Apply pdefunctions
    for f in sys.pdefunctions
        merged = f(merged)
    end

    return merged
end

"""
Group ODE systems by their effective [`DomainInfo`](@ref).

Systems with [`SysDomainInfo`](@ref) metadata use their own `DomainInfo`;
others use `default_di`. Returns a `Vector{Tuple{DomainInfo, Vector{Int}}}`
where each tuple is `(domaininfo, system_indices)`.
"""
function _group_by_domaininfo(
        systems::AbstractVector, default_di::Union{Nothing, DomainInfo})
    groups = Tuple{DomainInfo, Vector{Int}}[]
    id_to_idx = Dict{UInt, Int}() # objectid(di) => index into groups

    for (i, sys) in enumerate(systems)
        di = get_sys_domaininfo(sys)
        if isnothing(di)
            di = default_di
        end
        @assert di !== nothing "DomainInfo is required for system $(nameof(sys))."
        key = objectid(di)
        if haskey(id_to_idx, key)
            push!(groups[id_to_idx[key]][2], i)
        else
            push!(groups, (di, [i]))
            id_to_idx[key] = length(groups)
        end
    end
    return groups
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
