"""
$(SIGNATURES)

Return the time step length common to all of the given `timesteps`.
Throw an error if not all timesteps are the same length.
"""
function steplength(timesteps)
    Δs = [timesteps[i] - timesteps[i - 1] for i in 2:length(timesteps)]
    @assert all(Δs[1] .≈ Δs) "Not all time steps are the same."
    return Δs[1]
end

"""
$(SIGNATURES)

Return an expression for the observed value of a variable `x` after
substituting in the constants observed values of other variables.
`extra_eqs` is a list of additional equations to use in the substitution.
"""
function observed_expression(eqs, x)
    Base.depwarn(
        "EarthSciMLBase.observed_expression is deprecated",
        :observed_expression)
    expr = nothing
    for eq in eqs
        if isequal(eq.lhs, x)
            expr = eq.rhs
        end
    end
    if isnothing(expr)
        return nothing
    end
    for v in Symbolics.get_variables(expr)
        v_expr = observed_expression(eqs, v)
        if !isnothing(v_expr)
            expr = Symbolics.substitute(expr, v => v_expr)
        end
    end
    # Do it again to catch extra variables TODO(CT): Theoretically this could recurse forever; when to stop?
    for v in Symbolics.get_variables(expr)
        v_expr = observed_expression(eqs, v)
        if !isnothing(v_expr)
            expr = Symbolics.replace(expr, v => v_expr)
        end
    end
    expr
end

"""
$(SIGNATURES)

Return a function to  for the observed value of a variable `x` based
on the input arguments in `coords`.
`extra_eqs` is a list of additional equations to use to determine
the value of `x`.
"""
function observed_function(eqs, x, coords)
    Base.depwarn(
        "EarthSciMLBase.observed_function is deprecated",
        :observed_function)
    expr = observed_expression(eqs, x)
    vars = Symbolics.get_variables(expr)
    if length(vars) > length(coords)
        @warn "Extra variables: $(vars) > $(coords) in observed function for $x."
        return (x...) -> 0.0
    end
    coordvars = []
    for c in coords
        i = findfirst(v -> split(String(Symbol(v)), "₊")[end] == String(Symbol(c)), vars)
        if isnothing(i)
            push!(coordvars, c)
        else
            push!(coordvars, vars[i])
        end
    end
    return Symbolics.build_function(expr, coordvars...; expression = Val{false})
end

"""
$(SIGNATURES)

Return the time points during which integration should be stopped to run the operators.
"""
function timesteps(tsteps::AbstractVector{T}...)::Vector{T} where {T <: AbstractFloat}
    allt = sort(union(vcat(tsteps...)))
    allt2 = [allt[1]]
    for i in 2:length(allt) # Remove nearly duplicate times.
        if allt[i] ≉ allt[i - 1]
            push!(allt2, allt[i])
        end
    end
    allt2
end

"""
$(SIGNATURES)

Return the system variables that the state variables of the final
simplified system depend on. This should be done before running `mtkcompile`
on the system.
`extra_vars` is a list of additional variables that need to be kept.
"""
function get_needed_vars(original_sys::System, simplified_sys::System,
        extra_vars = [])
    Base.depwarn(
        "EarthSciMLBase.get_needed_vars is deprecated",
        :get_needed_vars)

    # Move observed equations back into the simplified system.
    # This allows us to account for any changes to the equations that have been made
    # by `mtkcompile`.
    simplified_sys_obs_reintegrated = copy_with_change(simplified_sys;
        eqs = vcat(equations(simplified_sys), observed(simplified_sys)),
        unknowns = unknowns(original_sys)
    )

    varvardeps = ModelingToolkit.varvar_dependencies(
        ModelingToolkit.asgraph(simplified_sys_obs_reintegrated),
        ModelingToolkit.variable_dependencies(simplified_sys_obs_reintegrated)
    )
    g = SimpleDiGraph(length(unknowns(simplified_sys_obs_reintegrated)))
    for (i, es) in enumerate(varvardeps.badjlist)
        for e in es
            add_edge!(g, i, e)
        end
    end
    allst = unknowns(original_sys) # Get the state variables of the original system.
    # Get the state variables of the simplified system.
    simpst = unique(vcat(unknowns(simplified_sys), extra_vars))
    # Get the state variables in `simplified_sys_obs_reintegrated`.
    simpobsst = unknowns(simplified_sys_obs_reintegrated)
    # Get the index of the simplified state variables in `simplified_sys_obs_reintegrated`.
    stidx = [only(findall(isequal(s), simpobsst)) for s in simpst]
    # Get the index of the variables we need to keep from `simplified_sys_obs_reintegrated`.
    idx = collect(Graphs.DFSIterator(g, stidx))
    # Get the index of the state variables in the original system corresponding to the
    # variables we need from the simplified system.
    stidx = [only(findall(isequal(s), allst)) for s in simpobsst[idx]]
    allst[stidx] # Return the list of variables we need from the original system.
end

"""
$(SIGNATURES)

Create a copy of an System with the given changes.
"""
function copy_with_change(sys::System;
        eqs = equations(sys),
        name = nameof(sys),
        unknowns = nothing,
        parameters = nothing,
        metadata = ModelingToolkit.get_metadata(sys),
        continuous_events = ModelingToolkit.get_continuous_events(sys),
        discrete_events = ModelingToolkit.get_discrete_events(sys),
        defaults = getfield(sys, :initial_conditions)
)
    try
        if isnothing(unknowns) && isnothing(parameters)
            return System(eqs, ModelingToolkit.get_iv(sys);
                name = name, metadata = metadata,
                continuous_events = continuous_events, discrete_events = discrete_events,
                initial_conditions = defaults
            )
        else
            return System(eqs, ModelingToolkit.get_iv(sys), unknowns, parameters;
                name = name, metadata = metadata,
                continuous_events = continuous_events, discrete_events = discrete_events,
                initial_conditions = defaults
            )
        end
    catch e
        if isa(e, ModelingToolkit.ValidationError)
            @warn "Equations:\n$(join(["  $i. $eq" for (i, eq) in enumerate(eqs)], "\n"))"
        end
        rethrow(e)
    end
end

# Get variables effected by this event.
function get_affected_vars(event)
    Base.depwarn(
        "EarthSciMLBase.get_affected_vars is deprecated",
        :get_affected_vars)
    vars = []
    if event.affects isa AbstractVector
        for aff in event.affects
            push!(vars, aff.lhs)
        end
    else
        push!(vars, event.affects.pars...)
        push!(vars, event.affects.sts...)
        push!(vars, event.affects.discretes...)
    end
    unique(vars)
end

function var2symbol(var)
    if var isa Symbolics.CallAndWrap
        var = var.f
    elseif SymbolicUtils.iscall(var)
        var = operation(var)
    end
    Symbolics.tosymbol(var; escape = false)
end

function var_in_eqs(var, eqs)
    any([any(isequal.((var2symbol(var),), var2symbol.(get_variables(eq)))) for eq in eqs])
end

# Return the discrete events that affect variables that are
# needed to specify the state variables of the given system.
# This function should be run after running `mtkcompile`.
function filter_discrete_events(simplified_sys, obs_eqs)
    Base.depwarn(
        "EarthSciMLBase.filter_discrete_events is deprecated",
        :filter_discrete_events)
    de = ModelingToolkit.get_discrete_events(simplified_sys)
    needed_eqs = vcat(equations(simplified_sys), obs_eqs)
    keep = []
    for e in de
        evars = EarthSciMLBase.get_affected_vars(e)
        if any(var_in_eqs.(evars, (needed_eqs,)))
            push!(keep, e)
        end
    end
    keep
end

"""
$(SIGNATURES)

Remove equations from an System where the variable in the LHS is not
present in any of the equations for the state variables. This can be used to
remove computationally intensive equations that are not used in the final model.
"""
function prune_observed(original_sys::System, simplified_sys, extra_vars)
    Base.depwarn(
        "EarthSciMLBase.prune_observed is deprecated",
        :prune_observed)
    needed_vars = var2symbol.(get_needed_vars(original_sys, simplified_sys, extra_vars))
    deleteindex = []
    obs = observed(simplified_sys)
    for (i, eq) in enumerate(obs)
        vars = var2symbol.(get_unknowns([eq]))
        # Only keep equations where all variables are in at least one
        # equation describing the system state.
        if !all((var) -> var ∈ needed_vars, vars)
            push!(deleteindex, i)
        end
    end
    deleteat!(obs, deleteindex)
    discrete_events = filter_discrete_events(simplified_sys, obs)
    new_eqs = [equations(simplified_sys); obs]
    sys2 = copy_with_change(original_sys;
        eqs = new_eqs,
        unknowns = get_unknowns(new_eqs),
        discrete_events = discrete_events
    )
    return sys2
end

# Get the unknown variables in the system of equations.
function get_unknowns(eqs)
    all_vars = unique(vcat(get_variables.(eqs)...))
    unk = []
    for v in all_vars
        if !isnothing(v.metadata) && v.metadata[Symbolics.VariableSource][1] == :variables
            push!(unk, v)
        end
    end
    unk
end

function get_parameters(eqs)
    all_vars = unique(vcat(get_variables.(eqs)...))
    unk = []
    for v in all_vars
        if !isnothing(v.metadata) && v.metadata[Symbolics.VariableSource][1] == :parameters
            push!(unk, v)
        end
    end
    unk
end

# Remove extra variable defaults that would cause a solver initialization error.
function remove_extra_defaults(original_sys, simplified_sys)
    Base.depwarn(
        "EarthSciMLBase.remove_extra_defaults is deprecated",
        :remove_extra_defaults)
    all_vars = unknowns(original_sys)

    unk = var2symbol.(unknowns(simplified_sys))

    # Check if v is not in the unknowns and has a default.
    function checkextra(v)
        !(var2symbol(v) in unk) &&
            (Symbolics.VariableDefaultValue in keys(v.metadata))
    end
    extra_default_vars = all_vars[checkextra.(all_vars)]

    replacements = []
    for v in extra_default_vars
        newmeta = Base.ImmutableDict(filter(kv -> kv[1] != Symbolics.VariableDefaultValue,
            Dict(v.metadata))...)
        newv = @set v.metadata = newmeta
        push!(replacements, v => newv)
    end
    new_eqs = substitute.(equations(original_sys), (Dict(replacements...),))
    new_unk = get_unknowns(new_eqs)
    new_params = get_parameters(new_eqs)
    copy_with_change(original_sys; eqs = new_eqs, unknowns = new_unk,
        parameters = new_params)
end

"""
Initialize the state variable array.
"""
function init_u(mtk_sys::System, d::DomainInfo{ET}) where {ET}
    vars = unknowns(mtk_sys)
    ics = ModelingToolkit.initial_conditions(mtk_sys)
    # Convert symbolic numeric values to concrete Float64
    u0_single = [Symbolics.value(Symbolics.Num(ics[u])) for u in vars]

    u0 = init_array(d, length(u0_single)*prod(size(d)))
    u0_tmp = reshape(u0, length(u0_single), :)
    for (i, u) in enumerate(u0_single)
        @view(u0_tmp[i, :, :, :]) .= ET(u)
    end
    return u0
end

"""
    $(SIGNATURES)

Initialize an arrays with the given dimensions
"""
init_array(d::DomainInfo, sizes...) = similar(d.u_proto, sizes...)

function default_params(mtk_sys::AbstractSystem)
    ics = ModelingToolkit.initial_conditions(mtk_sys)
    MTKParameters(mtk_sys, ics)
end

# return whether the part of a after the last "₊" character matches b.
function is_matching_suffix(a, b)
    is_matching_suffix(a, var2symbol(b))
end
function is_matching_suffix(a, b::Symbol)
    split(String(var2symbol(a)), "₊")[end] == String(b)
end
function matching_suffix_idx(a::AbstractVector, b)
    findall(v -> is_matching_suffix(v, b), a)
end

# Return the coordinate parameters from the parameter vector.
function coord_params(mtk_sys::AbstractSystem, domain::DomainInfo)
    pv = pvars(domain)
    params = parameters(mtk_sys)

    _pvidx = [matching_suffix_idx(params, p) for p in pv]
    for (i, idx) in enumerate(_pvidx)
        if length(idx) > 1
            error("Partial independent variable '$(pv[i])' has multiple matches in system parameters: [$(parameters(mtk_sys)[idx])].")
        elseif length(idx) == 0
            error("Partial independent variable '$(pv[i])' not found in system parameters [$(parameters(mtk_sys))].")
        end
    end
    params[only.(_pvidx)]
end

# Create a function to set the coordinates in a parameter vector for a given grid cell
function coord_setter(sys_mtk::System, domain::DomainInfo)
    coords = coord_params(sys_mtk, domain)
    coord_setter = setp(sys_mtk, coords)
    II = CartesianIndices(tuple(size(domain)...))
    grd = grid(domain)
    function setp!(p, ii::CartesianIndex) # Set the parameters for the given grid cell index.
        vals = (g[ii[jj]] for (jj, g) in enumerate(grd)) # Get the coordinates of this grid cell.
        coord_setter(p, vals)
    end
    function setp!(p, j::Int) # Set the parameters for the jth grid cell.
        setp!(p, II[j])
    end
    return setp!
end
