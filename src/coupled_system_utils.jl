"""
$(SIGNATURES)

Return the time step length common to all of the given `timesteps`.
Throw an error if not all timesteps are the same length.
"""
function steplength(timesteps)
    Δs = [timesteps[i] - timesteps[i-1] for i ∈ 2:length(timesteps)]
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
    expr = nothing
    for eq ∈ eqs
        if isequal(eq.lhs, x)
            expr = eq.rhs
        end
    end
    if isnothing(expr)
        return nothing
    end
    expr = ModelingToolkit.subs_constants(expr)
    for v ∈ Symbolics.get_variables(expr)
        v_expr = observed_expression(eqs, v)
        if !isnothing(v_expr)
            expr = Symbolics.substitute(expr, v => v_expr)
        end
    end
    # Do it again to catch extra variables TODO(CT): Theoretically this could recurse forever; when to stop?
    for v ∈ Symbolics.get_variables(expr)
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
    expr = observed_expression(eqs, x)
    vars = Symbolics.get_variables(expr)
    if length(vars) > length(coords)
        @warn "Extra variables: $(vars) > $(coords) in observed function for $x."
        return (x...) -> 0.0
    end
    coordvars = []
    for c ∈ coords
        i = findfirst(v -> split(String(Symbol(v)), "₊")[end] == String(Symbol(c)), vars)
        if isnothing(i)
            push!(coordvars, c)
        else
            push!(coordvars, vars[i])
        end
    end
    return Symbolics.build_function(expr, coordvars...; expression=Val{false})
end

"""
$(SIGNATURES)

Return the time points during which integration should be stopped to run the operators.
"""
function timesteps(tsteps::AbstractVector{T}...)::Vector{T} where {T<:AbstractFloat}
    allt = sort(union(vcat(tsteps...)))
    allt2 = [allt[1]]
    for i ∈ 2:length(allt) # Remove nearly duplicate times.
        if allt[i] ≉ allt[i-1]
            push!(allt2, allt[i])
        end
    end
    allt2
end

"""
$(SIGNATURES)

Return the indexes of the system variables that the state variables of the final
simplified system depend on. This should be done before running `structural_simplify`
on the system.
"""
function get_needed_vars(sys::ODESystem)
    varvardeps = ModelingToolkit.varvar_dependencies(
        ModelingToolkit.asgraph(sys),
        ModelingToolkit.variable_dependencies(sys),
    )
    g = SimpleDiGraph(length(unknowns(sys)))
    for (i, es) in enumerate(varvardeps.badjlist)
        for e in es
            add_edge!(g, i, e)
        end
    end
    allst = unknowns(sys)
    simpst = unknowns(structural_simplify(sys))
    stidx = [only(findall(isequal(s), allst)) for s in simpst]
    collect(Graphs.DFSIterator(g, stidx))
end

"""
$(SIGNATURES)

Create a copy of an ODESystem with the given changes.
"""
function copy_with_change(sys::ODESystem;
    eqs=equations(sys),
    name=nameof(sys),
    metadata=ModelingToolkit.get_metadata(sys),
    continuous_events=ModelingToolkit.get_continuous_events(sys),
    discrete_events=ModelingToolkit.get_discrete_events(sys),
)
    ODESystem(eqs, ModelingToolkit.get_iv(sys), name=name, metadata=metadata,
        continuous_events=continuous_events, discrete_events=discrete_events)
end

"""
$(SIGNATURES)

Remove equations from an ODESystem where the variable in the LHS is not
present in any of the equations for the state variables. This can be used to
remove computationally intensive equations that are not used in the final model.
"""
function prune_observed(sys::ODESystem)
    needed_var_idxs = get_needed_vars(sys)
    needed_vars = Symbolics.tosymbol.(unknowns(sys)[needed_var_idxs]; escape=true)
    sys = structural_simplify(sys, split=false)
    deleteindex = []
    for (i, eq) ∈ enumerate(observed(sys))
        lhsvars = Symbolics.tosymbol.(Symbolics.get_variables(eq.lhs); escape=true)
        # Only keep equations where all variables on the LHS are in at least one
        # equation describing the system state.
        if !all((var) -> var ∈ needed_vars, lhsvars)
            push!(deleteindex, i)
        end
    end
    obs = observed(sys)
    deleteat!(obs, deleteindex)
    sys2 = structural_simplify(
        copy_with_change(sys; eqs=[equations(sys); obs]),
        split=false,
    )
    return sys2, observed(sys)
end

# Remove extra variable defaults that would cause a solver initialization error.
# This should be done before running `structural_simplify` on the system.
function remove_extra_defaults(sys)
    all_vars = unique(vcat(get_variables.(equations(sys))...))

    unk = Symbol.(unknowns(structural_simplify(sys)))

    # Check if v is not in the unknowns, is a variable, and has a default.
    checkextra(v) = !(Symbol(v) in unk) &&
                    v.metadata[Symbolics.VariableSource][1] == :variables &&
                    (Symbolics.VariableDefaultValue in keys(v.metadata))
    extra_default_vars = all_vars[checkextra.(all_vars)]

    replacements = []
    for v in extra_default_vars
        newmeta = Base.ImmutableDict(filter(kv -> kv[1] != Symbolics.VariableDefaultValue,
            Dict(v.metadata))...)
        newv = @set v.metadata = newmeta
        push!(replacements, v => newv)
    end
    new_eqs = substitute.(equations(sys), (Dict(replacements...),))
    copy_with_change(sys, eqs=new_eqs)
end

"Initialize the state variables."
function init_u(mtk_sys::ODESystem, d::DomainInfo)
    vars = unknowns(mtk_sys)
    dflts = ModelingToolkit.get_defaults(mtk_sys)
    u0 = [dflts[u] for u ∈ vars]

    T = dtype(d)
    g = grid(d)
    u = Array{T}(undef, length(vars), size(d)...)
    # Set initial conditions
    for i ∈ eachindex(u0), j ∈ eachindex(g[1]), k ∈ eachindex(g[2]), l ∈ eachindex(g[3])
        u[i, j, k, l] = u0[i]
    end
    u
end

function default_params(mtk_sys::AbstractSystem)
    dflts = ModelingToolkit.get_defaults(mtk_sys)
    [dflts[p] for p ∈ parameters(mtk_sys)]
end

# Return the indexes of the coordinate parameters in the parameter vector.
function coord_idx(mtk_sys::AbstractSystem, domain::DomainInfo)
    pv = pvars(domain)
    _pvidx = [findall(v -> split(String(Symbol(v)), "₊")[end] == String(Symbol(p)), parameters(mtk_sys)) for p ∈ pv]
    for (i, idx) in enumerate(_pvidx)
        if length(idx) > 1
            error("Partial independent variable '$(pv[i])' has multiple matches in system parameters: [$(parameters(mtk_sys)[idx])].")
        elseif length(idx) == 0
            error("Partial independent variable '$(pv[i])' not found in system parameters [$(parameters(mtk_sys))].")
        end
    end
    only.(_pvidx)
end

# Create a function to set the coordinates in a parameter vector for a given grid cell
function coord_setter(sys_mtk::ODESystem, domain::DomainInfo)
    icoord = coord_idx(sys_mtk, domain)
    II = CartesianIndices(tuple(size(domain)...))
    grd = grid(domain)
    function setp!(p, ii::CartesianIndex) # Set the parameters for the give grid cell index.
        for (jj, g) ∈ enumerate(grd) # Set the coordinates of this grid cell.
            p[icoord[jj]] = g[ii[jj]]
        end
    end
    function setp!(p, j::Int) # Set the parameters for the jth grid cell.
        setp!(p, II[j])
    end
    return setp!
end

# Create functions to get concrete values for the observed variables.
# The return value is a function that returns the function when
# given a value.
function obs_functions(obs_eqs, domain::DomainInfo)
    pv = pvars(domain)
    iv = ivar(domain)

    obs_fs_idx = Dict()
    obs_fs = []
    for (i, x) ∈ enumerate([eq.lhs for eq ∈ obs_eqs])
        obs_fs_idx[x] = i
        push!(obs_fs, observed_function(obs_eqs, x, [iv, pv...]))
    end
    obs_fs = Tuple(obs_fs)

    (v) -> obs_fs[obs_fs_idx[v]]
end

# Return functions to perform coordinate transforms for each of the coordinates.
function coord_trans_functions(obs_eqs, domain::DomainInfo)
    pv = pvars(domain)
    iv = ivar(domain)

    # Get functions for coordinate transforms
    tf_fs = []
    @variables 🌈🐉🏒 # Dummy variable.
    for tf ∈ partialderivative_transforms(domain)
        push!(tf_fs, observed_function([obs_eqs..., 🌈🐉🏒 ~ tf], 🌈🐉🏒, [iv, pv...]))
    end
    tf_fs = Tuple(tf_fs)
end
