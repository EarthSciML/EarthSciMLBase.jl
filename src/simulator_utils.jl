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
            expr = Symbolics.replace(expr, v => v_expr)
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

Return the data type of the state variables for this domain,
based on the data types of the boundary conditions domain intervals.
"""
function utype(_::DomainInfo{T}) where {T}
    return T
end

"""
$(SIGNATURES)

Return the ranges representing the discretization of the partial independent 
variables for this domain, based on the discretization intervals given in `Δs`
"""
function grid(d::DomainInfo{T}, Δs::AbstractVector) where {T<:AbstractFloat}
    i = 1
    rngs = []
    for icbc ∈ d.icbc
        if icbc isa BCcomponent
            for pd ∈ icbc.partialdomains
                rng = T(DomainSets.infimum(pd.domain)):T(Δs[i]):T(DomainSets.supremum(pd.domain))
                push!(rngs, rng)
                i += 1
            end
        end
    end
    @assert length(rngs) == length(Δs) "The number of partial independent variables ($(length(rng))) must equal the number of Δs provided ($(length(Δs)))"
    return rngs
end

"""
$(SIGNATURES)

Return the time range associated with this domain.
"""
function time_range(d::DomainInfo{T})::Tuple{T,T} where {T<:AbstractFloat}
    for icbc ∈ d.icbc
        if icbc isa ICcomponent
            return DomainSets.infimum(icbc.indepdomain.domain), DomainSets.supremum(icbc.indepdomain.domain)
        end
    end
    throw(ArgumentError("Could not find a time range for this domain."))
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
    g = SimpleDiGraph(length(states(sys)))
    for (i, es) in enumerate(varvardeps.badjlist)
        for e in es
            add_edge!(g, i, e)
        end
    end
    allst = states(sys)
    simpst = states(structural_simplify(sys))
    stidx = [only(findall(isequal(s), allst)) for s in simpst]
    collect(Graphs.DFSIterator(g, stidx))    
end

"""
$(SIGNATURES)

Remove equations from an ODESystem where the variable in the LHS is not
present in any of the equations for the state variables. This can be used to 
remove computationally intensive equations that are not used in the final model.
"""
function prune_observed(sys::ODESystem)
    needed_var_idxs = get_needed_vars(sys)
    needed_vars = Symbolics.tosymbol.(states(sys)[needed_var_idxs]; escape=true)
    sys = structural_simplify(sys)
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
    sys2 = structural_simplify(ODESystem([equations(sys); obs], 
        ModelingToolkit.get_iv(sys), name=nameof(sys)))
    return sys2, observed(sys)
end