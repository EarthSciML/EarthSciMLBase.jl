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
function observed_expression(sys::ODESystem, x; extra_eqs=[])
    expr = nothing
    eqs = observed(sys)
    push!(eqs, extra_eqs...)
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
        v_expr = observed_expression(sys, v)
        if !isnothing(v_expr)
            expr = Symbolics.replace(expr, v => v_expr)
        end
    end
    # Do it again to catch extra variables TODO(CT): Theoretically this could recurse forever; when to stop?
    for v ∈ Symbolics.get_variables(expr)
        v_expr = observed_expression(sys, v, extra_eqs=extra_eqs)
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
function observed_function(sys::ODESystem, x, coords; extra_eqs=[])
    expr = observed_expression(sys, x, extra_eqs=extra_eqs)
    vars = Symbolics.get_variables(expr)
    @assert (length(vars) <= length(coords)) "Extra variables: $(vars) != $(coords)"
    @assert all(Bool.([sum(isequal.((v,), coords)) for v ∈ vars])) "Incorrect variables: $(vars) != $(coords)"
    return Symbolics.build_function(expr, coords...; expression=Val{false})
end

"""
$(SIGNATURES)

Return the data type of the state variables for this domain,
based on the data types of the boundary conditions domain intervals.
"""
function utype(_::DomainInfo{T}) where T
    return T
end

"""
$(SIGNATURES)

Return the ranges representing the discretization of the partial independent 
variables for this domain, based on the discretization intervals given in `Δs`
"""
function grid(d::DomainInfo{T}, Δs::AbstractVector)::Vector{AbstractRange{T}} where {T<:AbstractFloat}
    i = 1
    rngs = []
    for icbc ∈ d.icbc
        if icbc isa BCcomponent
            for pd ∈ icbc.partialdomains
                rng = DomainSets.infimum(pd.domain):Δs[i]:DomainSets.supremum(pd.domain)
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
function time_range(d::DomainInfo{T})::Tuple{T, T} where T<:AbstractFloat
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
function timesteps(tsteps::AbstractVector{T}...)::Vector{T} where T<:AbstractFloat
    allt = sort(union(vcat(tsteps...)))
    allt2 = [allt[1]]
    for i ∈ 2:length(allt) # Remove nearly duplicate times.
        if allt[i] ≉ allt[i-1]
            push!(allt2, allt[i])
        end
    end
    allt2
end