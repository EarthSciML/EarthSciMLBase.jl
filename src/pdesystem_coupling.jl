"""
    $(SIGNATURES)

Validate that all PDESystems have compatible domains and compute the union of
their independent variables and domains.

Systems may have different numbers of spatial dimensions (e.g., a 2D system
coupled with a 3D system). For any independent variable that appears in
multiple systems, the domain ranges must match. The returned IVs and domains
are the union across all systems, preserving the order from the system with
the most dimensions and appending any additional IVs from other systems.

Returns `(unified_ivs, unified_domains)`.
"""
function validate_and_unify_domains(pdesystems::AbstractVector{<:ModelingToolkit.PDESystem})
    @assert !isempty(pdesystems) "At least one PDESystem is required."

    if length(pdesystems) == 1
        p = only(pdesystems)
        return (p.ivs, p.domain)
    end

    # Start from the system with the most IVs so that the ordering is natural.
    sorted_idx = sortperm(pdesystems; by = p -> length(p.ivs), rev = true)

    seen = Set{Symbol}()
    unified_ivs = []
    unified_domains = []
    # Maps Symbol => domain spec for validation of shared IVs.
    domain_map = Dict{Symbol, Any}()

    for k in sorted_idx
        pdesys = pdesystems[k]
        @assert length(pdesys.ivs) == length(pdesys.domain) "System $k has $(length(pdesys.ivs)) IVs but $(length(pdesys.domain)) domains."
        for (iv, dom) in zip(pdesys.ivs, pdesys.domain)
            sym = Symbol(iv)
            if sym ∈ seen
                # Validate domain range matches the first occurrence.
                existing_dom = domain_map[sym]
                lo1 = DomainSets.infimum(existing_dom.domain)
                hi1 = DomainSets.supremum(existing_dom.domain)
                lo2 = DomainSets.infimum(dom.domain)
                hi2 = DomainSets.supremum(dom.domain)
                if !(lo1 ≈ lo2) || !(hi1 ≈ hi2)
                    error("Domain range mismatch for $sym: [$lo1, $hi1] vs [$lo2, $hi2] " *
                          "(in system $k).")
                end
            else
                push!(seen, sym)
                push!(unified_ivs, iv)
                push!(unified_domains, dom)
                domain_map[sym] = dom
            end
        end
    end

    return (unified_ivs, unified_domains)
end

"""
Deduplicate symbolic variables by comparing their symbol names.
"""
function unique_syms(syms::AbstractVector)
    seen = Set{Symbol}()
    result = eltype(syms)[]
    for s in syms
        sym = Symbol(Symbolics.tosymbol(s, escape = false))
        if sym ∉ seen
            push!(seen, sym)
            push!(result, s)
        end
    end
    result
end

"""
Deduplicate equations by symbolic equality of both LHS and RHS.
"""
function unique_eqs(eqs::AbstractVector{<:Equation})
    result = Equation[]
    for eq in eqs
        found = false
        for existing in result
            if isequal(existing.lhs, eq.lhs) && isequal(existing.rhs, eq.rhs)
                found = true
                break
            end
        end
        if !found
            push!(result, eq)
        end
    end
    result
end

"""
    $(SIGNATURES)

Merge multiple PDESystems into a single flat PDESystem.

Input PDESystems may have different numbers of spatial dimensions. For shared
independent variables, domain ranges must match. The merged system uses the
union of all independent variables and domains, and each dependent variable
retains its original dimensions.

Coupling equations are applied by matching LHS: if a coupling equation has the same LHS
as an existing equation, its RHS is added to the existing equation's RHS. Otherwise, it
is added as a new equation.

# Arguments
- `pdesystems`: Vector of PDESystems to merge
- `coupling_eqs`: Additional coupling equations to apply (default: empty)
- `name`: Name for the resulting PDESystem (default: `:coupled`)
"""
function merge_pdesystems(pdesystems::AbstractVector{<:ModelingToolkit.PDESystem},
        coupling_eqs::Vector{Equation} = Equation[];
        name = :coupled)
    @assert !isempty(pdesystems) "At least one PDESystem is required."

    if length(pdesystems) == 1 && isempty(coupling_eqs)
        return only(pdesystems)
    end

    unified_ivs, unified_domains = validate_and_unify_domains(pdesystems)

    # Collect all equations
    all_eqs = Equation[]
    for p in pdesystems
        append!(all_eqs, equations(p))
    end

    # Apply coupling equations
    for ceq in coupling_eqs
        idx = findfirst(eq -> isequal(eq.lhs, ceq.lhs), all_eqs)
        if idx !== nothing
            all_eqs[idx] = all_eqs[idx].lhs ~ all_eqs[idx].rhs + ceq.rhs
        else
            push!(all_eqs, ceq)
        end
    end

    # Merge BCs, DVs, Ps (deduplicate)
    all_bcs = unique_eqs(vcat([p.bcs for p in pdesystems]...))
    all_dvs = unique_syms(vcat([p.dvs for p in pdesystems]...))
    all_ps = unique_syms(vcat(
        [_collect_ps(p.ps) for p in pdesystems]...))

    PDESystem(all_eqs, all_bcs, unified_domains, unified_ivs, all_dvs, all_ps; name = name)
end

# Safely collect parameters, handling NullParameters from SciMLBase.
_collect_ps(ps::AbstractVector) = ps
_collect_ps(_) = Num[]  # Fallback for NullParameters or other non-iterable types

"""
    $(SIGNATURES)

Create a lower-dimensional dependent variable by fixing one spatial dimension
of a higher-dimensional variable at a specific value. This is useful for
coupling systems with different numbers of spatial dimensions, e.g.,
extracting ground-level data from a 3D atmospheric variable for use in a
2D surface model.

Returns `(new_dv, equation)` where `new_dv` is the sliced dependent variable
(with the fixed dimension removed from its arguments) and `equation` defines
`new_dv` in terms of the original variable evaluated at `slice_value`.

# Arguments
- `var`: A symbolic dependent variable call, e.g., `U(t, x, y, lev)`
- `slice_dim`: The independent variable to fix, e.g., `lev`
- `slice_value`: The numeric value at which to evaluate `slice_dim`

# Example
```julia
@parameters x y lev
@variables U(..)
new_dv, eq = slice_variable(U(t, x, y, lev), lev, 1.0)
# new_dv = U(t, x, y)
# eq: U(t, x, y) ~ U(t, x, y, 1.0)
```
"""
function slice_variable(var, slice_dim, slice_value)
    args = Symbolics.arguments(Symbolics.unwrap(var))
    op = Symbolics.operation(Symbolics.unwrap(var))
    slice_sym = Symbol(slice_dim)

    # Build new argument lists
    reduced_args = [a for a in args if Symbol(a) != slice_sym]
    fixed_args = [Symbol(a) == slice_sym ? slice_value : a for a in args]

    new_dv = Symbolics.wrap(op)(reduced_args...)
    fixed_var = Symbolics.wrap(op)(fixed_args...)

    return (new_dv, new_dv ~ fixed_var)
end
