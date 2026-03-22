"""
    $(SIGNATURES)

Validate that all PDESystems have compatible independent variables and domains.
Checks that all systems share the same set of independent variables (by name)
and that their domain ranges match.
"""
function validate_compatible_domains(pdesystems::AbstractVector{<:ModelingToolkit.PDESystem})
    length(pdesystems) <= 1 && return nothing

    ref = first(pdesystems)
    ref_ivs = Set(Symbol.(ref.ivs))
    ref_domains = ref.domain

    for (k, pdesys) in enumerate(pdesystems)
        k == 1 && continue
        # Check independent variables match
        ivs = Set(Symbol.(pdesys.ivs))
        if ivs != ref_ivs
            error("PDESystems must share the same independent variables. " *
                  "System 1 has $(ref_ivs) but system $k has $(ivs).")
        end

        # Check domain ranges are compatible
        if length(pdesys.domain) != length(ref_domains)
            error("PDESystems must have the same number of domains. " *
                  "System 1 has $(length(ref_domains)) but system $k has $(length(pdesys.domain)).")
        end
        for (i, (d1, d2)) in enumerate(zip(ref_domains, pdesys.domain))
            v1 = Symbol(d1.variables)
            v2 = Symbol(d2.variables)
            if v1 != v2
                error("Domain variable mismatch at position $i: $v1 vs $v2.")
            end
            lo1, hi1 = DomainSets.infimum(d1.domain), DomainSets.supremum(d1.domain)
            lo2, hi2 = DomainSets.infimum(d2.domain), DomainSets.supremum(d2.domain)
            if !(lo1 ≈ lo2) || !(hi1 ≈ hi2)
                error("Domain range mismatch for $v1: [$lo1, $hi1] vs [$lo2, $hi2].")
            end
        end
    end
    nothing
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

All input PDESystems must share the same independent variables and compatible domain ranges.
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

    validate_compatible_domains(pdesystems)

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

    # Use IVs and domains from the first system (validated compatible)
    ivs = first(pdesystems).ivs
    domains = first(pdesystems).domain

    PDESystem(all_eqs, all_bcs, domains, ivs, all_dvs, all_ps; name = name)
end

# Safely collect parameters, handling NullParameters from SciMLBase.
_collect_ps(ps::AbstractVector) = ps
_collect_ps(_) = Num[]  # Fallback for NullParameters or other non-iterable types
