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
    # Maps Symbol => IV for unit validation.
    iv_map = Dict{Symbol, Any}()

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
                # Validate unit compatibility.
                existing_iv = iv_map[sym]
                existing_unit = ModelingToolkit.get_unit(existing_iv)
                new_unit = ModelingToolkit.get_unit(iv)
                if !isequal(existing_unit, new_unit)
                    error("Unit mismatch for independent variable $sym: " *
                          "$existing_unit vs $new_unit (in system $k).")
                end
            else
                push!(seen, sym)
                push!(unified_ivs, iv)
                push!(unified_domains, dom)
                domain_map[sym] = dom
                iv_map[sym] = iv
            end
        end
    end

    return (unified_ivs, unified_domains)
end

"""
Deduplicate symbolic variables by comparing their symbol names.
"""
function unique_syms(syms::AbstractVector)
    seen = Dict{Symbol, Any}()  # name => first occurrence string representation
    result = eltype(syms)[]
    for s in syms
        sym = Symbol(Symbolics.tosymbol(s, escape = false))
        s_str = string(s)
        if !haskey(seen, sym)
            seen[sym] = s_str
            push!(result, s)
        elseif seen[sym] != s_str
            # Same name but different representation — keep both
            push!(result, s)
            @warn "Variables with same base name but different representations: $(seen[sym]) and $s_str"
        end
        # If seen[sym] == s_str, it's a true duplicate — skip
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

    # Add coupling equations from couple2 connectors as new equations.
    # In the ODE path, couple2 connector equations are composed via MTK's
    # compose — no additive merge. The PDE path should work the same way.
    for ceq in coupling_eqs
        push!(all_eqs, ceq)
    end

    # Merge BCs, DVs, Ps (deduplicate)
    all_bcs = unique_eqs(vcat([p.bcs for p in pdesystems]...))
    all_dvs = unique_syms(vcat([p.dvs for p in pdesystems]...))
    all_ps = unique_syms(vcat(
        [_collect_ps(p.ps) for p in pdesystems]...))

    # Ensure all LHS dependent variables are registered in dvs.
    # This handles variables introduced by coupling equations, such as
    # slice_variable outputs (which have distinct names like v_at_z_0ₓ0).
    existing_dv_names = Set(Symbol(Symbolics.tosymbol(dv, escape = false)) for dv in all_dvs)
    for eq in all_eqs
        for var in Symbolics.get_variables(eq.lhs)
            uvar = Symbolics.unwrap(var)
            # Only consider callable terms whose operator is NOT a Differential
            # (derivatives show up as callable but are not dependent variables).
            if Symbolics.iscall(uvar) && !(Symbolics.operation(uvar) isa Differential)
                vname = Symbol(Symbolics.tosymbol(var, escape = false))
                if vname ∉ existing_dv_names
                    push!(all_dvs, var)
                    push!(existing_dv_names, vname)
                end
            end
        end
    end

    # Scan all equations (including coupling) for parameters/constants not yet
    # collected. This catches @constants introduced in couple2 connector equations.
    existing_ps_names = Set(Symbol(Symbolics.tosymbol(s, escape = false)) for s in all_ps)
    iv_names = Set(Symbol.(unified_ivs))
    for eq in all_eqs
        for var in Symbolics.get_variables(eq)
            uvar = Symbolics.unwrap(var)
            vname = Symbol(Symbolics.tosymbol(var, escape = false))
            vname in existing_ps_names && continue
            vname in iv_names && continue
            vname in existing_dv_names && continue
            # Skip callable terms (DVs) - only collect leaf parameters/constants
            if Symbolics.iscall(uvar) && !(Symbolics.operation(uvar) isa Differential)
                continue
            end
            # Skip derivative symbols (e.g., uˍt from Differential decomposition)
            occursin("ˍ", string(vname)) && continue
            push!(all_ps, var)
            push!(existing_ps_names, vname)
        end
    end

    # Replace namespaced IV copies (e.g., LANDFIRE₊x) with bare IVs (x).
    # After composition, data source components may retain namespaced copies
    # of independent variables as parameters. MOL only knows about bare IVs.
    iv_subs = Dict{Any, Any}()
    iv_name_to_sym = Dict{String, Any}()
    for iv in unified_ivs
        iv_name_to_sym[string(Symbol(iv))] = iv
    end
    for p in all_ps
        p_str = string(Symbolics.tosymbol(p, escape = false))
        for (iv_name, iv_sym) in iv_name_to_sym
            if p_str != iv_name && endswith(p_str, "₊" * iv_name)
                iv_subs[Symbolics.unwrap(p)] = Symbolics.unwrap(iv_sym)
            end
        end
    end
    if !isempty(iv_subs)
        # Use substitute_in_deriv_and_depvar because the namespaced IVs
        # appear as arguments to dependent variable calls (e.g., v(t, LANDFIRE₊x))
        # and substitute_in_deriv alone doesn't recurse into depvar arguments.
        all_eqs = map(all_eqs) do eq
            Symbolics.substitute_in_deriv_and_depvar(eq.lhs, iv_subs) ~
                Symbolics.substitute_in_deriv_and_depvar(eq.rhs, iv_subs)
        end
        all_bcs = map(all_bcs) do bc
            Symbolics.substitute_in_deriv_and_depvar(bc.lhs, iv_subs) ~
                Symbolics.substitute_in_deriv_and_depvar(bc.rhs, iv_subs)
        end
        all_ps = filter(p -> !(Symbolics.unwrap(p) in keys(iv_subs)), all_ps)
    end

    # Collect initial_conditions from all PDESystems and parameter defaults
    # so that MOL's internal pipeline can find parameter values.
    merged_ics = Dict{Any, Any}()
    for p in pdesystems
        for (k, v) in p.initial_conditions
            merged_ics[k] = v
        end
    end
    for p in all_ps
        if ModelingToolkit.hasdefault(p)
            merged_ics[Symbolics.unwrap(p)] = ModelingToolkit.getdefault(p)
        end
    end

    # Add t=0 ICs for any DVs that lack one. This handles variables
    # promoted by param_to_var whose ICs were lost during DV dedup,
    # as well as any other DVs missing initial conditions.
    t_iv = unified_ivs[1]
    t_domain = first(d for d in unified_domains if isequal(d.variables, t_iv))
    t_start = DomainSets.infimum(t_domain.domain)
    # Check each DV: see if any existing BC is a t=0 IC for this DV.
    # An IC has the DV's operator called with a numeric first argument
    # (the t boundary), not the symbolic t variable.
    function _has_ic(dv, bcs, t_iv)
        dv_op = Symbolics.operation(Symbolics.unwrap(dv))
        for bc in bcs
            for var in Symbolics.get_variables(bc.lhs)
                uvar = Symbolics.unwrap(var)
                Symbolics.iscall(uvar) || continue
                Symbolics.operation(uvar) isa Differential && continue
                Symbolics.operation(uvar) === dv_op || continue
                # Check if the first argument is NOT the symbolic t variable
                # (i.e., it's a numeric boundary value like 0 or 0.0).
                first_arg = Symbolics.arguments(uvar)[1]
                if !isequal(first_arg, Symbolics.unwrap(t_iv))
                    return true
                end
            end
        end
        return false
    end
    # Only add ICs for DVs that have a time-derivative equation (D(dv) ~ ...).
    # Algebraic DVs (defined by dv ~ expression) are determined by their
    # algebraic equation and don't need ICs.
    dvs_with_deriv = Set{Any}()
    for eq in all_eqs
        for var in Symbolics.get_variables(eq)
            uvar = Symbolics.unwrap(var)
            if Symbolics.iscall(uvar) && Symbolics.operation(uvar) isa Differential
                # The argument of D(...) is the DV call - extract its operator.
                inner = Symbolics.arguments(uvar)[1]
                if Symbolics.iscall(Symbolics.unwrap(inner))
                    push!(dvs_with_deriv, Symbolics.operation(Symbolics.unwrap(inner)))
                end
            end
        end
    end
    for dv in all_dvs
        dv_op = Symbolics.operation(Symbolics.unwrap(dv))
        dv_op in dvs_with_deriv || continue  # skip algebraic DVs
        _has_ic(dv, all_bcs, t_iv) && continue
        uvar = Symbolics.unwrap(dv)
        op = Symbolics.operation(uvar)
        dv_args = Symbolics.arguments(uvar)
        dv_spatial = dv_args[2:end]
        push!(all_bcs, Symbolics.wrap(op)(t_start, dv_spatial...) ~ 0.0)
    end

    PDESystem(all_eqs, all_bcs, unified_domains, unified_ivs, all_dvs, all_ps;
        name = name, initial_conditions = merged_ics)
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

Returns `(new_dv, equation)` where `new_dv` is a new dependent variable with a
distinct name and `equation` defines `new_dv` in terms of the original variable
evaluated at `slice_value`. The distinct name avoids conflicts with the original
variable in the PDESystem `dvs` list.

# Arguments
- `var`: A symbolic dependent variable call, e.g., `U(t, x, y, lev)`
- `slice_dim`: The independent variable to fix, e.g., `lev`
- `slice_value`: The numeric value at which to evaluate `slice_dim`
- `name`: (keyword) Optional `Symbol` for the new variable. Defaults to
  `Symbol(varname, "_at_", dim)`, e.g., `:U_at_lev`.

# Example
```julia
@parameters x y lev
@variables U(..)
new_dv, eq = slice_variable(U(t, x, y, lev), lev, 1.0)
# new_dv = U_at_lev(t, x, y)
# eq: U_at_lev(t, x, y) ~ U(t, x, y, 1.0)
```
"""
function slice_variable(var, slice_dim, slice_value; name = nothing)
    args = Symbolics.arguments(Symbolics.unwrap(var))
    op = Symbolics.operation(Symbolics.unwrap(var))
    slice_sym = Symbol(slice_dim)

    # Build new argument lists
    reduced_args = [a for a in args if Symbol(a) != slice_sym]
    fixed_args = [Symbol(a) == slice_sym ? slice_value : a for a in args]

    # Create a new operator with a distinct name so the sliced variable
    # can coexist with the original in the PDESystem dvs list (MTK
    # forbids duplicate base names).
    if isnothing(name)
        base_name = Symbolics.tosymbol(var, escape = false)
        name = Symbol(base_name, "_at_", slice_sym)
    end
    new_name = name
    new_op = only(@variables $new_name(..))
    new_op = add_metadata(new_op, var)
    new_dv = new_op(reduced_args...)
    fixed_var = Symbolics.wrap(op)(fixed_args...)

    return (new_dv, new_dv ~ fixed_var)
end
