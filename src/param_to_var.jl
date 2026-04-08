export param_to_var, get_promoted_dv

"""
Add the units and description in the variable `from` to the variable `to`.
"""
function add_metadata(to, from; exclude_default = false)
    unit = ModelingToolkit.get_unit(from)
    to = Symbolics.setmetadata(to, ModelingToolkit.VariableUnit, unit)
    desc = ModelingToolkit.getdescription(from)
    to = Symbolics.setmetadata(to, ModelingToolkit.VariableDescription, desc)
    if ModelingToolkit.hasdefault(from) && !exclude_default
        default = ModelingToolkit.getdefault(from)
        to = ModelingToolkit.setdefault(to, default)
    end
    to
end

"""
Replace the parameter `p` in the system `sys` with a new variable that has
the same name, units, and description as `p`.

$(SIGNATURES)

This can be useful to replace a parameter that does not change in time in a model component
with one specified by another system that does change in time (or space). For example, the
code below specifies a first-order loss equation, and then changes the temperature (which
determines the loss rate) with a temperature value that varies in time.

```
```
"""
function param_to_var(sys::ModelingToolkit.AbstractSystem, ps::Symbol...)
    params = parameters(sys)
    defaults = copy(getfield(sys, :initial_conditions))
    replace = Dict()
    for p in ps
        if p in ModelingToolkit.tosymbol.(unknowns(sys), escape = false) # Skip if it is already a variable.
            continue
        end
        iparam = findfirst(isequal(p), Symbol.(params))
        @assert !isnothing(iparam) "Parameter `$p` not found in the system parameters $(Symbol.(params))"
        param = params[iparam]

        iv = ModelingToolkit.get_iv(sys)
        newvar = only(@variables $p(iv))
        newvar = add_metadata(newvar, param; exclude_default = true)
        replace[param] = newvar
        delete!(defaults, param)
    end

    if isempty(replace)
        return sys # Nothing to replace
    end
    newsys = SymbolicUtils.substitute(sys, replace)
    copy_with_change(newsys;
        metadata = ModelingToolkit.get_metadata(sys),
        discrete_events = ModelingToolkit.get_discrete_events(sys),
        continuous_events = ModelingToolkit.get_continuous_events(sys),
        defaults = defaults
    )
end

"""
Replace the parameter(s) `ps` in a `PDESystem` with new time-dependent variable(s)
that have the same name, units, and description.

$(SIGNATURES)

Since `PDESystem` does not support `SymbolicUtils.substitute`, this method manually
substitutes in all equations and boundary conditions, then reconstructs the `PDESystem`.
"""
function param_to_var(sys::ModelingToolkit.PDESystem, ps::Symbol...)
    params = sys.ps
    replace = Dict()
    for p in ps
        dv_names = [Symbolics.tosymbol(dv, escape = false) for dv in sys.dvs]
        ns_name = Symbol(nameof(sys), "₊", p)
        if ns_name in dv_names || p in dv_names
            continue
        end
        iparam = findfirst(isequal(p), Symbol.(params))
        if isnothing(iparam)
            # Parameter not found — it may have already been converted to a
            # variable by a prior call to param_to_var (e.g. when couple2 is
            # invoked at both Phase 1.5 and Phase 3).  Skip silently.
            continue
        end
        param = params[iparam]

        # Create a variable with ALL independent variables so it has the
        # same spatial dimensionality as the other dependent variables.
        # Namespace the variable with the system name (e.g., sysname₊p)
        # to avoid naming collisions when multiple PDESystems are merged.
        ivs = sys.ivs
        ns_p = Symbol(nameof(sys), "₊", p)
        newvar = only(@variables $ns_p(..))
        newvar = add_metadata(newvar, param; exclude_default = true)
        newvar_call = newvar(ivs...)
        replace[Symbolics.unwrap(param)] = Symbolics.unwrap(newvar_call)
    end

    if isempty(replace)
        return sys
    end

    # Manually substitute in equations and boundary conditions.
    # Use substitute_in_deriv_and_depvar for Symbolics v7 compatibility:
    # substitute alone doesn't recurse into Differential or depvar arguments.
    new_eqs = map(sys.eqs) do eq
        Symbolics.substitute_in_deriv_and_depvar(eq.lhs, replace) ~
            Symbolics.substitute_in_deriv_and_depvar(eq.rhs, replace)
    end
    new_bcs = map(sys.bcs) do bc
        Symbolics.substitute_in_deriv_and_depvar(bc.lhs, replace) ~
            Symbolics.substitute_in_deriv_and_depvar(bc.rhs, replace)
    end

    # Remove converted parameters
    new_ps = [p for p in params if !(Symbolics.unwrap(p) in keys(replace))]

    # Add promoted variables to the dependent variables list
    new_dvs = copy(sys.dvs)
    for newvar_unwrapped in values(replace)
        push!(new_dvs, Symbolics.wrap(newvar_unwrapped))
    end

    # Forward initial_conditions, removing converted parameters.
    # Note: ICs for promoted variables are NOT added here because the DV
    # names may change during merge_pdesystems (DV dedup). Instead,
    # merge_pdesystems adds ICs for any DVs that lack them.
    new_ics = Dict{Any, Any}()
    for (k, v) in sys.initial_conditions
        if !(k in keys(replace))
            new_ics[k] = v
        end
    end

    PDESystem(new_eqs, new_bcs, sys.domain, sys.ivs, new_dvs, new_ps;
        name = nameof(sys), metadata = sys.metadata, initial_conditions = new_ics)
end

"""
    get_promoted_dv(sys::ModelingToolkit.PDESystem, base_name::Symbol)

Return the dependent variable from `sys` whose base name (the part after
the last `₊`) matches `base_name`. This is useful in `couple2` methods
after calling [`param_to_var`](@ref) to retrieve the promoted variable
without hardcoding the namespace prefix.

$(SIGNATURES)
"""
function get_promoted_dv(sys::ModelingToolkit.PDESystem, base_name::Symbol)
    for dv in sys.dvs
        dv_sym = Symbolics.tosymbol(dv, escape = false)
        if dv_sym == base_name ||
           Symbol(last(split(string(dv_sym), "₊"))) == base_name
            return dv
        end
    end
    error("Promoted variable for :$base_name not found in system $(nameof(sys))")
end
