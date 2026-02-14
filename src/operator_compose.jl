export operator_compose

"""
Return the dependent variable, which is the first argument of the term,
unless the term is a time derivative, in which case the dependent variable
is the argument of the time derivative.
"""
function get_dv(term, iv)
    if operation(term) == Differential(iv)
        return arguments(term)[1]
    else
        return term
    end
end

"""
$(SIGNATURES)

Add a system scope to a variable name, for example so that
`x` in system `sys1` becomes `sys1₊x`.
`iv` is the independent variable.
"""
function add_scope(sys, v, iv)
    n = String(nameof(sys))
    vstr = String(Symbolics.tosymbol(v, escape = false))
    vsym = Symbol("$(n)₊$(vstr)")
    vv = (@variables $vsym(iv))[1]
    add_metadata(vv, v)
end

# Convert translation dictionary to the format (a_var => b_var, factor)
function normalize_translate(translate::AbstractVector)
    function process_entry(entry::Pair)
        if entry.second isa Pair
            return (entry.first => entry.second.first, entry.second.second)
        else
            return (entry.first => entry.second, 1)
        end
    end
    process_entry.(translate)
end
function normalize_translate(translate::T) where {T <: AbstractDict}
    normalize_translate([k => v for (k, v) in translate])
end
function normalize_translate(translate)
    error("Invalid translation object: $translate")
end

function get_matching_translate(translate, a)
    idx = findall(x -> isequal(x[1].first, a), translate)
    translate[idx]
end

# Generate a unique name for a connector variable.
function connector_name(name::AbstractString, taken_names::AbstractDict)
    if haskey(taken_names, name)
        i = 1
        while haskey(taken_names, "$(name)_$(i)")
            i += 1
        end
        taken_names["$(name)_$(i)"] = true
        return Symbol("$(name)_$(i)"), taken_names
    else
        taken_names[name] = true
        return Symbol(name), taken_names
    end
end

"""
$(SIGNATURES)

Compose to systems of equations together by adding the right-hand side terms together of equations that have matching left-hand sides.
The left hand sides of two equations will be considered matching if:

 1. They are both time derivatives of the same variable.
 2. The first one is a time derivative of a variable and the second one is the variable itself.
 3. There is an entry in the optional `translate` dictionary that maps the dependent variable in the first system to the dependent variable in the second system, e.g. `Dict(sys1.sys.x => sys2.sys.y)`.
 4. There is an entry in the optional `translate` dictionary that maps the dependent variable in the first system to the dependent variable in the second system, with a conversion factor, e.g. `Dict(sys1.sys.x => sys2.sys.y => 6)`.
"""
function operator_compose(a::ModelingToolkit.System, b::ModelingToolkit.System,
        translate = Dict())
    translate = normalize_translate(translate)
    a_eqs = deepcopy(equations(a))
    b_eqs = deepcopy(equations(b))
    iv = ModelingToolkit.get_iv(a) # independent variable
    aname = String(nameof(a))
    bname = String(nameof(b))
    connections = Equation[]
    all_matches = []
    all_beq_matches = []
    taken_names = Dict()
    extra_params = Set()
    extra_avars = Set()
    extra_bvars = Set()
    for a_eq in a_eqs
        if isequal(a_eq.lhs, 0)
            # If the LHS == 0 (i.e. everything has already been shifted to the RHS),
            # there's not anything we can do.
            push!(all_matches, [])
            push!(all_beq_matches, [])
            continue
        end
        adv = add_scope(a, get_dv(a_eq.lhs, iv), iv) # dependent variable
        matches = get_matching_translate(translate, adv)
        if length(matches) == 0 # If adv is not in the translation dictionary, then assume it is the same in both systems.
            bdv, conv = add_scope(b, get_dv(a_eq.lhs, iv), iv), 1
            matches = [(adv => bdv, conv)]
        end
        b_eq_matches = [
                        # If the LHS == 0 (i.e. everything has already been shifted to the RHS),
                        # there's not anything we can do.
                        [!isequal(b_eq.lhs, 0) &&
                         isequal(abdv.second, add_scope(b, get_dv(b_eq.lhs, iv), iv))
                         for b_eq in b_eqs]
                        for (abdv, _) in matches]
        push!(all_matches, matches)
        push!(all_beq_matches, b_eq_matches)
    end
    for i in eachindex(a_eqs)
        matches = all_matches[i]
        b_eq_matches = all_beq_matches[i]
        for (ii, match) in enumerate(matches)
            adv, bdv, conv = match[1].first, match[1].second, match[2]
            if ModelingToolkit.isparameter(conv)
                push!(extra_params, conv)
            end
            js = (1:length(b_eqs))[b_eq_matches[ii]]
            for j in js
                bdv = add_scope(b, get_dv(b_eqs[j].lhs, iv), iv) # Make sure the units are correct.
                # The dependent variable of the LHS of this equation matches the dependent
                # variable of interest,
                # so create a new variable to represent the dependent variable
                # of interest and add it to the RHS of the other equation, and then also set the two
                # dependent variables to be equal.
                bvar = String(Symbolics.tosymbol(b_eqs[j].lhs, escape = false))
                if operation(b_eqs[j].lhs) == Differential(iv)
                    # The LHS of this equation is the time derivative of the dependent variable of interest,
                    var1, taken_names = connector_name("$(bname)_ddt_$(bvar)", taken_names)
                    term1 = (@variables $var1(iv))[1]
                    term1 = add_metadata(term1, b_eqs[j].lhs; exclude_default = true)
                    b_eqs[j] = term1 ~ b_eqs[j].rhs
                    push!(extra_bvars, term1)

                    var2 = Symbol("$(aname)₊", var1)
                    term2 = (@variables $var2(iv))[1]
                    term2 = add_metadata(term2, b_eqs[j].lhs; exclude_default = true)
                    var3 = Symbol("$(bname)₊", var1)
                    term3 = (@variables $var3(iv))[1]
                    term3 = add_metadata(term3, b_eqs[j].lhs; exclude_default = true)
                    push!(connections, term2 ~ term3)
                    a_eqs[i] = a_eqs[i].lhs ~ a_eqs[i].rhs + term1 * conv
                    push!(extra_avars, term1)
                    # Now set the dependent variables in the two systems to be equal.
                    push!(connections, adv ~ bdv * conv)
                else # The LHS of this equation is the dependent variable of interest.
                    var1, taken_names = connector_name("$(bname)_$(bvar)", taken_names)
                    term1 = (@variables $var1(iv))[1]
                    term1 = add_metadata(term1, b_eqs[j].lhs * conv; exclude_default = true)
                    var2 = Symbol("$(aname)₊", var1)
                    term2 = (@variables $var2(iv))[1]
                    term2 = add_metadata(term2, b_eqs[j].lhs * conv; exclude_default = true)
                    a_eqs[i] = a_eqs[i].lhs ~ a_eqs[i].rhs + term1
                    push!(extra_avars, term1)
                    push!(connections, term2 ~ bdv * conv)
                end
            end
        end
    end
    aa = copy_with_change(a; eqs = a_eqs, unknowns = [unknowns(a); extra_avars...],
        parameters = [parameters(a); extra_params...])
    bb = copy_with_change(b; eqs = b_eqs, unknowns = [unknowns(b); extra_bvars...],
        parameters = [parameters(b); extra_params...])
    ConnectorSystem(connections, aa, bb)
end

# PDESystems don't have a compose function, so we just add the equations together
# here without trying to keep the systems in separate namespaces.
# TODO(CT): Handle events
function operator_compose!(a::ModelingToolkit.PDESystem,
        b::Vector{Symbolics.Equation})::ModelingToolkit.PDESystem
    a_eqs = equations(a)
    for (i, a_eq) in enumerate(a_eqs)
        for b_eq in b
            if isequal(a_eq.lhs, b_eq.lhs)
                a_eqs[i] = a_eq.lhs ~ a_eq.rhs + b_eq.rhs
            end
        end
    end
    a
end
