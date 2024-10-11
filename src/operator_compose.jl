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
    vstr = String(Symbolics.tosymbol(v, escape=false))
    vsym = Symbol("$(n)₊$(vstr)")
    vv = (@variables $vsym(iv))[1]
    add_metadata(vv, v)
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
function operator_compose(a::ModelingToolkit.ODESystem, b::ModelingToolkit.ODESystem, translate=Dict())
    a_eqs = deepcopy(equations(a))
    b_eqs = deepcopy(equations(b))
    iv = ModelingToolkit.get_iv(a) # independent variable
    aname = String(nameof(a))
    bname = String(nameof(b))
    connections = Equation[]
    for (i, a_eq) ∈ enumerate(a_eqs)
        adv = add_scope(a, get_dv(a_eq.lhs, iv), iv) # dependent variable
        if adv ∉ keys(translate) # If adv is not in the translation dictionary, then assume it is the same in both systems.
            bdv, conv = add_scope(b, get_dv(a_eq.lhs, iv), iv), 1
        else
            tt = translate[adv]
            if length(tt) == 1 # Handle the optional inclusion of a conversion factor.
                bdv, conv = tt, 1
            elseif length(tt) == 2
                bdv, conv = tt
            else
                error("Invalid transformation for dependent variable $adv: $tt")
            end
        end
        for (j, b_eq) ∈ enumerate(b_eqs)
            if isequal(bdv, add_scope(b, get_dv(b_eq.lhs, iv), iv))
                bdv = add_scope(b, get_dv(b_eq.lhs, iv), iv) # Make sure the units are correct.
                # The dependent variable of the LHS of this equation matches the dependent
                # variable of interest,
                # so create a new variable to represent the dependent variable
                # of interest and add it to the RHS of the other equation, and then also set the two
                # dependent variables to be equal.
                bvar = String(Symbolics.tosymbol(b_eq.lhs, escape=false))
                if operation(b_eq.lhs) == Differential(iv)
                    # The LHS of this equation is the time derivative of the dependent variable of interest,
                    var1 = Symbol("$(bname)_ddt_$(bvar)")
                    term1 = (@variables $var1(iv))[1]
                    term1 = add_metadata(term1, b_eq.lhs)
                    b_eqs[j] = term1 ~ b_eq.rhs

                    var2 = Symbol("$(aname)₊", var1)
                    term2 = (@variables $var2(iv))[1]
                    term2 = add_metadata(term2, b_eq.lhs)
                    var3 = Symbol("$(bname)₊", var1)
                    term3 = (@variables $var3(iv))[1]
                    term3 = add_metadata(term3, b_eq.lhs)
                    push!(connections, term2 ~ term3)
                    a_eqs[i] = a_eq.lhs ~ a_eq.rhs + term1 * conv
                    # Now set the dependent variables in the two systems to be equal.
                    push!(connections, adv ~ bdv * conv)
                else # The LHS of this equation is the dependent variable of interest.
                    var1 = Symbol("$(bname)_$(bvar)")
                    term1 = (@variables $var1(iv))[1]
                    term1 = add_metadata(term1, b_eq.lhs * conv)
                    var2 = Symbol("$(aname)₊", var1)
                    term2 = (@variables $var2(iv))[1]
                    term2 = add_metadata(term2, b_eq.lhs * conv)
                    a_eqs[i] = a_eq.lhs ~ a_eq.rhs + term1
                    push!(connections, term2 ~ bdv * conv)
                end
            end
        end
    end
    aa = copy_with_change(a; eqs=a_eqs)
    bb = copy_with_change(b; eqs=b_eqs)
    ConnectorSystem(connections, aa, bb)
end

# PDESystems don't have a compose function, so we just add the equations together
# here without trying to keep the systems in separate namespaces.
# TODO(CT): Handle events
function operator_compose!(a::ModelingToolkit.PDESystem, b::Vector{Symbolics.Equation})::ModelingToolkit.PDESystem
    a_eqs = equations(a)
    for (i, a_eq) ∈ enumerate(a_eqs)
        for b_eq ∈ b
            if isequal(a_eq.lhs, b_eq.lhs)
                a_eqs[i] = a_eq.lhs ~ a_eq.rhs + b_eq.rhs
            end
        end
    end
    a
end
