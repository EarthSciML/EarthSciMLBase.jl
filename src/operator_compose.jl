export operator_compose

"""
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
$(TYPEDSIGNATURES)

Compose to systems of equations together by adding the right-hand side terms together of equations that have matching left-hand sides.
The left hand sides of two equations will be considered matching if:

    1. They are both time derivatives of the same variable.
    2. There is an entry in the optional `translate` dictionary that maps the dependent variable in the first system to the dependent variable in the second system, e.g. `Dict(sys1.sys.x => sys2.sys.y)`.
    3. There is an entry in the optional `translate` dictionary that maps the dependent variable in the first system to the dependent variable in the second system, with a conversion factor, e.g. `Dict(sys1.sys.x => sys2.sys.y => 6)`.

The example below shows that when we `operator_compose` two systems together that are both equal to `D(x) = p`, the resulting system is equal to `D(x) = 2p`.

# Example with matching variables
``` jldoctest
using EarthSciMLBase
using ModelingToolkit

@parameters t

struct ExampleSys <: EarthSciMLODESystem
    sys::ODESystem

    function ExampleSys(t; name)
        @variables x(t)
        @parameters p
        D = Differential(t)
        new(ODESystem([D(x) ~ p], t; name))
    end
end

@named sys1 = ExampleSys(t)
@named sys2 = ExampleSys(t)

combined = operator_compose(sys1, sys2)

combined_mtk = get_mtk(combined)

# The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
equations(structural_simplify(combined_mtk))

# output
1-element Vector{Equation}:
 Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_xˍt(t)
```

This second example demonstrates the second case above, where one variable in the first system is equal to another variable in the second system:

# Example with non-matching variables
``` jldoctest
using EarthSciMLBase
using ModelingToolkit

@parameters t

struct ExampleSys <: EarthSciMLODESystem
    sys::ODESystem
    function ExampleSys(t; name)
        @variables x(t)
        @parameters p
        D = Differential(t)
        new(ODESystem([D(x) ~ p], t; name))
    end
end

struct ExampleSys2 <: EarthSciMLODESystem
    sys::ODESystem
    function ExampleSys2(t; name)
        @variables y(t)
        @parameters p
        D = Differential(t)
        new(ODESystem([D(y) ~ p], t; name))
    end
end

@named sys1 = ExampleSys(t)
@named sys2 = ExampleSys2(t)

combined = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y))
equations(structural_simplify(get_mtk(combined)))

# output
1-element Vector{Equation}:
 Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_yˍt(t)
```

Finally, this last example shows the third case, where a conversion factor is included in the translation dictionary.

# Example with non-matching variables and a conversion factor
``` jldoctest
using EarthSciMLBase
using ModelingToolkit

@parameters t

struct ExampleSys <: EarthSciMLODESystem
    sys::ODESystem
    function ExampleSys(t; name)
        @variables x(t)
        @parameters p
        D = Differential(t)
        new(ODESystem([D(x) ~ p], t; name))
    end
end

struct ExampleSys2 <: EarthSciMLODESystem
    sys::ODESystem
    function ExampleSys2(t; name)
        @variables y(t)
        @parameters p
        D = Differential(t)
        new(ODESystem([D(y) ~ p], t; name))
    end
end

@named sys1 = ExampleSys(t)
@named sys2 = ExampleSys2(t)

combined = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y => 6))
equations(structural_simplify(get_mtk(combined)))

# output
1-element Vector{Equation}:
 Differential(t)(sys1₊x(t)) ~ sys1₊p + 6sys1₊sys2_yˍt(t)
```
"""
function operator_compose(a::EarthSciMLODESystem, b::EarthSciMLODESystem, translate=Dict())
    a_eqs = equations(a.sys)
    b_eqs = equations(b.sys)
    iv = ModelingToolkit.get_iv(a.sys) # independent variable
    aname = String(nameof(a.sys))
    bname = String(nameof(b.sys))
    connections = Equation[]
    for (i, a_eq) ∈ enumerate(a_eqs)
        adv = add_scope(a.sys, arguments(a_eq.lhs)[1], iv) # dependent variable
        if adv ∉ keys(translate) # If adv is not in the translation dictionary, then assume it is the same in both systems.
            bdv, conv = add_scope(b.sys, arguments(a_eq.lhs)[1], iv), 1
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
            if isequal(bdv, add_scope(b.sys, arguments(b_eq.lhs)[1], iv)) && operation(b_eq.lhs) == Differential(iv) # dependent variables are equal
                # Create a new variable to connect the two systems.
                bvar = String(Symbolics.tosymbol(b_eq.lhs, escape=false))
                var1 = Symbol("$(bname)_$(bvar)")
                term1 = (@variables $var1(iv))[1]
                term1 = add_metadata(term1, b_eq.lhs)
                a_eqs[i] = a_eq.lhs ~ a_eq.rhs + term1 * conv
                b_eqs[j] = term1 ~ b_eq.rhs
                var2 = Symbol("$(aname)₊", var1)
                term2 = (@variables $var2(iv))[1]
                term2 = add_metadata(term2, b_eq.lhs)
                var3 = Symbol("$(bname)₊", var1)
                term3 = (@variables $var3(iv))[1]
                term3 = add_metadata(term3, b_eq.lhs)
                push!(connections, term2 ~ term3)

                # Now set the dependent variables in the two systems to be equal.
                push!(connections, adv ~ bdv * conv)
            end
        end
    end

    ComposedEarthSciMLSystem(
        ConnectorSystem(connections, a, b),
        a, b,
    )
end

# PDESystems don't have a compose function, so we just add the equations together
# here without trying to keep the systems in separate namespaces.
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
