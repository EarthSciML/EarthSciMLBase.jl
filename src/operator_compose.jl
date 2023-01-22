export operator_compose

"""
$(TYPEDSIGNATURES)

Compose to systems of equations together by adding the right-hand side terms together of equations that have matching left-hand sides.

The example below shows that when we `operator_compose` two systems together that are both equal to `D(x) = p`, the resulting system is equal to `D(x) = 2p`.

# Example
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
equations(structural_simplify(combined_mtk))

# The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
# output
1-element Vector{Equation}:
 Differential(t)(sys1₊x(t)) ~ sys1₊p + sys2₊sys2_xˍt(t)
```
"""
function operator_compose(a::EarthSciMLODESystem, b::EarthSciMLODESystem)
    a_eqs = equations(a.sys)
    b_eqs = equations(b.sys)
    connections = Equation[]
    for (i, a_eq) ∈ enumerate(a_eqs)
        for (j, b_eq) ∈ enumerate(b_eqs)
            if isequal(a_eq.lhs, b_eq.lhs)
                # Create a new variable to connect the two systems.
                bname = String(nameof(b.sys))
                aname = String(nameof(a.sys))
                bvar = String(Symbolics.tosymbol(b_eq.lhs, escape=false))
                var1 = Symbol("$(bname)_$(bvar)")
                term1 = (@variables $var1(ModelingToolkit.get_iv(a.sys)))[1]
                a_eqs[i] = a_eq.lhs ~ a_eq.rhs + term1
                b_eqs[j] = term1 ~ b_eq.rhs
                var2 = Symbol("$(aname)₊$(bname)_$(bvar)")
                term2 = (@variables $var2(ModelingToolkit.get_iv(a.sys)))[1]
                var3 = Symbol("$(bname)₊$(bname)_$(bvar)")
                term3 = (@variables $var3(ModelingToolkit.get_iv(a.sys)))[1]
                push!(connections, term2 ~ term3)
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
