function operator_compose(a::EarthSciMLSystem, b::EarthSciMLSystem)
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
                term1 = (@variables $var1(a.sys.iv))[1]
                a_eqs[i] = a_eq.lhs ~ a_eq.rhs + term1
                b_eqs[j] = term1 ~ b_eq.rhs
                var2 = Symbol("$(aname)₊$(bname)_$(bvar)")
                term2 = (@variables $var2(a.sys.iv))[1]
                var3 = Symbol("$(bname)₊$(bname)_$(bvar)")
                term3 = (@variables $var3(a.sys.iv))[1]
                push!(connections, term2 ~ term3)
            end
        end
    end

    ComposedEarthSciMLSystem(
        ConnectorSystem(connections, a, b),
        a, b,
    )
end

@parameters t

struct ExampleSys <: EarthSciMLSystem
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

ox = get_mtk(combined)
equations(ox)
states(ox)
op = structural_simplify(ox)
equations(op)
states(op)
observed(op)