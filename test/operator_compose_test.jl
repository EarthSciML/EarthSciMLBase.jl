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

ox = get_mtk(combined)
equations(ox)
states(ox)
op = structural_simplify(ox)
eq = equations(op)
states(op)
observed(op)

b = IOBuffer()
show(b, eq)
# The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
@test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys2₊sys2_xˍt(t)]"
