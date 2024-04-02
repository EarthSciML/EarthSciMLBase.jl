using EarthSciMLBase
using ModelingToolkit
using Catalyst


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

@testset "basic" begin
    @parameters t

    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSys(t)

    combined = operator_compose(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)

    b = IOBuffer()
    show(b, eq)
    # The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_xˍt(t)]"
end

@testset "translated" begin
    @parameters t

    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSys2(t)

    combined = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y))

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)

    b = IOBuffer()
    show(b, eq)
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_yˍt(t)]"
end

@testset "Non-ODE" begin
    @parameters t
    struct ExampleSysNonODE <: EarthSciMLODESystem
        sys::ODESystem
        function ExampleSysNonODE(t; name)
            @variables y(t)
            @parameters p
            new(ODESystem([y ~ p], t; name))
        end
    end

    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSysNonODE(t)

    combined = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y))
    sys_combined = structural_simplify(get_mtk(combined))

    @unpack x, p = sys1.sys
    wanteq = Differential(t)(x) ~ p + sys2.sys.y
    @test isequal(wanteq, only(equations(sys_combined)))
end

@testset "translated with conversion factor" begin
    @parameters t
    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSys2(t)

    combined = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y => 6.0))

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    b = IOBuffer()
    eq = equations(op)
    show(b, eq)
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + 6.0sys1₊sys2_yˍt(t)]"
end

@testset "Reaction-Deposition" begin
    struct Chem <: EarthSciMLODESystem
        sys::ODESystem
        rxn_sys::ReactionSystem
        function Chem(t)
            @species SO2(t) O2(t) SO4(t)
            @parameters α β
            rxns = [
                Reaction(α, [SO2, O2], [SO4], [1, 1], [1])
            ]
            rxn_sys = ReactionSystem(rxns, t; name=:chem)
            new(convert(ODESystem, rxn_sys; combinatoric_ratelaws=false), rxn_sys)
        end
    end

    struct Deposition <: EarthSciMLODESystem
        sys::ODESystem
        function Deposition(t)
            @variables SO2(t)
            @parameters k = 2
            D = Differential(t)

            eqs = [
                D(SO2) ~ -k * SO2
            ]
            new(ODESystem(eqs, t, [SO2], [k]; name=:deposition))
        end
    end
    @variables t
    rn = Chem(t)
    dep = Deposition(t)

    combined = operator_compose(rn, dep)
    cs = structural_simplify(get_mtk(combined))
    eq = equations(cs)

    b = IOBuffer()
    show(b, eq)
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(chem₊SO2(t)) ~ chem₊deposition_SO2ˍt(t) - chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊O2(t)) ~ -chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊SO4(t)) ~ chem₊α*chem₊O2(t)*chem₊SO2(t)]"
end