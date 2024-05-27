using EarthSciMLBase
using ModelingToolkit
using Catalyst
using Test

struct ExampleSys <: EarthSciMLODESystem
    sys::ODESystem

    function ExampleSys(t; name)
        @variables x(t)
        @parameters p
        D = Differential(t)
        new(ODESystem([D(x) ~ p], t; name))
    end
end

struct ExampleSysCopy <: EarthSciMLODESystem
    sys::ODESystem

    function ExampleSysCopy(t; name)
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
    @named sys2 = ExampleSysCopy(t)

    EarthSciMLBase.couple(sys1::ExampleSys, sys2::ExampleSysCopy) = operator_compose(sys1, sys2)

    combined = sys1 + sys2

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)

    b = IOBuffer()
    show(b, eq)
    # The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_ddt_xˍt(t)]"
end

@testset "translated" begin
    @parameters t

    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSys2(t)

    EarthSciMLBase.couple(sys1::ExampleSys, sys2::ExampleSys2) = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y))
    combined = sys1 + sys2

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)

    b = IOBuffer()
    show(b, eq)
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_ddt_yˍt(t)]"
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

    EarthSciMLBase.couple(sys1::ExampleSys, sys2::ExampleSysNonODE) = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y))
    combined = sys1 + sys2
    combined_mtk = get_mtk(combined)
    sys_combined = structural_simplify(combined_mtk)

    streq = string(equations(sys_combined))
    @test occursin("sys1₊sys2_y(t)", streq)
    @test occursin("sys1₊p", streq)
end

@testset "translated with conversion factor" begin
    @parameters t
    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSys2(t)

    EarthSciMLBase.couple(sys1::ExampleSys, sys2::ExampleSys2) = operator_compose(sys1, sys2, Dict(sys1.sys.x => sys2.sys.y => 6.0))
    combined = sys1 + sys2

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys2_ddt_yˍt(t)", streq)
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

    EarthSciMLBase.couple(rn::Chem, dep::Deposition) = operator_compose(rn, dep)
    combined = rn + dep
    cs = structural_simplify(get_mtk(combined))
    eq = equations(cs)

    b = IOBuffer()
    show(b, eq)
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(chem₊SO2(t)) ~ chem₊deposition_ddt_SO2ˍt(t) - chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊O2(t)) ~ -chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊SO4(t)) ~ chem₊α*chem₊O2(t)*chem₊SO2(t)]"
end