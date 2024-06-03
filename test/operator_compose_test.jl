using EarthSciMLBase
using ModelingToolkit
using Catalyst
using Test

@parameters t
function ExampleSys()
    @variables x(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(x) ~ p], t; name=:sys1)
end

function ExampleSysCopy()
    @variables x(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(x) ~ p], t; name=:syscopy)
end

function ExampleSys2(; name=:sys2)
    @variables y(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(y) ~ p], t; name=name)
end

@testset "basic" begin
    sys1 = ExampleSys()
    sys2 = ExampleSysCopy()

    register_coupling(sys1, sys2) do s1, s2
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)

    b = IOBuffer()
    show(b, eq)
    # The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊syscopy_ddt_xˍt(t)]"
end

@testset "translated" begin
    sys1 = ExampleSys()
    sys2 = ExampleSys2()

    register_coupling(sys1, sys2) do s1, s2
        operator_compose(s1, s2, Dict(s1.x => s2.y))
    end

    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)

    b = IOBuffer()
    show(b, eq)
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_ddt_yˍt(t)]"
end

@testset "Non-ODE" begin
    function ExampleSysNonODE()
        @variables y(t)
        @parameters p
        ODESystem([y ~ p], t; name=:sysnonode)
    end

    sys1 = ExampleSys()
    sys2 = ExampleSysNonODE()

    register_coupling(sys1, sys2) do s1, s2
        operator_compose(s1, s2, Dict(s1.x => s2.y))
    end

    combined = couple(sys1, sys2)
    combined_mtk = get_mtk(combined)
    sys_combined = structural_simplify(combined_mtk)

    streq = string(equations(sys_combined))
    @test occursin("sys1₊sysnonode_y(t)", streq)
    @test occursin("sys1₊p", streq)
end

@testset "translated with conversion factor" begin
    sys1 = ExampleSys()
    sys2 = ExampleSys2(; name=:sys22)

    register_coupling(sys1, sys2) do s1, s2
        operator_compose(s1, s2, Dict(s1.x => s2.y => 6.0))
    end
    
    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys22_ddt_yˍt(t)", streq)
end

@testset "Reaction-Deposition" begin
    function Chem()
        @species SO2(t) O2(t) SO4(t)
        @parameters α β
        rxns = [
            Reaction(α, [SO2, O2], [SO4], [1, 1], [1])
        ]
        ReactionSystem(rxns, t; name=:chem)
    end

    function Deposition()
        @variables SO2(t)
        @parameters k = 2
        D = Differential(t)

        eqs = [
            D(SO2) ~ -k * SO2
        ]
        ODESystem(eqs, t, [SO2], [k]; name=:deposition)
    end

    rn = Chem()
    dep = Deposition()

    register_coupling(rn, dep) do r, d
        r = convert(ODESystem, r)
        operator_compose(r, d)
    end

    combined = couple(rn, dep)
    cs = structural_simplify(get_mtk(combined))
    eq = equations(cs)

    b = IOBuffer()
    show(b, eq)
    @test String(take!(b)) == "Symbolics.Equation[Differential(t)(chem₊SO2(t)) ~ chem₊deposition_ddt_SO2ˍt(t) - chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊O2(t)) ~ -chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊SO4(t)) ~ chem₊α*chem₊O2(t)*chem₊SO2(t)]"
end