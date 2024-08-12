using Main.EarthSciMLBase
using ModelingToolkit
using Catalyst
using Test
using Unitful

@parameters t

struct ExampleSysCoupler
    sys
end
function ExampleSys()
    @variables x(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(x) ~ p], t; name=:sys1,
        metadata=Dict(:coupletype => ExampleSysCoupler))
end

struct ExampleSysCopyCoupler
    sys
end
function ExampleSysCopy()
    @variables x(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(x) ~ p], t; name=:syscopy,
        metadata=Dict(:coupletype => ExampleSysCopyCoupler))
end

struct ExampleSys2Coupler
    sys
end
function ExampleSys2(; name=:sys2)
    @variables y(t)
    @parameters p
    D = Differential(t)
    ODESystem([D(y) ~ p], t; name=name,
        metadata=Dict(:coupletype => ExampleSys2Coupler))
end

@testset "basic" begin
    sys1 = ExampleSys()
    sys2 = ExampleSysCopy()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysCopyCoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)

    eqstr = replace(string(eq), "Symbolics." => "")
    # The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
    @test eqstr == "Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊syscopy_ddt_xˍt(t)]"
end

@testset "translated" begin
    sys1 = ExampleSys()
    sys2 = ExampleSys2()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSys2Coupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, Dict(s1.x => s2.y))
    end

    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    eq = equations(op)
    eqstr = replace(string(eq), "Symbolics." => "")
    @test eqstr == "Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_ddt_yˍt(t)]"
end

@testset "Non-ODE" begin
    struct ExampleSysNonODECoupler
        sys
    end
    function ExampleSysNonODE()
        @variables y(t)
        @parameters p
        ODESystem([y ~ p], t; name=:sysnonode,
            metadata=Dict(:coupletype => ExampleSysNonODECoupler))
    end

    sys1 = ExampleSys()
    sys2 = ExampleSysNonODE()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysNonODECoupler)
        s1, s2 = s1.sys, s2.sys
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

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSys2Coupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, Dict(s1.x => s2.y => 6.0))
    end

    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys22_ddt_yˍt(t)", streq)
end

@testset "Units" begin
    @parameters t [unit = u"s"]
    struct U1Coupler
        sys
    end
    function U1()
        @variables x(t) [unit = u"kg"]
        @parameters p [unit = u"kg/s"]
        D = Differential(t)
        ODESystem([D(x) ~ p], t; name=:sys1,
            metadata=Dict(:coupletype => U1Coupler))
    end
    struct U2Coupler
        sys
    end
    function U2(; name=:sys2)
        @variables y(t) [unit = u"m"]
        @parameters p [unit = u"m/s"]
        D = Differential(t)
        ODESystem([D(y) ~ p], t; name=name,
            metadata=Dict(:coupletype => U2Coupler))
    end
    
    
    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        @constants uconv = 6.0 [unit=u"kg/m"]
        operator_compose(s1, s2, Dict(s1.x => s2.y => uconv))
    end

    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys2_ddt_yˍt(t)", streq)
end

@testset "Units Non-ODE" begin
    @parameters t [unit = u"s"]
    struct U1Coupler
        sys
    end
    function U1()
        @variables x(t) [unit = u"kg"]
        @parameters p [unit = u"kg/s"]
        D = Differential(t)
        ODESystem([D(x) ~ p], t; name=:sys1,
            metadata=Dict(:coupletype => U1Coupler))
    end
    struct U2Coupler
        sys
    end
    function U2(; name=:sys2)
        @variables y(t) [unit = u"m/s"]
        @parameters p [unit = u"m/s"]
        ODESystem([y ~ p], t; name=name,
            metadata=Dict(:coupletype => U2Coupler))
    end
    
    
    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        @constants uconv = 6.0 [unit=u"kg/m"]
        operator_compose(s1, s2, Dict(s1.x => s2.y => uconv))
    end

    combined = couple(sys1, sys2)

    ox = get_mtk(combined)
    op = structural_simplify(ox)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys2_y(t)", streq)
end

@testset "Reaction-Deposition" begin
    struct ChemCoupler
        sys
    end
    function Chem()
        @species SO2(t) O2(t) SO4(t)
        @parameters α β
        rxns = [
            Reaction(α, [SO2, O2], [SO4], [1, 1], [1])
        ]
        rs = ReactionSystem(rxns, t; name=:chem)
        convert(ODESystem, rs; metadata=Dict(:coupletype => ChemCoupler))
    end

    struct DepositionCoupler
        sys
    end
    function Deposition()
        @variables SO2(t)
        @parameters k = 2
        D = Differential(t)

        eqs = [
            D(SO2) ~ -k * SO2
        ]
        ODESystem(eqs, t, [SO2], [k]; name=:deposition,
            metadata=Dict(:coupletype => DepositionCoupler))
    end

    rn = Chem()
    dep = Deposition()

    function EarthSciMLBase.couple2(rn::ChemCoupler, dep::DepositionCoupler)
        r, d = rn.sys, dep.sys
        operator_compose(r, d)
    end

    combined = couple(rn, dep)
    cs = structural_simplify(get_mtk(combined))
    eq = equations(cs)

    eqstr = replace(string(eq), "Symbolics." => "")
    @test eqstr == "Equation[Differential(t)(chem₊SO2(t)) ~ chem₊deposition_ddt_SO2ˍt(t) - chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊O2(t)) ~ -chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊SO4(t)) ~ chem₊α*chem₊O2(t)*chem₊SO2(t)]"
end