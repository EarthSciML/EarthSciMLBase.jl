using EarthSciMLBase
using ModelingToolkit
using ModelingToolkit: t, D, t_nounits, D_nounits
using Catalyst
using Test
using DynamicQuantities

struct ExampleSysCoupler
    sys::Any
end
function ExampleSys(; name = :sys1)
    @variables x(t_nounits)
    @parameters p
    ODESystem([D_nounits(x) ~ p], t_nounits; name = name,
        metadata = Dict(:coupletype => ExampleSysCoupler))
end

struct ExampleSysCopyCoupler
    sys::Any
end
function ExampleSysCopy()
    @variables x(t_nounits)
    @parameters p
    ODESystem([D_nounits(x) ~ p], t_nounits; name = :syscopy,
        metadata = Dict(:coupletype => ExampleSysCopyCoupler))
end

struct ExampleSys2Coupler
    sys::Any
end
function ExampleSys2(; name = :sys2)
    @variables y(t_nounits)
    @parameters p
    ODESystem([D_nounits(y) ~ p], t_nounits; name = name,
        metadata = Dict(:coupletype => ExampleSys2Coupler))
end

@testset "basic" begin
    sys1 = ExampleSys()
    sys2 = ExampleSysCopy()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysCopyCoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)

    op = convert(ODESystem, combined)
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

    op = convert(ODESystem, combined)
    eq = equations(op)
    eqstr = replace(string(eq), "Symbolics." => "")
    @test eqstr == "Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_ddt_yˍt(t)]"
end

@testset "translate" begin
    @test EarthSciMLBase.normalize_translate(Dict(:a => :b, :c => :d => 2)) == [
        (:a => :b, 1),
        (:c => :d, 2)
    ]
    @test EarthSciMLBase.normalize_translate([:a => :b, :a => :c => 2]) == [
        (:a => :b, 1),
        (:a => :c, 2)
    ]
    @test EarthSciMLBase.get_matching_translate(
        [(:a => :b, 1), (:b => :b, 1), (:a => :c, 2)], :a) == [(:a => :b, 1), (:a => :c, 2)]
end

@testset "multiple" begin
    sys1 = ExampleSys()

    struct ExampleSysXYCoupler
        sys::Any
    end
    function ExampleSysXY(; name = :sysXY)
        @variables y1(t_nounits)
        @variables y2(t_nounits)
        @parameters p
        ODESystem([D_nounits(y1) ~ p, D_nounits(y2) ~ p], t_nounits; name = name,
            metadata = Dict(:coupletype => ExampleSysXYCoupler))
    end

    sys2 = ExampleSysXY()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysXYCoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, [s1.x => s2.y1, s1.x => s2.y2 => 2])
    end

    combined = couple(sys1, sys2)

    op = convert(ODESystem, combined)
    eq = equations(op)
    obs = observed(op)
    eqstr = replace(string(eq), "Symbolics." => "")
    @test eqstr ==
          "Equation[Differential(t)(sys1₊x(t)) ~ sys1₊p + 2sys1₊sysXY_ddt_y2ˍt(t) + sys1₊sysXY_ddt_y1ˍt(t)]"
end

@testset "Non-ODE" begin
    struct ExampleSysNonODECoupler
        sys::Any
    end
    function ExampleSysNonODE()
        @variables y(t_nounits)
        @parameters p
        ODESystem([y ~ p], t; name = :sysnonode,
            metadata = Dict(:coupletype => ExampleSysNonODECoupler))
    end

    sys1 = ExampleSys()
    sys2 = ExampleSysNonODE()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysNonODECoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, Dict(s1.x => s2.y))
    end

    combined = couple(sys1, sys2)
    sys_combined = convert(ODESystem, combined)

    streq = string(equations(sys_combined))
    @test occursin("sys1₊sysnonode_y(t)", streq)
    @test occursin("sys1₊p", streq)
end

@testset "translated with conversion factor" begin
    sys1 = ExampleSys()
    sys2 = ExampleSys2(; name = :sys22)

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSys2Coupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, Dict(s1.x => s2.y => 6.0))
    end

    combined = couple(sys1, sys2)

    op = convert(ODESystem, combined)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys22_ddt_yˍt(t)", streq)
end

@testset "translated with multiple conversion factors" begin
    struct KCoupler
        sys::Any
    end
    function KSys(; name = :ksys)
        @variables k(t_nounits)
        @parameters p
        ODESystem([k ~ p], t_nounits; name = name,
            metadata = Dict(:coupletype => KCoupler))
    end
    struct twovarCoupler
        sys::Any
    end
    function twovar(; name = :twovar)
        @variables x(t_nounits) y(t_nounits)
        @parameters p
        ODESystem([D_nounits(x) ~ p, D_nounits(y) ~ p], t_nounits; name = name,
            metadata = Dict(:coupletype => twovarCoupler))
    end

    sys1 = twovar()
    sys2 = KSys()

    function EarthSciMLBase.couple2(s1::twovarCoupler, s2::KCoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, Dict(
            s1.x => s2.k => s1.x,
            s1.y => s2.k => s1.y
        ))
    end

    combined = couple(sys1, sys2)
    op = convert(ODESystem, combined)
    streq = string(equations(op))
    @test occursin("twovar₊p", streq)
    @test occursin("twovar₊ksys_k(t)", streq)
    @test occursin("twovar₊ksys_k_1(t)", streq)
end

@testset "Units" begin
    struct U1Coupler
        sys::Any
    end
    function U1()
        @variables x(t) [unit = u"kg"]
        @parameters p [unit = u"kg/s"]
        ODESystem([ModelingToolkit.D(x) ~ p], t; name = :sys1,
            metadata = Dict(:coupletype => U1Coupler))
    end
    struct U2Coupler
        sys::Any
    end
    function U2(; name = :sys2)
        @variables y(t) [unit = u"m"]
        @parameters p [unit = u"m/s"]
        ODESystem([ModelingToolkit.D(y) ~ p], t; name = name,
            metadata = Dict(:coupletype => U2Coupler))
    end

    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        @constants uconv=6.0 [unit = u"kg/m"]
        operator_compose(s1, s2, Dict(s1.x => s2.y => uconv))
    end

    combined = couple(sys1, sys2)

    op = convert(ODESystem, combined)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys2_ddt_yˍt(t)", streq)
end

@testset "Units Non-ODE" begin
    struct U1Coupler
        sys::Any
    end
    function U1()
        @variables x(t) [unit = u"kg"]
        @parameters p [unit = u"kg/s"]
        ODESystem([D(x) ~ p], t; name = :sys1,
            metadata = Dict(:coupletype => U1Coupler))
    end
    struct U2Coupler
        sys::Any
    end
    function U2(; name = :sys2)
        @variables y(t) [unit = u"m/s"]
        @parameters p [unit = u"m/s"]
        ODESystem([y ~ p], t; name = name,
            metadata = Dict(:coupletype => U2Coupler))
    end

    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        @constants uconv=6.0 [unit = u"kg/m"]
        operator_compose(s1, s2, Dict(s1.x => s2.y => uconv))
    end

    combined = couple(sys1, sys2)

    op = convert(ODESystem, combined)
    streq = string(equations(op))
    @test occursin("sys1₊p", streq)
    @test occursin("sys1₊sys2_y(t)", streq)
end

@testset "Units 2" begin
    struct U1Coupler
        sys::Any
    end
    function U1()
        @variables x(t) [unit = u"kg*m^-3"]
        ODESystem([D(x) ~ 0], t; name = :sys1,
            metadata = Dict(:coupletype => U1Coupler))
    end
    struct U2Coupler
        sys::Any
    end
    function U2(; name = :sys2)
        @variables x(t) [unit = u"kg*m^-3/s"]
        @parameters p [unit = u"kg*m^-3/s"]
        ODESystem([x ~ p], t; name = name,
            metadata = Dict(:coupletype => U2Coupler))
    end

    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)

    sys = convert(ODESystem, combined)
    @test occursin("sys1₊sys2_x(t)", string(equations(sys)))
end

@testset "Reaction-Deposition" begin
    struct ChemCoupler
        sys::Any
    end
    function Chem()
        @species SO2(t_nounits) O2(t_nounits) SO4(t_nounits)
        @parameters α β
        rxns = [
            Reaction(α, [SO2, O2], [SO4], [1, 1], [1])
        ]
        rs = complete(ReactionSystem(rxns, t_nounits; name = :chem))
        convert(ODESystem, rs; metadata = Dict(:coupletype => ChemCoupler))
    end

    struct DepositionCoupler
        sys::Any
    end
    function Deposition()
        @variables SO2(t_nounits)
        @parameters k = 2

        eqs = [
            D_nounits(SO2) ~ -k * SO2
        ]
        ODESystem(eqs, t_nounits, [SO2], [k]; name = :deposition,
            metadata = Dict(:coupletype => DepositionCoupler))
    end

    rn = Chem()
    dep = Deposition()

    function EarthSciMLBase.couple2(rn::ChemCoupler, dep::DepositionCoupler)
        r, d = rn.sys, dep.sys
        operator_compose(r, d)
    end

    combined = couple(rn, dep)
    cs = convert(ODESystem, combined)
    eq = equations(cs)

    eqstr = replace(string(eq), "Symbolics." => "")
    @test eqstr ==
          "Equation[Differential(t)(chem₊SO2(t)) ~ chem₊deposition_ddt_SO2ˍt(t) - chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊O2(t)) ~ -chem₊α*chem₊O2(t)*chem₊SO2(t), Differential(t)(chem₊SO4(t)) ~ chem₊α*chem₊O2(t)*chem₊SO2(t)]"
end

@testset "events" begin
    @parameters α=1 [unit = u"kg", description = "α description"]
    @parameters β=2 [unit = u"kg*s", description = "β description"]
    @variables x(t) [unit = u"m", description = "x description"]
    eq = D(x) ~ α * x / β
    @named sys1 = ODESystem([eq], t; metadata = :metatest,
        continuous_events = [x ~ 0],
        discrete_events = (t == 1.0) => [x ~ x + 1]
    )
    @named sys2 = ODESystem([eq], t; metadata = :metatest,
        continuous_events = [(x ~ 1.0) => [x ~ x + 1], (x ~ 2.0) => [x ~ x - 1]],
        discrete_events = [(t == 1.0) => [x ~ x + 1], (t == 2.0) => [x ~ x - 1]]
    )
    sys3 = operator_compose(sys1, sys2)
    @test length(ModelingToolkit.get_continuous_events(sys3.from)) == 1
    @test length(ModelingToolkit.get_discrete_events(sys3.from)) == 1
    @test length(ModelingToolkit.get_continuous_events(sys3.to)) == 2
    @test length(ModelingToolkit.get_discrete_events(sys3.to)) == 2
end

# https://github.com/EarthSciML/EarthSciMLBase.jl/issues/76
@testset "NonlinearSystem" begin
    struct ChemistryCoupler
        sys::Any
    end
    function Chemistry(; name = :chemistry)
        @variables k(t_nounits)
        @parameters a0 = 1
        sys1 = NonlinearSystem([k ~ a0], [k], [a0]; name = :rate)

        function rate()
            return sys1.k
        end
        rx_sys = @reaction_network rx begin
            @species(A(t)=20,
                B(t)=0,)
            rate(), A --> B
        end
        rxns = compose(rx_sys, sys1)
        convert(ODESystem, complete(rxns); combinatoric_ratelaws = false, name = name,
            metadata = Dict(:coupletype => ChemistryCoupler))
    end
    sys1 = Chemistry()

    struct EmisCoupler
        sys::Any
    end
    function Emis()
        @variables A(t_nounits)
        @parameters p
        ODESystem([D_nounits(A) ~ 2p], t_nounits; name = :Emis,
            metadata = Dict(:coupletype => EmisCoupler))
    end

    sys2 = Emis()

    result = operator_compose(sys1, sys2)
    @test occursin("Emis_ddt_Aˍt", string(equations(result.to)))
    @test occursin("Emis_ddt_Aˍt", string(equations(result.from)))
end
