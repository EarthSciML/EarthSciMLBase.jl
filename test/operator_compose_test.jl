using EarthSciMLBase
using ModelingToolkit
using ModelingToolkit: t, D, t_nounits, D_nounits
using Test
using DynamicQuantities

struct ExampleSysCoupler
    sys::Any
end
function ExampleSys(; name = :sys1)
    @variables x(t_nounits)
    @parameters p
    System([D_nounits(x) ~ p], t_nounits; name = name,
        metadata = Dict(CoupleType => ExampleSysCoupler))
end

struct ExampleSysCopyCoupler
    sys::Any
end
function ExampleSysCopy()
    @variables x(t_nounits)
    @parameters p
    System([D_nounits(x) ~ p], t_nounits; name = :syscopy,
        metadata = Dict(CoupleType => ExampleSysCopyCoupler))
end

struct ExampleSys2Coupler
    sys::Any
end
function ExampleSys2(; name = :sys2)
    @variables y(t_nounits)
    @parameters p
    System([D_nounits(y) ~ p], t_nounits; name = name,
        metadata = Dict(CoupleType => ExampleSys2Coupler))
end

@testset "basic" begin
    sys1 = ExampleSys()
    sys2 = ExampleSysCopy()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysCopyCoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)

    op = convert(System, combined)
    eq = equations(op)

    eqstr = replace(string(eq), "Symbolics." => "")
    # The simplified equation should be D(x) = p + sys2_xˍt, where sys2_xˍt is also equal to p.
    @test eqstr ==
          "Equation[Differential(t, 1)(sys1₊x(t)) ~ sys1₊p + sys1₊syscopy_ddt_xˍt(t)]"
end

@testset "translated" begin
    sys1 = ExampleSys()
    sys2 = ExampleSys2()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSys2Coupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, Dict(s1.x => s2.y))
    end

    combined = couple(sys1, sys2)

    op = convert(System, combined)
    eq = equations(op)
    eqstr = replace(string(eq), "Symbolics." => "")
    @test eqstr == "Equation[Differential(t, 1)(sys1₊x(t)) ~ sys1₊p + sys1₊sys2_ddt_yˍt(t)]"
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
        System([D_nounits(y1) ~ p, D_nounits(y2) ~ p], t_nounits; name = name,
            metadata = Dict(CoupleType => ExampleSysXYCoupler))
    end

    sys2 = ExampleSysXY()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysXYCoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, [s1.x => s2.y1, s1.x => s2.y2 => 2])
    end

    combined = couple(sys1, sys2)

    op = convert(System, combined)
    eq = equations(op)
    obs = observed(op)
    eqstr = replace(string(eq), "Symbolics." => "")
    @test occursin("sys1₊p", eqstr)
    @test occursin("2sys1₊sysXY_ddt_y2ˍt(t)", eqstr)
    @test occursin("sys1₊sysXY_ddt_y1ˍt(t)", eqstr)
end

@testset "Non-ODE" begin
    struct ExampleSysNonODECoupler
        sys::Any
    end
    function ExampleSysNonODE()
        @variables y(t_nounits)
        @parameters p
        System([y ~ p], t; name = :sysnonode,
            metadata = Dict(CoupleType => ExampleSysNonODECoupler))
    end

    sys1 = ExampleSys()
    sys2 = ExampleSysNonODE()

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler, s2::ExampleSysNonODECoupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2, Dict(s1.x => s2.y))
    end

    combined = couple(sys1, sys2)
    sys_combined = convert(System, combined)

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

    op = convert(System, combined)
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
        System([k ~ p], t_nounits; name = name,
            metadata = Dict(CoupleType => KCoupler))
    end
    struct twovarCoupler
        sys::Any
    end
    function twovar(; name = :twovar)
        @variables x(t_nounits) y(t_nounits)
        @parameters p
        System([D_nounits(x) ~ p, D_nounits(y) ~ p], t_nounits; name = name,
            metadata = Dict(CoupleType => twovarCoupler))
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
    op = convert(System, combined)
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
        System([ModelingToolkit.D(x) ~ p], t; name = :sys1,
            metadata = Dict(CoupleType => U1Coupler))
    end
    struct U2Coupler
        sys::Any
    end
    function U2(; name = :sys2)
        @variables y(t) [unit = u"m"]
        @parameters p [unit = u"m/s"]
        System([ModelingToolkit.D(y) ~ p], t; name = name,
            metadata = Dict(CoupleType => U2Coupler))
    end

    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        @constants uconv=6.0 [unit = u"kg/m"]
        operator_compose(s1, s2, Dict(s1.x => s2.y => uconv))
    end

    combined = couple(sys1, sys2)

    op = convert(System, combined)
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
        System([D(x) ~ p], t; name = :sys1,
            metadata = Dict(CoupleType => U1Coupler))
    end
    struct U2Coupler
        sys::Any
    end
    function U2(; name = :sys2)
        @variables y(t) [unit = u"m/s"]
        @parameters p [unit = u"m/s"]
        System([y ~ p], t; name = name,
            metadata = Dict(CoupleType => U2Coupler))
    end

    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        @constants uconv=6.0 [unit = u"kg/m"]
        operator_compose(s1, s2, Dict(s1.x => s2.y => uconv))
    end

    combined = couple(sys1, sys2)

    op = convert(System, combined)
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
        System([D(x) ~ 0], t; name = :sys1,
            metadata = Dict(CoupleType => U1Coupler))
    end
    struct U2Coupler
        sys::Any
    end
    function U2(; name = :sys2)
        @variables x(t) [unit = u"kg*m^-3/s"]
        @parameters p [unit = u"kg*m^-3/s"]
        System([x ~ p], t; name = name,
            metadata = Dict(CoupleType => U2Coupler))
    end

    sys1 = U1()
    sys2 = U2()

    function EarthSciMLBase.couple2(s1::U1Coupler, s2::U2Coupler)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)

    sys = convert(System, combined)
    @test occursin("sys1₊sys2_x(t)", string(equations(sys)) * string(observed(sys)))
end

@testset "Reaction-Deposition" begin
    struct ChemCoupler
        sys::Any
    end
    function Chem()
        @variables SO2(t_nounits) O2(t_nounits) SO4(t_nounits)
        @parameters α β
        eqs = [
            D_nounits(SO2) ~ -α * SO2 * O2,
            D_nounits(O2) ~ -α * SO2 * O2,
            D_nounits(SO4) ~ α * SO2 * O2
        ]
        System(eqs, t_nounits; name = :chem,
            metadata = Dict(CoupleType => ChemCoupler))
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
        System(eqs, t_nounits, [SO2], [k]; name = :deposition,
            metadata = Dict(CoupleType => DepositionCoupler))
    end

    rn = Chem()
    dep = Deposition()

    function EarthSciMLBase.couple2(rn::ChemCoupler, dep::DepositionCoupler)
        r, d = rn.sys, dep.sys
        operator_compose(r, d)
    end

    combined = couple(rn, dep)
    cs = convert(System, combined)
    eq = equations(cs)

    eqstr = replace(string(eq), "Symbolics." => "")
    @test occursin("chem₊deposition_ddt_SO2ˍt(t)", eqstr)
end

@testset "events" begin
    @parameters α=1 [unit = u"kg", description = "α description"]
    @parameters β=2 [unit = u"kg*s", description = "β description"]
    @variables x(t) [unit = u"m", description = "x description"]
    @constants onex = 1 [unit = u"m", description = "unit x"]
    eq = D(x) ~ α * x / β
    @named sys1 = System([eq], t;
        continuous_events = [x ~ 0],
        discrete_events = (t == 1.0) => [x ~ x + onex]
    )
    @named sys2 = System([eq], t;
        continuous_events = [(x ~ 1.0) => [x ~ Pre(x) + onex],
            (x ~ 2.0) => [x ~ Pre(x) - onex]],
        discrete_events = [(t == 1.0) => [x ~ Pre(x) + onex],
            (t == 2.0) => [x ~ Pre(x) - onex]]
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
        sys1 = System([k ~ a0], [k], [a0]; name = :rate)

        function rate()
            return sys1.k
        end
        @variables begin
            A(t) = 20
            B(t) = 0
        end
        eqs = [D_nounits(A) ~ -rate() * A,
            D_nounits(B) ~ rate() * A]
        System(eqs, t_nounits; name = name,
            metadata = Dict(CoupleType => ChemistryCoupler))
    end
    sys1 = Chemistry()

    struct EmisCoupler
        sys::Any
    end
    function Emis()
        @variables A(t_nounits)
        @parameters p
        System([D_nounits(A) ~ 2p], t_nounits; name = :Emis,
            metadata = Dict(CoupleType => EmisCoupler))
    end

    sys2 = Emis()

    result = operator_compose(sys1, sys2)
    @test occursin("Emis_ddt_Aˍt", string(equations(result.to)))
    @test occursin("Emis_ddt_Aˍt", string(equations(result.from)))
end
