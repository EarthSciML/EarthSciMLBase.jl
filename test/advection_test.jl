using EarthSciMLBase
using DomainSets, MethodOfLines, ModelingToolkit, DifferentialEquations
import SciMLBase
using Unitful

@testset "Composed System" begin
    @parameters t [unit = u"s"]
    @parameters x [unit = u"m"]

    struct ExampleSys <: EarthSciMLODESystem
        sys::ODESystem

        function ExampleSys(t; name)
            @variables y(t) [unit = u"kg"]
            @parameters p = 1.0 [unit = u"kg/s"]
            D = Differential(t)
            new(ODESystem([D(y) ~ p], t; name))
        end
    end

    struct ExampleSysCopy <: EarthSciMLODESystem
        sys::ODESystem

        function ExampleSysCopy(t; name)
            @variables y(t) [unit = u"kg"]
            @parameters p = 1.0 [unit = u"kg/s"]
            D = Differential(t)
            new(ODESystem([D(y) ~ p], t; name))
        end
    end

    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSysCopy(t)
    domain = DomainInfo(constIC(0.0, t ∈ Interval(0, 1.0)), constBC(1.0, x ∈ Interval(0, 1.0)))

    EarthSciMLBase.couple(sys1::ExampleSys, sys2::ExampleSysCopy) = operator_compose(sys1, sys2)

    combined = sys1 + sys2
    combined_pde = combined + domain + ConstantWind(t, 1.0u"m/s") + Advection()
    combined_mtk = get_mtk(combined_pde)

    @test length(equations(combined_mtk)) == 6
    @test length(combined_mtk.ivs) == 2
    @test length(combined_mtk.dvs) == 6
    @test length(combined_mtk.bcs) == 3

    @test_broken begin # Test fails because PDEs don't currently work with units.
        discretization = MOLFiniteDifference([x => 6], t, approx_order=2)
        prob = discretize(combined_mtk, discretization)
        sol = solve(prob, Tsit5(), saveat=0.1)
        sol.retcode == SciMLBase.ReturnCode.Success
    end
end