using EarthSciMLBase
using DomainSets, MethodOfLines, ModelingToolkit, DifferentialEquations
import SciMLBase
using Unitful

@testset "Composed System" begin
    @parameters t [unit = u"s"]
    @parameters x [unit = u"m"]

    function ExampleSys()
        @variables y(t) [unit = u"kg"]
        @parameters p = 1.0 [unit = u"kg/s"]
        D = Differential(t)
        ODESystem([D(y) ~ p], t; name=:Docs₊examplesys)
    end

    function ExampleSysCopy()
        @variables y(t) [unit = u"kg"]
        @parameters p = 1.0 [unit = u"kg/s"]
        D = Differential(t)
        ODESystem([D(y) ~ p], t; name=:Docs₊examplesyscopy)
    end

    sys1 = ExampleSys()
    sys2 = ExampleSysCopy()
    domain = DomainInfo(constIC(0.0, t ∈ Interval(0, 1.0)), constBC(1.0, x ∈ Interval(0, 1.0)))

    register_coupling(sys1, sys2) do s1, s2
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)
    combined_pde = couple(combined, domain, ConstantWind(t, 1.0u"m/s"), Advection())
    combined_mtk = get_mtk(combined_pde)

    @test length(equations(combined_mtk)) == 6
    @test length(combined_mtk.ivs) == 2
    @test length(combined_mtk.dvs) == 6
    @test length(combined_mtk.bcs) == 3

    @test occursin("- EarthSciMLBase₊MeanWind₊v_x(t, x)*Differential(x)(Docs₊examplesys₊y(t, x))", 
        string(equations(combined_mtk))) || 
        occursin("- Differential(x)(Docs₊examplesys₊y(t, x))*EarthSciMLBase₊MeanWind₊v_x(t, x)", 
            string(equations(combined_mtk)))

    @test_broken begin # Test fails because PDEs don't currently work with units.
        discretization = MOLFiniteDifference([x => 6], t, approx_order=2)
        prob = discretize(combined_mtk, discretization)
        sol = solve(prob, Tsit5(), saveat=0.1)
        sol.retcode == SciMLBase.ReturnCode.Success
    end
end