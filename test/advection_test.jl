using EarthSciMLBase
using DomainSets, MethodOfLines, ModelingToolkit, DifferentialEquations

@testset "Composed System" begin
    @parameters t, x

    struct ExampleSys <: EarthSciMLODESystem
        sys::ODESystem

        function ExampleSys(t; name)
            @variables y(t)
            @parameters p = 1.0
            D = Differential(t)
            new(ODESystem([D(y) ~ p], t; name))
        end
    end

    @named sys1 = ExampleSys(t)
    @named sys2 = ExampleSys(t)
    domain = DomainInfo(constIC(0.0, t ∈ Interval(0, 1.0)), constBC(1.0, x ∈ Interval(0, 1.0)))

    combined = operator_compose(sys1, sys2)
    combined_pde = combined + domain + ConstantWind(t, 1.0) + Advection()
    combined_mtk = get_mtk(combined_pde)

    @test length(equations(combined_mtk)) == 5
    @test length(combined_mtk.ivs) == 2
    @test length(combined_mtk.dvs) == 6
    @test length(combined_mtk.bcs) == 3

    discretization = MOLFiniteDifference([x => 6], t, approx_order=2)
    prob = discretize(combined_mtk, discretization)
    sol = solve(prob, Tsit5(), saveat=0.1)
    @test sol.retcode == :Success
end