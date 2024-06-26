using EarthSciMLBase
using DomainSets, MethodOfLines, ModelingToolkit, DifferentialEquations
import SciMLBase
using Unitful
using Dates, DomainSets

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

@testset "Coordinate transform" begin
    @parameters t [unit = u"s"]
    @parameters lon [unit = u"rad"]
    @parameters lat [unit = u"rad"]
    @constants c_unit = 180 / π / 6 [unit = u"rad" description = "constant to make units cancel out"]
    wind = ConstantWind(t, 1.0u"m/s", 2.0u"m/s")

    function Example(t)
        @variables c(t) = 5.0 [unit = u"mol/m^3"]
        D = Differential(t)
        ODESystem([D(c) ~ (sin(lat / c_unit) + sin(lon / c_unit)) * c / t], t, name=:Test₊ExampleSys)
    end
    examplesys = Example(t)

    deg2rad(x) = x * π / 180.0
    domain = DomainInfo(
        Function[
            partialderivatives_δxyδlonlat,
        ],
        constIC(0.0, t ∈ Interval(Dates.datetime2unix(DateTime(2022, 1, 1)), Dates.datetime2unix(DateTime(2022, 1, 3)))),
        zerogradBC(lat ∈ Interval(deg2rad(-85.0f0), deg2rad(85.0f0))),
        periodicBC(lon ∈ Interval(deg2rad(-180.0f0), deg2rad(175.0f0))),
    )

    composed_sys = couple(examplesys, domain, Advection(), wind)
    pde_sys = get_mtk(composed_sys)

    eqs = equations(pde_sys)

    want_terms = [
        "EarthSciMLBase₊MeanWind₊v_lat(t, lat, lon)", "EarthSciMLBase₊ConstantWind₊v_1(t, lat, lon)",
        "EarthSciMLBase₊MeanWind₊v_lon(t, lat, lon)", "EarthSciMLBase₊ConstantWind₊v_2(t, lat, lon)",
        "Differential(t)(Test₊ExampleSys₊c(t, lat, lon))", "lat2meters",
    ]
    have_eqs = string.(eqs)
    for term ∈ want_terms
        @test any(occursin.((term,), have_eqs))
    end
end