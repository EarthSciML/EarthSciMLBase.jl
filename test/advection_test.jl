using EarthSciMLBase
using Test
using DomainSets, ModelingToolkit
using OrdinaryDiffEqTsit5
using ModelingToolkit: t, D
import SciMLBase
using DynamicQuantities
using Dates, DomainSets

@testset "Composed System" begin
    @parameters x [unit = u"m"]
    @constants czero=0 [unit = u"kg/s/m"]

    struct ExampleSysCoupler2
        sys::Any
    end
    function ExampleSys()
        @variables y(t) [unit = u"kg"]
        @parameters p=1.0 [unit = u"kg/s"]
        System([D(y) ~ p + x * czero], t, [y], [x, p]; name = :examplesys,
            metadata = Dict(CoupleType => ExampleSysCoupler2))
    end

    struct ExampleSysCopyCoupler2
        sys::Any
    end
    function ExampleSysCopy()
        @variables y(t) [unit = u"kg"]
        @parameters p=1.0 [unit = u"kg/s"]
        System([D(y) ~ p], t; name = :examplesyscopy,
            metadata = Dict(CoupleType => ExampleSysCopyCoupler2))
    end

    sys1 = ExampleSys()
    sys2 = ExampleSysCopy()
    domain = DomainInfo(
        constIC(0.0, t ∈ Interval(0, 1.0)), constBC(1.0, x ∈ Interval(0, 1.0)))

    function EarthSciMLBase.couple2(s1::ExampleSysCoupler2, s2::ExampleSysCopyCoupler2)
        s1, s2 = s1.sys, s2.sys
        operator_compose(s1, s2)
    end

    combined = couple(sys1, sys2)
    combined_pde = couple(combined, domain, ConstantWind(t, 1.0u"m/s"), Advection())
    # ConstantWind + Advection coupling with PDESystem is not yet working with MTK v11
end

@testset "Coordinate transform" begin
    @parameters lon [unit = u"rad"]
    @parameters lat [unit = u"rad"]
    @constants c_unit=180 / π / 6 [unit = u"rad" description = "constant to make units cancel out"]
    wind = ConstantWind(t, 1.0u"m/s", 2.0u"m/s")

    function Example(t)
        @variables c(t)=5.0 [unit = u"mol/m^3"]
        System([D(c) ~ (sin(lat / c_unit) + sin(lon / c_unit)) * c / t],
            t, name = :Test₊ExampleSys)
    end
    examplesys = Example(t)

    deg2rad(x) = x * π / 180.0
    domain = DomainInfo(
        Function[
        partialderivatives_δxyδlonlat,
    ],
        constIC(0.0,
            t ∈ Interval(Dates.datetime2unix(DateTime(2022, 1, 1)),
                Dates.datetime2unix(DateTime(2022, 1, 3)))),
        zerogradBC(lat ∈ Interval(deg2rad(-85.0f0), deg2rad(85.0f0))),
        periodicBC(lon ∈ Interval(deg2rad(-180.0f0), deg2rad(175.0f0)))
    )

    composed_sys = couple(examplesys, domain, Advection(), wind)
    pde_sys = convert(PDESystem, composed_sys)

    eqs = equations(pde_sys)

    want_terms = [
        "MeanWind₊v_lat(t, lat, lon)", "ConstantWind₊v_1(t, lat, lon)",
        "MeanWind₊v_lon(t, lat, lon)", "ConstantWind₊v_2(t, lat, lon)",
        "Differential(t, 1)(Test₊ExampleSys₊c(t, lat, lon))", "lat2meters"
    ]
    have_eqs = string.(eqs)
    for term in want_terms
        @test any(occursin.((term,), have_eqs))
    end
end
