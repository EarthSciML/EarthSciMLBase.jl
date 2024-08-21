using EarthSciMLBase
using ModelingToolkit, DomainSets
using ModelingToolkit: t, D
using DynamicQuantities

@testset "varindex" begin
    @parameters lon lat x y
    @test EarthSciMLBase.varindex([lon, lat, x, y, t], :lat) == 2
end

@testset "lon2meters" begin
    x = substitute(EarthSciMLBase.lon2meters(0.0), Dict(EarthSciMLBase.lon2m => 40075.0e3 / 360.0))
    @test x ≈ 40075000.0 / 360.0
end

@testset "δxyδlonlat" begin
    @parameters lon [unit = u"rad"]
    @parameters lat [unit = u"rad"]
    @parameters x [unit = u"m"]
    @parameters y [unit = u"m"]
    pd = partialderivatives_δxyδlonlat([lon, x, lat, y, t])
    @test isequal(pd, Dict(3 => 1.0 / EarthSciMLBase.lat2meters, 1 => 1.0 / (EarthSciMLBase.lon2m * cos(lat))))
end

@testset "system" begin
    @parameters lon [unit = u"rad"]
    @parameters lat [unit = u"rad"]
    @parameters lev [unit = u"m"]

    function Example()
        @variables c(t) = 5.0 [unit = u"kg"]
        @constants t_c = 1.0 [unit = u"s"] # constant to make `sin` unitless
        @constants c_c = 1.0 [unit = u"kg/s"] # constant to make equation units work out
        ODESystem([D(c) ~ sin(t / t_c) * c_c], t, name=:examplesys)
    end
    examplesys = Example()

    deg2rad(x) = x * π / 180.0
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(0.0, t ∈ Interval(0.0f0, 3600.0f0)),
        periodicBC(lat ∈ Interval(deg2rad(-90.0f0), deg2rad(90.0f0))),
        periodicBC(lon ∈ Interval(deg2rad(-180.0f0), deg2rad(180.0f0))),
        zerogradBC(lev ∈ Interval(1.0f0, 10.0f0)),
    )

    composed_sys = couple(examplesys, domain, Advection())

    sys_mtk = convert(PDESystem, composed_sys)

    have_eq = equations(sys_mtk)
    @assert length(have_eq) == 1
    @variables examplesys₊c(..) MeanWind₊v_lon(..) MeanWind₊v_lat(..) MeanWind₊v_lev(..)
    @constants examplesys₊t_c = 1.0 examplesys₊c_c = 1.0
    want_eq = Differential(t)(examplesys₊c(t, lat, lon, lev)) ~ examplesys₊c_c * sin(t / examplesys₊t_c) +
                                                                (-MeanWind₊v_lat(t, lat, lon, lev) * Differential(lat)(examplesys₊c(t, lat, lon, lev))) / EarthSciMLBase.lat2meters +
                                                                (-MeanWind₊v_lon(t, lat, lon, lev) * Differential(lon)(examplesys₊c(t, lat, lon, lev))) / (EarthSciMLBase.lon2m * cos(lat)) -
                                                                MeanWind₊v_lev(t, lat, lon, lev) * Differential(lev)(examplesys₊c(t, lat, lon, lev))
    @test isequal(have_eq[1], want_eq)
end

@testset "DomainInfo" begin
    @parameters lon [unit = u"rad"]
    @parameters lat [unit = u"rad"]
    @parameters lev [unit = u"m"]

    deg2rad(x) = x * π / 180.0
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(0.0, t ∈ Interval(0.0f0, 3600.0f0)),
        periodicBC(lat ∈ Interval(deg2rad(-90.0f0), deg2rad(90.0f0))),
        periodicBC(lon ∈ Interval(deg2rad(-180.0f0), deg2rad(180.0f0))),
        zerogradBC(lev ∈ Interval(1.0f0, 10.0f0)),
    )

    @variables u

    δs = partialderivatives(domain)

    have = [δs[i](u) for i ∈ eachindex(δs)]

    want = [
        Differential(lat)(u) / EarthSciMLBase.lat2meters,
        Differential(lon)(u) / (EarthSciMLBase.lon2m * cos(lat)),
        Differential(lev)(u),
    ]

    isequal(have, want)
end