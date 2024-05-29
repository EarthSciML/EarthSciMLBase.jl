using EarthSciMLBase
using ModelingToolkit, DomainSets
using Unitful

@testset "varindex" begin
    @parameters lon lat x y t
    @test EarthSciMLBase.varindex([lon, lat, x, y, t], :lat) == 2
end

@testset "lon2meters" begin
    x = substitute(EarthSciMLBase.lon2meters(0.0), Dict(EarthSciMLBase.lon2m => 40075.0e3 / 360.0))
    @test x ≈ 40075000.0 / 360.0
end

@testset "equation" begin
    @parameters lon [unit=u"rad"]
    @parameters lat [unit=u"rad"]
    @parameters x [unit=u"m"]
    @parameters y [unit=u"m"]
    @parameters t [unit=u"s"]
    @variables c(..)
    Dt = Differential(t)
    pd = partialderivatives_lonlat2xymeters([lon, lat, x, y, t])

    haveeq = Dt(c(t, lon, lat)) ~ pd[1](c(t, lon, lat)) + pd[2](c(t, lon, lat))

    wanteq = Differential(t)(c(t, lon, lat)) ~ Differential(lat)(c(t, lon, lat)) / EarthSciMLBase.lat2meters + 
            Differential(lon)(c(t, lon, lat)) / (EarthSciMLBase.lon2m*cos(lat))

    @test isequal(haveeq, wanteq)
end

@testset "system" begin
    @parameters lon [unit=u"rad"]
    @parameters lat [unit=u"rad"]
    @parameters lev [unit=u"m"]
    @parameters t [unit=u"s"]

    function Example()
        @variables c(t) = 5.0 [unit=u"kg"]
        @constants t_c = 1.0 [unit=u"s"] # constant to make `sin` unitless
        @constants c_c = 1.0 [unit=u"kg/s"] # constant to make equation units work out
        D = Differential(t)
        ODESystem([D(c) ~ sin(t/t_c)*c_c], t, name=:examplesys)
    end
    examplesys = Example()

    deg2rad(x) = x * π / 180.0
    domain = DomainInfo(
        partialderivatives_lonlat2xymeters,
        constIC(0.0, t ∈ Interval(0.0f0, 3600.0f0)),
        periodicBC(lat ∈ Interval(deg2rad(-90.0f0), deg2rad(90.0f0))),
        periodicBC(lon ∈ Interval(deg2rad(-180.0f0), deg2rad(180.0f0))),
        zerogradBC(lev ∈ Interval(1.0f0, 10.0f0)),
    )

    composed_sys = couple(examplesys, domain, Advection())

    sys_mtk = get_mtk(composed_sys)

    have_eq = equations(sys_mtk)
    @assert length(have_eq) == 1
    @variables examplesys₊c(..) meanwind₊v_lon(..) meanwind₊v_lat(..) meanwind₊v_lev(..)
    @constants examplesys₊t_c=1.0 examplesys₊c_c=1.0
    want_eq = Differential(t)(examplesys₊c(t, lat, lon, lev)) ~ examplesys₊c_c*sin(t / examplesys₊t_c) + 
        (-meanwind₊v_lat(t, lat, lon, lev)*Differential(lat)(examplesys₊c(t, lat, lon, lev))) / EarthSciMLBase.lat2meters + 
        (-meanwind₊v_lon(t, lat, lon, lev)*Differential(lon)(examplesys₊c(t, lat, lon, lev))) / (EarthSciMLBase.lon2m*cos(lat)) - 
        meanwind₊v_lev(t, lat, lon, lev)*Differential(lev)(examplesys₊c(t, lat, lon, lev))
    @test isequal(have_eq[1], want_eq)
end