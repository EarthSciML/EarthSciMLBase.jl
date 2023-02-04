using EarthSciMLBase
using ModelingToolkit, DomainSets

@testset "varindex" begin
    @parameters lon lat x y t
    @test EarthSciMLBase.varindex([lon, lat, x, y, t], :lat) == 2
end

@testset "lon2meters" begin
    @test EarthSciMLBase.lon2meters(0.0) ≈ 40075000.0 / 360.0
end

@testset "equation" begin
    @parameters lon lat x y t
    @variables c(..)
    Dt = Differential(t)
    pd = partialderivatives_lonlat2xymeters([lon, lat, x, y, t])

    haveeq = Dt(c(t, lon, lat)) ~ pd[1](c(t, lon, lat)) + pd[2](c(t, lon, lat))

    wanteq = Dt(c(t, lon, lat)) ~ Differential(lon)(c(t, lon, lat)) / (40075.0e3 * cos(0.017453292519943295lat) / 360.0) +
                                  Differential(lat)(c(t, lon, lat)) / 111.32e3

    @test isequal(haveeq, wanteq)
end

@testset "system" begin
    @parameters lon lat lev t

    struct Example <: EarthSciMLODESystem
        sys
        function Example(t; name)
            @variables c(t) = 5.0
            D = Differential(t)
            new(ODESystem([D(c) ~ sin(t)], t, name=name))
        end
    end
    @named examplesys = Example(t)

    domain = DomainInfo(
        partialderivatives_lonlat2xymeters,
        constIC(0.0, t ∈ Interval(0.0f0, 3600.0f0)),
        periodicBC(lat ∈ Interval(-90.0f0, 90.0f0)),
        periodicBC(lon ∈ Interval(-180.0f0, 180.0f0)),
        zerogradBC(lev ∈ Interval(1.0f0, 10.0f0)),
    )

    composed_sys = examplesys + domain + Advection()

    sys_mtk = get_mtk(composed_sys)

    have_eq = equations(sys_mtk)
    @assert length(have_eq) == 1
    @variables examplesys₊c(..) meanwind₊u(..) meanwind₊v(..) meanwind₊w(..)
    want_eq = Differential(t)(examplesys₊c(t, lat, lon, lev)) ~
        (-Differential(lon)(examplesys₊c(t, lat, lon, lev)) * meanwind₊v(t, lat, lon, lev)) / (111319.44444444445cos(0.017453292519943295lat)) + sin(t) -
        8.98311174991017e-6Differential(lat)(examplesys₊c(t, lat, lon, lev)) * meanwind₊u(t, lat, lon, lev) -
        Differential(lev)(examplesys₊c(t, lat, lon, lev)) * meanwind₊w(t, lat, lon, lev)
    @test isequal(have_eq[1], want_eq)
end