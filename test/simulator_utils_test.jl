using Main.EarthSciMLBase: steplength, observed_expression, observed_function, utype, grid, timesteps, icbc
using Main.EarthSciMLBase
using Test
using ModelingToolkit, DomainSets
using ModelingToolkit: t_nounits; t=t_nounits
using ModelingToolkit: D_nounits; D=D_nounits

@test steplength([0, 0.1, 0.2]) == 0.1

@parameters x y lon = 0.0 lat = 0.0 lev = 1.0 α = 10.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0
)

eqs = [D(u) ~ -α * √abs(v) + lon,
    D(v) ~ -α * √abs(u) + lat,
    x ~ 2α + p + y,
    y ~ 4α - 2p
]

@named sys = ODESystem(eqs, t)
sys = structural_simplify(sys)

xx = observed_expression(observed(sys), x)

@test isequal(xx, -1.0 + 6α)

coords = [α]
xf = observed_function(observed(sys), x, coords)

@test isequal(xf(α), -1.0 + 6α)
@test isequal(xf(2), -1.0 + 6 * 2)


@variables uu, vv
extra_eqs = [uu ~ x + 3, vv ~ uu * 4]

xx2 = observed_expression([observed(sys); extra_eqs], vv)

xf2 = observed_function([observed(sys); extra_eqs], vv, coords)

@test isequal(xf2(α), 4 * (2 + 6α))

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...))

vars = unknowns(sys)

bcs = icbc(domain, vars)

@test utype(domain) == Float64
@test utype(DomainInfo(constIC(0, t ∈ Interval(0, 1)), constBC(16.0, lon ∈ Interval(0.0, 1.0)), dtype=Float32)) == Float32

@test grid(domain, [0.1π, 0.01π]) ≈ [-π:0.1π:π, -0.45π:0.01π:0.45π]

@test timesteps(0:0.1:1, 0:0.15:1) == [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

@test timesteps(0:0.1:0.3, 0:0.1000000000001:0.3) == [0.0, 0.1, 0.2, 0.3]
