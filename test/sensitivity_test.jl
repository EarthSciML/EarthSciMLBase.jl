using EarthSciMLBase
using SciMLSensitivity, ForwardDiff, Zygote
using ModelingToolkit, DomainSets, OrdinaryDiffEq, SymbolicIndexingInterface
using SciMLOperators
using LinearSolve
using Test

struct ExampleOp <: Operator
end

function EarthSciMLBase.get_scimlop(op::ExampleOp, csys::CoupledSystem, mtk_sys, domain::DomainInfo, u0, p)
    α, trans1, trans2, trans3 = EarthSciMLBase.get_needed_vars(op, csys, mtk_sys, domain)

    obs_f = ModelingToolkit.build_explicit_observed_function(mtk_sys,
        [α, trans1, trans2, trans3], checkbounds=false, return_inplace=false)

    setp! = EarthSciMLBase.coord_setter(mtk_sys, domain)
    obscache = zeros(EarthSciMLBase.dtype(domain), 4)
    sz = length.(EarthSciMLBase.grid(domain))

    function run(du, u, p, t) # In-place
        u = reshape(u, :, sz...)
        du = reshape(du, :, sz...)
        II = CartesianIndices(size(u)[2:end])
        for ix ∈ 1:size(u, 1)
            for I in II
                # Demonstrate coordinate transforms and observed values
                setp!(p, I)
                t1, t2, t3, fv = obs_f(view(u, :, I), p, t)
                # Set derivative value.
                du[ix, I] = (t1 + t2 + t3) * fv
            end
        end
        nothing
    end
    function run(u, p, t) # Out-of-place
        u = reshape(u, :, sz...)
        II = CartesianIndices(size(u)[2:end])
        du = vcat([
            begin
                setp!(p, I)
                t1, t2, t3, fv = obs_f(view(u, :, I), p, t)
                (t1 + t2 + t3) * fv
            end for ix ∈ 1:size(u, 1), I in II
        ]...)
        reshape(du, :)
    end
    FunctionOperator(run, reshape(u0, :), p=p)
end

function EarthSciMLBase.get_needed_vars(::ExampleOp, csys, mtk_sys, domain::DomainInfo)
    ts = EarthSciMLBase.partialderivative_transform_vars(mtk_sys, domain)
    return [mtk_sys.sys₊windspeed, ts...]
end

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0 β = 1.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0, windspeed(t)
)
Dt = Differential(t)

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing=[1.0, 1.0, 1.0])

eqs = [Dt(u) ~ -α * √abs(v) + lon + β,
    Dt(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lat + lon + lev,
]
sys = ODESystem(eqs, t, name=:sys)

op = ExampleOp()

csys = EarthSciMLBase.couple(sys, op, domain)
model_sys = convert(ODESystem, csys)

st = SolverIMEX(stiff_sparse=false)
prob = ODEProblem{false}(csys, st)

function loss(p)
    new_params = remake_buffer(model_sys, prob.p, Dict(
        model_sys.sys₊α .=> p[1], model_sys.sys₊β .=> p[2]))
    newprob = remake(prob, p=new_params)
    sol = solve(newprob, KenCarp47(linsolve=LUFactorization()))
    sum(abs.(sol))
end

g = ForwardDiff.gradient(loss, [10.0, 1.0])
@test g ≈ [134064.58606757477, -2410.9599794591354]

@test_broken g = Zygote.gradient(loss, [10.0, 1.0])

prob = ODEProblem{true}(csys, st)
@test_broken ForwardDiff.gradient(loss, [10.0, 1.0])
