using EarthSciMLBase
using Test
using ModelingToolkit, DomainSets, OrdinaryDiffEq
using SciMLOperators
using DifferentialEquations
using SciMLBase: DiscreteCallback

struct ExampleOp <: Operator
    α::Num # Multiplier from ODESystem
end

function EarthSciMLBase.get_scimlop(op::ExampleOp, s::Simulator)
    obs_f = s.obs_fs[s.obs_fs_idx[op.α]]
    function run(du, u, p, t)
        u = reshape(u, size(s.u)...)
        du = reshape(du, size(s.u)...)
        for ix ∈ 1:size(s.u, 1)
            for (i, c1) ∈ enumerate(s.grid[1])
                for (j, c2) ∈ enumerate(s.grid[2])
                    for (k, c3) ∈ enumerate(s.grid[3])
                        # Demonstrate coordinate transforms
                        t1 = s.tf_fs[1](t, c1, c2, c3)
                        t2 = s.tf_fs[2](t, c1, c2, c3)
                        t3 = s.tf_fs[3](t, c1, c2, c3)
                        # Demonstrate calculating observed value.
                        fv = obs_f(t, c1, c2, c3)
                        # Set derivative value.
                        du[ix, i, j, k] = (t1 + t2 + t3) * fv
                    end
                end
            end
        end
        nothing
    end
    FunctionOperator(run, s.u[:], p=s.p)
end

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0
@constants p = 1.0
@variables(
    u(t) = 1.0, v(t) = 1.0, x(t) = 1.0, y(t) = 1.0, windspeed(t) = 1.0
)
Dt = Differential(t)

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...))

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lat + lon + lev,
]
sys = ODESystem(eqs, t, name=:Test₊sys)

op = ExampleOp(sys.windspeed)

csys = couple(sys, op, domain)

sim = Simulator(csys, [0.1, 0.1, 1])
st = SimulatorStrangThreads(Tsit5(), Euler(), 1.0)

@test 1 / (sim.tf_fs[1](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111319.44444444445
@test 1 / (sim.tf_fs[2](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111320.00000000001
@test sim.tf_fs[3](0.0, 0.0, 0.0, 0.0) == 1.0

@test sim.obs_fs[sim.obs_fs_idx[sys.windspeed]](0.0, 1.0, 3.0, 2.0) == 6.0
@test sim.obs_fs[sim.obs_fs_idx[op.α]](0.0, 1.0, 3.0, 2.0) == 6.0

scimlop = EarthSciMLBase.get_scimlop(op, sim)
du = similar(sim.u)
du .= 0
@views scimlop(du[:], sim.u[:], sim.p, 0.0)

@test sum(abs.(du)) ≈ 26094.203039436292

prob = ODEProblem(structural_simplify(sys), [], (0.0, 1.0), [
    lon => sim.grid[1][1], lat => sim.grid[2][1], lev => sim.grid[3][1]
])
sol1 = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
@test sol1.u[end] ≈ [-27.15156429366082, -26.264264199779465]

EarthSciMLBase.init_u!(sim)

IIchunks, integrators = let
    II = CartesianIndices(size(sim.u)[2:4])
    IIchunks = collect(Iterators.partition(II, length(II) ÷ st.threads))
    start, finish = EarthSciMLBase.time_range(sim.domaininfo)
    prob = ODEProblem(sim.sys_mtk, [], (start, finish), [])
    integrators = [init(remake(prob, u0=similar(sim.u_init), p=deepcopy(sim.p)), st.stiffalg, save_on=false,
        save_start=false, save_end=false, initialize_save=false; abstol=1e-12, reltol=1e-12)
                   for _ in 1:length(IIchunks)]
    (IIchunks, integrators)
end

EarthSciMLBase.threaded_ode_step!(sim, IIchunks, integrators, 0.0, 1.0)

@test sim.u[1, 1, 1, 1] ≈ sol1.u[end][1]
@test sim.u[2, 1, 1, 1] ≈ sol1.u[end][2]

@test sum(abs.(sim.u)) ≈ 212733.04492722102

@testset "mtk_func" begin
    ucopy = copy(sim.u)
    f = EarthSciMLBase.mtk_func(sim)
    EarthSciMLBase.init_u!(sim)
    du = similar(sim.u)
    prob = ODEProblem(f, sim.u[:], (0.0, 1.0), sim.p)
    sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES(), autodiff=false))
    uu = reshape(sol.u[end], size(ucopy)...)
    @test uu[:] ≈ ucopy[:] rtol = 0.01
end

run!(sim, st; abstol=1e-12, reltol=1e-12)

@test sum(abs.(sim.u)) ≈ 3.77224671877136e7

@testset "Float32" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32)

    csys = couple(sys, op, domain)

    sim = Simulator(csys, [0.1, 0.1, 1])

    run!(sim, st)

    @test sum(abs.(sim.u)) ≈ 3.77224671877136e7
end

@testset "No operator" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32)

    csys = couple(sys, domain)

    sim = Simulator(csys, [0.1, 0.1, 1])

    run!(sim, st; abstol=1e-6, reltol=1e-6)

    @test sum(abs.(sim.u)) ≈ 3.8660308f7
end

@testset "SimulatorStrategies" begin
    st = SimulatorStrangThreads(Tsit5(), Euler(), 1.0)
    run!(sim, st; abstol=1e-12, reltol=1e-12)
    @test sum(abs.(sim.u)) ≈ 3.77224671877136e7

    st = SimulatorStrangSerial(Tsit5(), Euler(), 1.0)
    run!(sim, st; abstol=1e-12, reltol=1e-12)
    @test sum(abs.(sim.u)) ≈ 3.77224671877136e7

    st = SimulatorIMEX(KenCarp47(linsolve=KrylovJL_GMRES(), autodiff=false))
    @test_broken run!(sim, st)
end

@testset "callback" begin
    runcount = 0
    af(integrator) = runcount += 1
    cb = DiscreteCallback(
        (u, t, integrator) -> true,
        af,
    )
    csys2 = couple(csys, cb)
    sim = Simulator(csys2, [0.1, 0.1, 1])
    run!(sim, st)
    @test runcount > 0
end