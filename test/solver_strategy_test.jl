using EarthSciMLBase
using Test
using ModelingToolkit, DomainSets, OrdinaryDiffEq
using SciMLOperators
using DifferentialEquations
using SciMLBase: DiscreteCallback, ReturnCode

struct ExampleOp <: Operator
    α::Num # Multiplier from ODESystem
end

function EarthSciMLBase.get_scimlop(op::ExampleOp, mtk_sys, domain::DomainInfo, obs_functions, coordinate_transform_functions, u0, p)
    obs_f = obs_functions(op.α)
    grd = EarthSciMLBase.grid(domain)
    function run(du, u, p, t)
        u = reshape(u, size(u0)...)
        du = reshape(du, size(u0)...)
        for ix ∈ 1:size(u, 1)
            for (i, c1) ∈ enumerate(grd[1])
                for (j, c2) ∈ enumerate(grd[2])
                    for (k, c3) ∈ enumerate(grd[3])
                        # Demonstrate coordinate transforms
                        t1 = coordinate_transform_functions[1](t, c1, c2, c3)
                        t2 = coordinate_transform_functions[2](t, c1, c2, c3)
                        t3 = coordinate_transform_functions[3](t, c1, c2, c3)
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
    FunctionOperator(run, u0[:], p=p)
end

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

@parameters y lon = 0.0 lat = 0.0 lev = 1.0 t α = 10.0
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
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing=[0.1, 0.1, 1.0])

eqs = [Dt(u) ~ -α * √abs(v) + lon,
    Dt(v) ~ -α * √abs(u) + lat + lev * 1e-14,
    windspeed ~ lat + lon + lev,
]
sys = ODESystem(eqs, t, name=:Test₊sys)

op = ExampleOp(sys.windspeed)

csys = Main.EarthSciMLBase.EarthSciMLBase.couple(sys, op, domain)

sys_mtk, obs_eqs = convert(ODESystem, csys; simplify=true)
tf_fs = EarthSciMLBase.coord_trans_functions(obs_eqs, domain)

@test 1 / (tf_fs[1](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111319.44444444445
@test 1 / (tf_fs[2](0.0, 0.0, 0.0, 0.0) * 180 / π) ≈ 111320.00000000001
@test tf_fs[3](0.0, 0.0, 0.0, 0.0) == 1.0

obs_fs = EarthSciMLBase.obs_functions(obs_eqs, domain)
@test obs_fs(sys.windspeed)(0.0, 1.0, 3.0, 2.0) == 6.0
@test obs_fs(op.α)(0.0, 1.0, 3.0, 2.0) == 6.0

u = EarthSciMLBase.init_u(sys_mtk, domain)
p = EarthSciMLBase.default_params(sys_mtk)
scimlop = EarthSciMLBase.nonstiff_ops(csys, sys_mtk, obs_eqs, domain, u, p)
du = similar(u)
du .= 0
@views scimlop(du[:], u[:], p, 0.0)

@test sum(abs.(du)) ≈ 26094.203039436292

setp! = EarthSciMLBase.coord_setter(sys_mtk, domain)

grid = EarthSciMLBase.grid(domain)
prob = ODEProblem(structural_simplify(sys), [], (0.0, 1.0), [
    lon => grid[1][1], lat => grid[2][1], lev => grid[3][1]
])
sol1 = solve(prob, Tsit5(); abstol=1e-12, reltol=1e-12)
@test sol1.retcode == ReturnCode.Success
@test sol1.u[end] ≈ [-27.15156429366082, -26.264264199779465]

st = SolverStrangThreads(Tsit5(), 1.0)

IIchunks, integrators = let
    II = CartesianIndices(size(u)[2:4])
    IIchunks = collect(Iterators.partition(II, length(II) ÷ st.threads))
    start, finish = EarthSciMLBase.tspan(domain)
    prob = ODEProblem(sys_mtk, [], (start, finish), [])
    integrators = [init(remake(prob, u0=zeros(length(unknowns(sys_mtk))), p=deepcopy(p)), st.stiffalg, save_on=false,
        save_start=false, save_end=false, initialize_save=false; abstol=1e-12, reltol=1e-12)
                   for _ in 1:length(IIchunks)]
    (IIchunks, integrators)
end

EarthSciMLBase.threaded_ode_step!(setp!, u, IIchunks, integrators, 0.0, 1.0)

@test u[1, 1, 1, 1] ≈ sol1.u[end][1]
@test u[2, 1, 1, 1] ≈ sol1.u[end][2]

@test sum(abs.(u)) ≈ 212733.04492722102

@testset "mtk_func" begin
    ucopy = copy(u)
    f = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy, p)
    uu = EarthSciMLBase.init_u(sys_mtk, domain)
    du = similar(uu)
    prob = ODEProblem(f, uu[:], (0.0, 1.0), p)
    sol = solve(prob, Tsit5())
    uu = reshape(sol.u[end], size(ucopy)...)
    @test uu[:] ≈ ucopy[:] rtol = 0.01
end



prob = ODEProblem(csys, st)
sol = solve(prob, Euler(); dt=1.0, abstol=1e-12, reltol=1e-12)

@test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7 rtol = 1e-3

@testset "Float32" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32, grid_spacing=[0.1, 0.1, 1])

    csys = couple(sys, op, domain)

    prob = ODEProblem(csys, st)
    sol = solve(prob, Euler(); dt=1.0)

    @test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7
end

@testset "No operator" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        dtype=Float32, grid_spacing=[0.1, 0.1, 1])

    csys = couple(sys, domain)

    prob = ODEProblem(csys, st)
    sol = solve(prob, Euler(); dt=1.0, abstol=1e-6, reltol=1e-6)

    @test sum(abs.(sol.u[end])) ≈ 3.8660308f7
end

@testset "SimulatorStrategies" begin
    @testset "Strang Threads" begin
        st = SolverStrangThreads(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt=1.0, abstol=1e-12, reltol=1e-12)
        @test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7 rtol = 1e-3
    end

    @testset "Strang Serial" begin
        st = SolverStrangSerial(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt=1.0, abstol=1e-12, reltol=1e-12)
        @test sum(abs.(sol.u[end])) ≈ 3.77224671877136e7 rtol = 1e-3
    end

    @testset "IMEX" begin
        st = SolverIMEX()
        prob = ODEProblem(csys, st)
        sol = solve(prob, Tsit5())
        @test sum(abs.(sol.u[end])) ≈ 3.3333500929324217e7 rtol = 1e-3 # No Splitting error in this one.

        sol = solve(prob, SplitEuler(), dt=0.001)
        @test sum(abs.(sol.u[end])) ≈ 3.3333500929324217e7 rtol = 1e-3 # No Splitting error in this one.
    end
end

mutable struct cbt
    runcount::Int
end
function EarthSciMLBase.init_callback(c::cbt, sys, sys_mtk, obs_eqs, dom)
    DiscreteCallback((u, t, integrator) -> true,
        (_) -> c.runcount += 1,
    )
end

@testset "callback" begin
    runcount = 0
    af(_) = runcount += 1
    cb = DiscreteCallback(
        (u, t, integrator) -> true,
        af,
    )
    @testset "strang" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, st)
        solve(prob, Euler(); dt=1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
    runcount = 0
    @testset "imex" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, SolverIMEX())
        solve(prob, Tsit5(); dt=1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
end
