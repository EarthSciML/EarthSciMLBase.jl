using EarthSciMLBase
using Test
using ModelingToolkit, DomainSets
using OrdinaryDiffEqTsit5, OrdinaryDiffEqSDIRK, OrdinaryDiffEqLowOrderRK
using DynamicQuantities
using SciMLBase: DiscreteCallback, ReturnCode
using LinearSolve
t = ModelingToolkit.t_nounits
D = ModelingToolkit.D_nounits

struct ExampleOp <: Operator
end

function EarthSciMLBase.get_odefunction(
        op::ExampleOp, csys::CoupledSystem, mtk_sys, coord_args,
        domain::DomainInfo{ET, AT}, u0, p, alg::MapAlgorithm) where {ET, AT}
    α, trans1, trans2, trans3 = EarthSciMLBase.get_needed_vars(op, csys, mtk_sys, domain)

    obs_f = EarthSciMLBase.build_coord_observed_function(mtk_sys, coord_args,
        [α, trans1, trans2, trans3])

    c1, c2, c3 = EarthSciMLBase.concrete_grid(domain)
    obscache = similar(domain.u_proto, 4)

    nrows = length(unknowns(mtk_sys))
    sz = tuple(size(domain)...)
    II = CartesianIndices(sz)

    function run(du, u, p, t) # In-place
        u = reshape(u, nrows, sz...)
        du = reshape(du, nrows, sz...)
        function f(j, du, u, p, t, c1, c2, c3)
            # Demonstrate coordinate transforms and observed values
            obs_f(obscache, view(u, :, II[j]), p, t, c1[j], c2[j], c3[j])
            t1, t2, t3, fv = obscache
            # Set derivative value.
            view(du, :, II[j]) .= (t1 + t2 + t3) * fv
            return nothing
        end
        EarthSciMLBase.map_closure_to_range(f, 1:((*)(sz...)), alg, du, u, p, t, c1, c2, c3)
        return reshape(du, :)
    end
    function run(u, p, t) # Out-of-place
        u = reshape(u, :, sz...)
        function f(j, u, p, t, c1, c2, c3)
            t1, t2, t3, fv = obs_f(view(u, :, II[j]), p, t, c1[j], c2[j], c3[j])
            [(t1 + t2 + t3) * fv for _ in 1:size(u, 1)]
        end
        du = EarthSciMLBase.map_closure_to_range(f, 1:((*)(sz...)),
            alg, u, p, t, c1, c2, c3)
        reshape(hcat(du...), :)
    end
    return run
end

function EarthSciMLBase.get_needed_vars(::ExampleOp, csys, mtk_sys, domain::DomainInfo)
    return [mtk_sys.sys₊windspeed, mtk_sys.sys₊x, mtk_sys.sys₊y, mtk_sys.sys₊z]
end

t_min = 0.0
lon_min, lon_max = -π, π
lat_min, lat_max = -0.45π, 0.45π
t_max = 11.5

@parameters y lon=0.0 lat=0.0 lev=1.0 α=10.0
@constants p = 1.0
@variables(u(t)=1.0, v(t)=1.0, x(t), [unit=u"1/m"], y(t), [unit=u"1/m"], z(t),
    windspeed(t))

indepdomain = t ∈ Interval(t_min, t_max)

partialdomains = [lon ∈ Interval(lon_min, lon_max),
    lat ∈ Interval(lat_min, lat_max),
    lev ∈ Interval(1, 3)]

domain = DomainInfo(
    partialderivatives_δxyδlonlat,
    constIC(16.0, indepdomain), constBC(16.0, partialdomains...); grid_spacing = [
        0.1, 0.1, 1.0])

eqs = [D(u) ~ -α * √abs(v) + lon,
    D(v) ~ -α * √abs(u) + lat + lev * 1.0f-14,
    windspeed ~ lat + lon + lev,
    x ~ 1 / EarthSciMLBase.lon2meters(lat),
    y ~ 1 / EarthSciMLBase.lat2meters,
    z ~ 1 / lev
]
sys = System(eqs, t, name = :sys)

op = ExampleOp()

csys = EarthSciMLBase.couple(sys, op, domain)

sys_mtk = convert(System, csys)

@test Symbol.(EarthSciMLBase.partialderivative_transform_vars(sys_mtk, domain)) == [
    Symbol("δsys₊lon_transform(t)"),
    Symbol("δsys₊lat_transform(t)"),
    Symbol("δsys₊lev_transform(t)")]

sys_coords, coord_args = EarthSciMLBase._prepare_coord_sys(sys_mtk, domain)
obs_f = EarthSciMLBase.build_coord_observed_function(sys_coords, coord_args,
    EarthSciMLBase.get_needed_vars(op, csys, sys_mtk, domain))

p = EarthSciMLBase.default_params(sys_coords)

obs_vals = obs_f([0.0, 0], p, 0.0, 0.0, 0.0, 1.0)
@test obs_vals[1] ≈ 1.0
@test 1 / (obs_vals[2] * 180 / π) ≈ 111319.44444444445
@test 1 / (obs_vals[3] * 180 / π) ≈ 111320.00000000001
@test obs_vals[4] == 1.0

obs_vals = obs_f([0.0, 0], p, 0.0, 1.0, 3.0, 2.0)
@test obs_vals[1] == 6.0

u = reshape(EarthSciMLBase.init_u(sys_coords, domain), :, size(domain)...)
scimlop = EarthSciMLBase.nonstiff_ops(csys, sys_coords, coord_args, domain, reshape(u, :),
    p, MapBroadcast())
du = similar(u)
du .= 0
scimlop(reshape(du, :), reshape(u, :), p, 0.0)
@test sum(abs.(du)) ≈ 14542.756845747295
du = scimlop(reshape(u, :), p, 0.0)
@test sum(abs.(du)) ≈ 14542.756845747295

du2 = scimlop(reshape(u, :), p, 0.0)
@test du2 ≈ reshape(du, :)

grid = EarthSciMLBase.grid(domain)
sys1 = mtkcompile(sys)
op_pt = [lon => grid[1][1], lat => grid[2][1], lev => grid[3][1]]
prob = ODEProblem(sys1, op_pt, (0.0, 1.0))
sol1 = solve(prob, Tsit5(); abstol = 1e-12, reltol = 1e-12)
@test sol1.retcode == ReturnCode.Success
@test sol1.u[end] ≈ [-27.15156429366082, -26.264264199779465] ||
      sol1.u[end] ≈ [-26.264264199779465, -27.15156429366082]

st = SolverStrangThreads(Tsit5(), 1.0; abstol = 1e-12, reltol = 1e-12)
p = EarthSciMLBase.default_params(sys_mtk)

f_ode, u0_single,
p = EarthSciMLBase._strang_ode_func(sys_coords, coord_args,
    get_tspan(domain), grid; sparse = false)
IIchunks,
integrators = EarthSciMLBase._strang_integrators(st, domain, f_ode, u0_single,
    get_tspan(domain)[1], p)

EarthSciMLBase.threaded_ode_step!(u, IIchunks, integrators, 0.0, 1.0)

let
    varsyms = EarthSciMLBase.var2symbol.(unknowns(sys_coords))
    @test u[1, 1, 1, 1] ≈ sol1[varsyms[1]][end]
    @test u[2, 1, 1, 1] ≈ sol1[varsyms[2]][end]
end

@test sum(abs.(u)) ≈ 212733.04492722102

@testset "mtk_func" begin
    ucopy = copy(u)
    f, sys_coords,
    coord_args = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy;
        sparse = false, tgrad = true)
    fthreads, = EarthSciMLBase.mtk_grid_func(sys_mtk, domain, ucopy,
        MapThreads(); sparse = false, tgrad = false)
    p = EarthSciMLBase.default_params(sys_coords)
    uu = EarthSciMLBase.init_u(sys_coords, domain)
    prob = ODEProblem(f, uu, (0.0, 1.0), p)
    sol = solve(prob, Tsit5())
    varsyms = [Symbol("sys₊", v) for v in EarthSciMLBase.var2symbol.(unknowns(sys1))]
    u_perm = [findfirst(isequal(u), varsyms)
              for u in EarthSciMLBase.var2symbol.(unknowns(sys_mtk))]
    uu = reshape(sol.u[end], size(ucopy)...)[u_perm, :, :, :]
    @test uu[:]≈u[:] rtol=0.01

    @testset "In-place vs. out of place" begin
        du1 = reshape(similar(ucopy), :)
        prob.f(du1, reshape(ucopy, :), p, 0.0)
        du2 = f(reshape(ucopy, :), p, 0.0)
        du3 = fthreads(reshape(ucopy, :), p, 0.0)
        @test du1 ≈ du2 ≈ du3
    end

    @testset "jac dense" begin
        du = similar(f.jac_prototype)
        f.jac(du, ucopy[:], p, 0.0)
        @test sum(du.data) ≈ 12617.772209024473

        du2 = f.jac(reshape(ucopy, :), p, 0.0)
        @test all(du.data .≈ du2.data)
    end

    @testset "jac MapThreads" begin
        du = similar(fthreads.jac_prototype)
        fthreads.jac(du, ucopy[:], p, 0.0)
        @test sum(du.data) ≈ 12617.772209024473

        du2 = fthreads.jac(reshape(ucopy, :), p, 0.0)
        @test all(du.data .≈ du2.data)
    end

    @testset "tgrad" begin
        du = similar(f.jac_prototype)
        for i in 1:size(du.data, 3)
            EarthSciMLBase.block(du, i) .= 0
        end
        f.tgrad(du, ucopy[:], p, 0.0)
        @test sum(du.data) ≈ 0.0
    end
end

prob = ODEProblem(csys, st)
sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-12, reltol = 1e-12)

@test sum(abs.(sol.u[end]))≈3.820642384890682e7 rtol=1e-3

st = SolverStrangThreads(Tsit5(), 1.0)

@testset "Float32" begin
    domain = DomainInfo(
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        u_proto = zeros(Float32, 1, 1, 1, 1), grid_spacing = [0.1, 0.1, 1])

    csys = couple(sys, op, domain)

    prob = ODEProblem(csys, st)
    @test_broken eltype(prob.f(prob.u0[:], prob.p, prob.tspan[1])) == Float32 # MTK v11 code generation does not preserve Float32
    sol = solve(prob, Euler(); dt = 1.0)

    @test sum(abs.(sol.u[end])) ≈ 3.820642384890682e7

    @testset "Split problem" begin
        prob = ODEProblem(csys, SolverIMEX())
        @test_broken eltype(prob.f(prob.u0[:], prob.p, prob.tspan[1])) == Float32 # MTK v11 code generation does not preserve Float32
        @test eltype(prob.f.f1(prob.u0[:], prob.p, prob.tspan[1])) == Float32
        @test_broken eltype(prob.f.f2(prob.u0[:], prob.p, prob.tspan[1])) == Float32 # MTK v11 code generation does not preserve Float32
    end
end

@testset "No operator" begin
    domain = DomainInfo(
        partialderivatives_δxyδlonlat,
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        u_proto = zeros(Float32, 1, 1, 1, 1), grid_spacing = [0.1, 0.1, 1])

    csys = couple(sys, domain)

    prob = ODEProblem(csys, st)
    sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-6, reltol = 1e-6)

    @test sum(abs.(sol.u[end])) ≈ 3.8660308f7
end

@testset "MapKernel" begin
    ucopy = Float32.(u)
    domain = DomainInfo(
        constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
        u_proto = ucopy, grid_spacing = [0.1, 0.1, 1])
    csys = couple(sys, domain)
    prob = ODEProblem(csys, SolverIMEX(MapKernel()))
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, prob.tspan[1])
    @test du[1] ≈ -13.141593f0
end

if Sys.isapple() # TODO: Why aren't the results of these tests deterministic?
    @testset "Metal GPU" begin
        using Metal
        ucopy = MtlArray(Float32.(u))
        domain = DomainInfo(
            constIC(16.0, indepdomain), constBC(16.0, partialdomains...);
            u_proto = ucopy, grid_spacing = [0.1, 0.1, 1])

        csys = couple(sys, op, domain)

        prob = ODEProblem(
            csys, SolverIMEX(MapKernel(), BlockDiagonalJacobian(),
                stiff_sparse = false))

        du = similar(prob.u0)
        prob.f.f1(du, prob.u0, prob.p, prob.tspan[1])
        @test Array(du)[1] ≈ -13.141593f0
        du .= 0.0f0
        prob.f.f2(du, prob.u0, prob.p, prob.tspan[1])
        @test Array(du)[1] ≈ -3.5553088f0
        du .= 0.0f0
        prob.f(du, prob.u0, prob.p, prob.tspan[1])
        @test Array(du)[1] ≈ -3.5553088f0 + -13.141593f0

        @testset "generic lu" begin
            sol = solve(
                prob, KenCarp47(linsolve = GenericLUFactorization()), abstol = 1.0f-7,
                reltol = 1.0f-7)
            @test sum(Float64.(Array(abs.(sol.u[end]))))≈4.0054012260009766e7 rtol=2e-2
        end

        @testset "krylov" begin
            prob = ODEProblem(csys,
                SolverIMEX(MapKernel(), BlockDiagonalOperatorJacobian(),
                    stiff_sparse = false))
            sol = solve(prob, KenCarp47(linsolve = KrylovJL()), abstol = 1.0f-7,
                reltol = 1.0f-7)
            @test sum(Float64.(Array(abs.(sol.u[end]))))≈3.477207642895508e7 rtol=2e-2 # TODO: Why is this different from above?
        end
    end
end

@testset "SimulatorStrategies" begin
    @testset "Strang Threads" begin
        st = SolverStrangThreads(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-12, reltol = 1e-12)
        @test sum(abs.(sol.u[end]))≈3.820642384890682e7 rtol=1e-3
    end

    @testset "Strang Serial" begin
        st = SolverStrangSerial(Tsit5(), 1.0)
        prob = ODEProblem(csys, st)
        sol = solve(prob, Euler(); dt = 1.0, abstol = 1e-12, reltol = 1e-12)
        @test sum(abs.(sol.u[end]))≈3.820642384890682e7 rtol=1e-3
    end

    @testset "IMEX" begin
        st = SolverIMEX()
        prob = ODEProblem(csys, st)
        sol = solve(prob, Tsit5())
        @test sum(abs.(sol.u[end]))≈3.444627331604664e7 rtol=1e-3 # No Splitting error in this one.

        for alg in [MapBroadcast(), MapThreads()]
            for iip in [true] # [true, false] oop not currently implemented for BlockDiagonalJacobian.
                for sparse in [false] # [true, false] # BlockDiagonalJacobian not currently implemented for sparse.
                    @testset "alg=$alg; iip=$iip; sparse=$sparse" begin
                        st = SolverIMEX(alg; stiff_sparse = sparse)
                        prob = ODEProblem{iip}(csys, st)
                        sol = solve(prob, KenCarp47(linsolve = LUFactorization()),
                            abstol = 1e-4, reltol = 1e-4)
                        @test sum(abs.(sol.u[end]))≈3.444627331604664e7 rtol=1e-3
                    end
                end
            end
        end
    end
end

mutable struct cbt
    runcount::Int
end
function EarthSciMLBase.init_callback(c::cbt, sys, sys_mtk, coord_args, dom, alg)
    DiscreteCallback((u, t, integrator) -> true,
        (_) -> c.runcount += 1
    )
end

@testset "callback" begin
    runcount = 0
    af(_) = runcount += 1
    cb = DiscreteCallback(
        (u, t, integrator) -> true,
        af
    )
    @testset "strang" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, st)
        solve(prob, Euler(); dt = 1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
    runcount = 0
    @testset "imex" begin
        cc = cbt(0)
        csys2 = couple(csys, cb, cc)
        prob = ODEProblem(csys2, SolverIMEX())
        solve(prob, Tsit5(); dt = 1.0)
        @test runcount > 0
        @test cc.runcount > 0
    end
end
