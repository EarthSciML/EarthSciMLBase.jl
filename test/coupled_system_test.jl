using EarthSciMLBase
using ModelingToolkit
using ModelingToolkit: t, D
using ModelingToolkit: t_nounits, D_nounits
using Test
using Catalyst
using DynamicQuantities
using OrdinaryDiffEqTsit5

@testset "Composed System" begin
    struct SEqnCoupler
        sys::Any
    end
    function SEqn()
        @variables S(t), I(t), R(t)
        N = S + I + R
        @parameters β [unit = u"s^-1"]
        @named seqn = ODESystem([D(S) ~ -β * S * I / N], t,
            metadata = Dict(:coupletype => SEqnCoupler))
    end

    struct IEqnCoupler
        sys::Any
    end
    function IEqn()
        @variables S(t), I(t), R(t)
        N = S + I + R
        @parameters β [unit = u"s^-1"]
        @parameters γ [unit = u"s^-1"]
        @named ieqn = ODESystem([D(I) ~ β * S * I / N - γ * I], t,
            metadata = Dict(:coupletype => IEqnCoupler))
    end

    struct REqnCoupler
        sys::Any
    end
    function REqn()
        @variables I(t), R(t)
        @parameters γ [unit = u"s^-1"]
        @named reqn = ODESystem([D(R) ~ γ * I], t,
            metadata = Dict(:coupletype => REqnCoupler))
    end

    function EarthSciMLBase.couple2(s::SEqnCoupler, i::IEqnCoupler)
        s, i = s.sys, i.sys
        ConnectorSystem([
                i.S ~ s.S,
                s.I ~ i.I], s, i)
    end

    function EarthSciMLBase.couple2(s::SEqnCoupler, r::REqnCoupler)
        s, r = s.sys, r.sys
        ConnectorSystem([s.R ~ r.R], s, r)
    end

    function EarthSciMLBase.couple2(i::IEqnCoupler, r::REqnCoupler)
        i, r = i.sys, r.sys
        ConnectorSystem([
                i.R ~ r.R,
                r.I ~ i.I], i, r)
    end

    seqn, ieqn, reqn = SEqn(), IEqn(), REqn()

    sir = couple(seqn, ieqn, reqn)

    sirfinal = convert(ODESystem, sir)

    want_eqs = [
        D(reqn.R) ~ reqn.γ * reqn.I,
        D(seqn.S) ~ (-seqn.β * seqn.I * seqn.S) / (seqn.I + seqn.R + seqn.S),
        D(ieqn.I) ~
        (ieqn.β * ieqn.I * ieqn.S) /
        (ieqn.I + ieqn.R + ieqn.S) - ieqn.γ * ieqn.I
    ]

    have_eqs = equations(sirfinal)
    obs = observed(sirfinal)
    for eq in want_eqs
        @test eq in have_eqs
    end
    for eq in have_eqs
        @test eq in want_eqs
    end

    @testset "Graph" begin
        using MetaGraphsNext

        seqn, ieqn, reqn = SEqn(), IEqn(), REqn()
        sir = couple(seqn, ieqn, reqn)

        g = graph(sir)
        l = collect(labels(g))
        el = collect(edge_labels(g))

        @test sort(l) == sort([:seqn, :ieqn, :reqn])
        @test length(el) == 3
    end
end

@testset "Composed System Permutations" begin
    struct ACoupler
        sys::Any
    end
    function A()
        @parameters j_unit=1 [unit = u"s^-1"]
        @variables j_NO2(t)=0.0149 [unit = u"s^-1"]
        eqs = [
            j_NO2 ~ j_unit
        ]
        ODESystem(eqs, t, [j_NO2], [j_unit]; name = :a,
            metadata = Dict(:coupletype => ACoupler))
    end

    struct BCoupler
        sys::Any
    end
    function B()
        @parameters jNO2=0.0149 [unit = u"s^-1"]
        @species NO2(t) = 10.0
        rxs = [
            Reaction(jNO2, [NO2], [], [1], [])
        ]
        rs = complete(ReactionSystem(rxs, t; combinatoric_ratelaws = false, name = :b))
        convert(ODESystem, rs; metadata = Dict(:coupletype => BCoupler))
    end

    function EarthSciMLBase.couple2(b::BCoupler, a::ACoupler)
        a, b = a.sys, b.sys
        b = param_to_var(b, :jNO2)
        ConnectorSystem([b.jNO2 ~ a.j_NO2], b, a)
    end

    struct CCoupler
        sys::Any
    end
    function C()
        @parameters emis=1 [unit = u"s^-1"]
        @variables NO2(t)=0.00014 [unit = u"s^-1"]
        eqs = [NO2 ~ emis]
        ODESystem(
            eqs, t, [NO2], [emis]; name = :c, metadata = Dict(:coupletype => CCoupler))
    end

    function EarthSciMLBase.couple2(b::BCoupler, c::CCoupler)
        b, c = b.sys, c.sys
        @constants uu = 1
        operator_compose(b, c, Dict(
            b.NO2 => c.NO2 => uu,
        ))
    end

    models = [couple(A(), B(), C())
              couple(C(), B(), A())
              couple(B(), A(), C())
              couple(C(), A(), B())
              couple(A(), C(), B())
              couple(B(), C(), A())]
    for (i, model) in enumerate(models)
        @testset "permutation $i" begin
            m = convert(ODESystem, model)
            eqstr = string(equations(m))
            @test occursin("b₊c_NO2(t)", eqstr)
            @test occursin("b₊jNO2(t)", eqstr)
            @test occursin("b₊NO2(t)", string(unknowns(m)))
            obstr = string(observed(m))
            @test occursin("a₊j_NO2(t) ~ a₊j_unit", obstr)
            @test occursin("c₊NO2(t) ~ c₊emis", obstr)
            @test occursin("b₊jNO2(t) ~ a₊j_NO2(t)", obstr)
            @test occursin("b₊c_NO2(t) ~ c₊NO2(t)", obstr)
        end
    end

    @testset "Stable evaluation" begin
        sys = couple(A(), B(), C())
        s = convert(ODESystem, sys)
        eqs1 = string(equations(s))
        @test occursin("b₊c_NO2(t)", eqs1)
        eqs2 = string(equations(s))
        @test occursin("b₊c_NO2(t)", eqs2)
        @test eqs1 == eqs2
    end
end

mutable struct ParamTest
    y::Any
end
(t::ParamTest)(x) = t.y - x

function update_affect!(integ, u, p, ctx)
    integ.p[only(p)].y = integ.t
end

@testset "Event filtering" begin
    p1 = ParamTest(1)
    tp1 = typeof(p1)
    @parameters (p_1::tp1)(..) = p1
    @parameters p_2(ModelingToolkit.t_nounits) = 1
    @parameters (p_3::tp1)(..) = ParamTest(1)
    @parameters p_4(ModelingToolkit.t_nounits) = 1
    @variables x(ModelingToolkit.t_nounits) = 0
    @variables x2(ModelingToolkit.t_nounits) = 0
    @variables x3(ModelingToolkit.t_nounits)

    event1 = [1.0, 2, 3] => (update_affect!, [], [p_1], [], p1)
    event2 = [1.0, 2, 3] => [p_2 ~ t]
    event3 = [1.0, 2, 3] => (update_affect!, [], [p_3], [], nothing)
    event4 = [1.0, 2, 3] => [p_4 ~ t]

    sys = ODESystem(
        [
            ModelingToolkit.D_nounits(x) ~ p_1(x),
            ModelingToolkit.D_nounits(x2) ~ p_2 - x2,
            x3 ~ p_3(x2) + p_4
        ],
        ModelingToolkit.t_nounits; name = :test,
        discrete_events = [event1, event2, event3, event4]
    )
    #sys = EarthSciMLBase.remove_extra_defaults(sys, structural_simplify(sys))

    prob = ODEProblem(structural_simplify(sys), [], (0, 100), [])
    sol = solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8)
    @test sol[x][end] ≈ 3
    @test sol[x2][end] ≈ 3

    @testset "affected vars" begin
        de = ModelingToolkit.get_discrete_events(sys)
        @test isequal(only(EarthSciMLBase.get_affected_vars(de[1])), p_1)
        @test isequal(only(EarthSciMLBase.get_affected_vars(de[2])), p_2)
        @test isequal(only(EarthSciMLBase.get_affected_vars(de[3])), p_3)
        @test isequal(only(EarthSciMLBase.get_affected_vars(de[4])), p_4)
    end

    @testset "variable in equations" begin
        sys2 = structural_simplify(sys)
        @test EarthSciMLBase.var_in_eqs(p_1, equations(sys2)) == true
        @test EarthSciMLBase.var_in_eqs(p_2, equations(sys2)) == true
        @test EarthSciMLBase.var_in_eqs(p_3, equations(sys2)) == false
        @test EarthSciMLBase.var_in_eqs(p_4, equations(sys2)) == false
    end

    @testset "filter events" begin
        kept_events = EarthSciMLBase.filter_discrete_events(structural_simplify(sys), [])
        @test length(kept_events) == 2
        @test EarthSciMLBase.var2symbol(only(EarthSciMLBase.get_affected_vars(kept_events[1]))) ==
              :p_1
        @test EarthSciMLBase.var2symbol(only(EarthSciMLBase.get_affected_vars(kept_events[2]))) ==
              :p_2
    end

    @testset "prune observed" begin
        sys2 = EarthSciMLBase.prune_observed(sys, structural_simplify(sys), [])
        @test length(equations(sys2)) == 2
        @test length(ModelingToolkit.get_discrete_events(sys2)) == 2
    end

    sys2 = EarthSciMLBase.prune_observed(sys, structural_simplify(sys), [])
    prob = ODEProblem(structural_simplify(sys2), [], (0, 100), [])
    sol = solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8)
    @test sol[x][end] ≈ 3
    @test sol[x2][end] ≈ 3
end

@testset "Composed System with Events" begin
    function create_sys(; name = :test)
        tp1 = typeof(ParamTest(1))
        @parameters (p_1::tp1)(..) = ParamTest(1)
        @parameters p_2(ModelingToolkit.t_nounits) = 1
        @parameters (p_3::tp1)(..) = ParamTest(1)
        @parameters p_4(ModelingToolkit.t_nounits) = 1
        @variables x(ModelingToolkit.t_nounits) = 0
        @variables x2(ModelingToolkit.t_nounits) = 0
        @variables x3(ModelingToolkit.t_nounits) = 0

        event1 = [1.0, 2, 3] => (update_affect!, [], [p_1], [], nothing)
        event2 = [1.0, 2, 3] => [p_2 ~ t]
        event3 = [1.0, 2, 3] => (update_affect!, [], [p_3], [], nothing)
        event4 = [1.0, 2, 3] => [p_4 ~ t]

        ODESystem(
            [
                ModelingToolkit.D_nounits(x) ~ p_1(x),
                ModelingToolkit.D_nounits(x2) ~ p_2 - x2,
                x3 ~ p_3(x2) + p_4
            ],
            ModelingToolkit.t_nounits; name = name,
            discrete_events = [event1, event2, event3, event4]
        )
    end

    sys_composed = compose(
        ODESystem(Equation[], ModelingToolkit.t_nounits; name = :coupled),
        create_sys(name = :a), create_sys(name = :b))
    sys_flattened = ModelingToolkit.flatten(sys_composed)
    #sysc2 = EarthSciMLBase.remove_extra_defaults(
    #    sys_flattened, structural_simplify(sys_flattened))
    sysc2 = sys_flattened
    sysc3 = EarthSciMLBase.prune_observed(sysc2, structural_simplify(sysc2), [])
    prob = ODEProblem(structural_simplify(sysc3), [], (0, 100), [])
    sol = solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8)
    @test length(sol.u[end]) == 4
    @test all(sol.u[end] .≈ 3)
end

@testset "Composed System with Events and operator_compose" begin
    struct CoupleType1
        sys::Any
    end
    struct CoupleType2
        sys::Any
    end
    function create_sys(coupletype; name = :test)
        tp1 = typeof(ParamTest(1))
        @parameters (p_1::tp1)(..) = ParamTest(1)
        @parameters p_2(ModelingToolkit.t_nounits) = 1
        @parameters (p_3::tp1)(..) = ParamTest(1)
        @parameters p_4(ModelingToolkit.t_nounits) = 1
        @variables x(ModelingToolkit.t_nounits) = 0
        @variables x2(ModelingToolkit.t_nounits) = 0
        @variables x3(ModelingToolkit.t_nounits)

        event1 = [1.0, 2, 3] => (update_affect!, [], [p_1], [], nothing)
        event2 = [1.0, 2, 3] => [p_2 ~ t]
        event3 = [1.0, 2, 3] => (update_affect!, [], [p_3], [], nothing)
        event4 = [1.0, 2, 3] => [p_4 ~ t]

        ODESystem(
            [
                ModelingToolkit.D_nounits(x) ~ p_1(x),
                ModelingToolkit.D_nounits(x2) ~ p_2 - x2,
                x3 ~ p_3(x2) + p_4
            ],
            ModelingToolkit.t_nounits; name = name,
            discrete_events = [event1, event2, event3, event4],
            metadata = Dict(:coupletype => coupletype)
        )
    end
    function EarthSciMLBase.couple2(a::CoupleType1, b::CoupleType2)
        a, b = a.sys, b.sys
        operator_compose(a, b)
    end

    a = create_sys(CoupleType1, name = :a)
    b = create_sys(CoupleType2, name = :b)
    coupled_sys = couple(a, b)
    sys = convert(ODESystem, coupled_sys)
    @test sys.a₊x in keys(ModelingToolkit.get_defaults(sys))
    @test sys.a₊x2 in keys(ModelingToolkit.get_defaults(sys))
    @test occursin("a₊b_ddt_xˍt(t)", string(equations(sys)))
    @test occursin("a₊b_ddt_x2ˍt(t)", string(equations(sys)))
    prob = ODEProblem(sys, [], (0, 100), [])
    sol = solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8)
    @test length(sol.u[end]) == 2
    @test all(sol.u[end] .≈ 3)
end

@testset "Transient Equality" begin
    struct SourceCoupler
        sys::Any
    end
    function Source(; name = :Source)
        @parameters T_0 = 300
        @variables T(t)
        ODESystem([T ~ T_0], t_nounits, [T], [T_0]; name = name,
            metadata = Dict(:coupletype => SourceCoupler))
    end

    struct DestCoupler
        sys::Any
    end
    function Dest(; name = :Dest)
        @parameters T = 400
        @variables x(t)
        ODESystem([D_nounits(x) ~ T], t_nounits, [x], [T]; name = name,
            metadata = Dict(:coupletype => DestCoupler))
    end
    struct DestCoupler
        sys::Any
    end
    function Dest2(; name = :Dest2)
        @parameters T = 400
        @variables x(t)
        ODESystem([x ~ T], t_nounits, [x], [T]; name = name,
            metadata = Dict(:coupletype => DestCoupler))
    end

    function EarthSciMLBase.couple2(s::SourceCoupler, d::DestCoupler)
        s, d = s.sys, d.sys
        d = param_to_var(d, :T)
        ConnectorSystem([d.T ~ s.T], s, d)
    end

    src = Source()
    @named dst1 = Dest()
    @named dst2 = Dest2()

    for systems in [(src, dst1, dst2),
        (dst1, src, dst2),
        (dst2, src, dst1),
        (src, dst2, dst1),
        (dst1, dst2, src),
        (dst2, dst1, src)]
        coupled_sys = couple(systems...)

        sys = convert(ODESystem, coupled_sys)
        equations(sys)
        observed(sys)

        obs_lhss = [eq.lhs for eq in observed(sys)]
        @test contains(string(obs_lhss), "dst1₊T(t)")
        @test contains(string(obs_lhss), "dst2₊T(t)")
    end
end

@testset "sys_discrete_event" begin
    @parameters a=0 b=0
    @variables begin
        x(t_nounits) = 0
        y(t_nounits)
    end

    # Utility function to check if a variable is needed in the system,
    # i.e., if one of the state variables depends on it.
    function is_var_needed(var, sys)
        var = EarthSciMLBase.var2symbol(var)
        if var in EarthSciMLBase.var2symbol.(unknowns(sys))
            return true
        end
        exprs = [eq.rhs for eq in equations(sys)]
        needed_obs = ModelingToolkit.observed_equations_used_by(sys, exprs)
        needed_vars = getproperty.(observed(sys)[needed_obs], :lhs)
        return var in EarthSciMLBase.var2symbol.(needed_vars)
    end

    runcount1 = 0
    function sysevent1(sys)
        function f1!(integ, u, p, ctx)
            if is_var_needed(sys.sys1₊x, sys) # Only run if x is needed in the system.
                #global runcount1 += 1
                runcount1 += 1
                integ.ps[p.sys1₊a] = 1
            end
        end
        return [3.0] => (f1!, [], [sys.sys1₊a], [], nothing)
    end
    runcount2 = 0
    function sysevent2(sys)
        function f2!(integ, u, p, ctx)
            if is_var_needed(sys.sys2₊y, sys) # Only run if y is needed in the system.
                #global runcount2 += 1
                runcount2 += 1
                integ.ps[p.sys2₊b] = 1
            end
        end
        return [5.0] => (f2!, [], [sys.sys2₊b], [], nothing)
    end

    sys1 = ODESystem([D(x) ~ a], t_nounits, [x], [a]; name = :sys1,
        metadata = Dict(:sys_discrete_event => sysevent1))
    sys2 = ODESystem([y ~ b], t_nounits, [y], [b]; name = :sys2,
        metadata = Dict(:sys_discrete_event => sysevent2))

    model1 = couple(sys1, sys2)
    sys = convert(ODESystem, model1)

    @test length(ModelingToolkit.get_discrete_events(sys)) == 2

    sol = solve(ODEProblem(sys), Tsit5(), tspan = (0, 10))

    # Here the derivative of x is 0 until t = 3, then because of sysevent1 it becomes 1 for
    # the rest of the simulation, so the final value of x should be 7.
    # Because sys2.y is not a state variable, sysevent 2 does not run.
    @test sol[sys.sys1₊x][end] ≈ 7
    @test runcount1 == 1
    @test runcount2 == 0

    ### Now, try again after coupling the two variables together so x depends on y.

    # reset count.
    runcount1, runcount2 = 0, 0

    struct Couple1
        sys::Any
    end
    struct Couple2
        sys::Any
    end

    sys1 = ODESystem([D(x) ~ a], t_nounits, [x], [a]; name = :sys1,
        metadata = Dict(:sys_discrete_event => sysevent1,
            :coupletype => Couple1))
    sys2 = ODESystem([y ~ b], t_nounits, [y], [b]; name = :sys2,
        metadata = Dict(:sys_discrete_event => sysevent2,
            :coupletype => Couple2))

    function EarthSciMLBase.couple2(s1::Couple1, s2::Couple2)
        s1, s2 = s1.sys, s2.sys
        EarthSciMLBase.operator_compose(s1, s2, Dict(
            s1.x => s2.y
        ))
    end

    model1 = couple(sys1, sys2)
    sys = convert(ODESystem, model1)

    @test length(ModelingToolkit.get_discrete_events(sys)) == 2

    sol = solve(ODEProblem(sys), Tsit5(), tspan = (0, 10))

    # Here the derivative of x is 0 until t = 3, then because of sysevent1 it becomes 1 for
    # until t = 5, and then because of sysevent2 it become 2 for the rest of the simulation.
    @test sol[sys.sys1₊x][end] ≈ 12
    @test runcount1 == 1
    @test runcount2 == 1
end
