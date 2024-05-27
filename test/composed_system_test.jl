using EarthSciMLBase
using ModelingToolkit
using Test
using Catalyst

@testset "Composed System" begin
    @parameters t

    struct SEqn <: EarthSciMLODESystem
        sys::ODESystem

        function SEqn(t)
            @variables S(t), I(t), R(t)
            D = Differential(t)
            N = S + I + R
            @parameters β
            @named seqn = ODESystem([D(S) ~ -β * S * I / N])
            new(seqn)
        end
    end

    struct IEqn <: EarthSciMLODESystem
        sys::ODESystem

        function IEqn(t)
            @variables S(t), I(t), R(t)
            D = Differential(t)
            N = S + I + R
            @parameters β, γ
            @named ieqn = ODESystem([D(I) ~ β * S * I / N - γ * I])
            new(ieqn)
        end
    end

    struct REqn <: EarthSciMLODESystem
        sys::ODESystem

        function REqn(t)
            @variables I(t), R(t)
            D = Differential(t)
            @parameters γ
            @named reqn = ODESystem([D(R) ~ γ * I])
            new(reqn)
        end
    end

    function EarthSciMLBase.couple(s::SEqn, i::IEqn)
        ConnectorSystem([
                i.sys.S ~ s.sys.S,
                s.sys.I ~ i.sys.I], s, i)
    end

    EarthSciMLBase.couple(s::SEqn, r::REqn) = ConnectorSystem([s.sys.R ~ r.sys.R], s, r)

    function EarthSciMLBase.couple(i::IEqn, r::REqn)
        ConnectorSystem([
                i.sys.R ~ r.sys.R,
                r.sys.I ~ i.sys.I], i, r)
    end

    seqn, ieqn, reqn = SEqn(t), IEqn(t), REqn(t)

    sir = seqn + ieqn + reqn

    sirfinal = get_mtk(sir)

    sir_simple = structural_simplify(sirfinal)

    want_eqs = [
        Differential(t)(reqn.sys.R) ~ reqn.sys.γ * reqn.sys.I,
        Differential(t)(seqn.sys.S) ~ (-seqn.sys.β * seqn.sys.I * seqn.sys.S) / (seqn.sys.I + seqn.sys.R + seqn.sys.S),
        Differential(t)(ieqn.sys.I) ~ (ieqn.sys.β * ieqn.sys.I * ieqn.sys.S) / (ieqn.sys.I + ieqn.sys.R + ieqn.sys.S) - ieqn.sys.γ * ieqn.sys.I,
    ]

    have_eqs = equations(sir_simple)
    for eq in want_eqs
        @test eq in have_eqs
    end
    for eq in have_eqs
        @test eq in want_eqs
    end

    @testset "Graph" begin
        using MetaGraphsNext

        seqn, ieqn, reqn = SEqn(t), IEqn(t), REqn(t)
        sir = seqn + ieqn + reqn

        g = graph(sir)
        l = collect(labels(g))
        el = collect(edge_labels(g))

        @test sort(l) == sort([:seqn, :ieqn, :reqn])
        @test length(el) == 3
    end
end

@testset "Composed System Permutations" begin
    @parameters t
    struct A <: EarthSciMLODESystem
        sys::ODESystem
        function A(t)
            @parameters j_unit = 1
            @variables j_NO2(t) = 0.0149
            eqs = [
                j_NO2 ~ j_unit
            ]
            new(ODESystem(eqs, t, [j_NO2], [j_unit]; name=:a))
        end
    end

    struct B <: EarthSciMLODESystem
        sys::ODESystem
        rxn_sys::ReactionSystem
        B(sys::ModelingToolkit.ODESystem, rxn_sys::ReactionSystem) = new(sys, rxn_sys)
        function B(t)
            @parameters jNO2 = 0.0149
            @species NO2(t) = 10.0
            rxs = [
                Reaction(jNO2, [NO2], [], [1], [1])
            ]
            rxn_sys = ReactionSystem(rxs, t; combinatoric_ratelaws=false, name=:b)
            new(convert(ODESystem, rxn_sys), rxn_sys)
        end
    end

    function EarthSciMLBase.couple(s::B, f::A)
        sys = param_to_var(s.sys, :jNO2)
        s = B(sys, s.rxn_sys)
        ConnectorSystem([s.sys.jNO2 ~ f.sys.j_NO2], s, f)
    end

    struct C <: EarthSciMLODESystem
        sys::ODESystem
        function C(t)
            @parameters emis = 1
            @variables NO2(t) = 0.00014
            eqs = [NO2 ~ emis]
            new(ODESystem(eqs, t, [NO2], [emis]; name=:c))
        end
    end

    @constants uu = 1
    EarthSciMLBase.couple(sf::B, emis::C) = operator_compose(sf, emis, Dict(
        sf.sys.NO2 => emis.sys.NO2 => uu,
    ))
    models = [
        A(t) + B(t) + C(t)
        C(t) + B(t) + A(t)
        B(t) + A(t) + C(t)
        C(t) + A(t) + B(t)
        A(t) + C(t) + B(t)
        B(t) + C(t) + A(t)
    ]
    for model ∈ models
        model_mtk = get_mtk(model)
        m = structural_simplify(model_mtk)
        eqstr = string(equations(m))
        @test occursin("b₊c_NO2(t)", eqstr)
        @test occursin("b₊jNO2(t)", eqstr)
        @test occursin("b₊NO2(t)", string(states(m)))
        obstr = string(observed(m))
        @test occursin("a₊j_NO2(t) ~ a₊j_unit", obstr)
        @test occursin("c₊NO2(t) ~ c₊emis", obstr)
        @test occursin("b₊jNO2(t) ~ a₊j_NO2(t)", obstr)
        @test occursin("b₊c_NO2(t) ~ c₊NO2(t)", obstr)
    end
end