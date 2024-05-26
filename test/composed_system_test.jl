using EarthSciMLBase
using ModelingToolkit
using Test

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

    function Base.:(+)(s::SEqn, i::IEqn)::ComposedEarthSciMLSystem
        seqn = s.sys
        ieqn = i.sys
        ComposedEarthSciMLSystem(
            ConnectorSystem([
                    ieqn.S ~ seqn.S,
                    seqn.I ~ ieqn.I], s, i),
            s, i,
        )
    end

    function Base.:(+)(s::SEqn, r::REqn)::ComposedEarthSciMLSystem
        seqn = s.sys
        reqn = r.sys
        ComposedEarthSciMLSystem(
            ConnectorSystem([seqn.R ~ reqn.R], s, r),
            s, r,
        )
    end

    function Base.:(+)(i::IEqn, r::REqn)::ComposedEarthSciMLSystem
        ieqn = i.sys
        reqn = r.sys
        ComposedEarthSciMLSystem(
            ConnectorSystem([
                    ieqn.R ~ reqn.R,
                    reqn.I ~ ieqn.I], i, r),
            i, r,
        )
    end

    seqn, ieqn, reqn = SEqn(t), IEqn(t), REqn(t)

    sir = seqn + ieqn + reqn

    sirfinal = get_mtk(sir)

    sir_simple = structural_simplify(sirfinal)

    want_eqs = [
        Differential(t)(reqn.R) ~ reqn.γ * reqn.I,
        Differential(t)(seqn.S) ~ (-seqn.β * seqn.I * seqn.S) / (seqn.I + seqn.R + seqn.S),
        Differential(t)(ieqn.I) ~ (ieqn.β * ieqn.I * ieqn.S) / (ieqn.I + ieqn.R + ieqn.S) - ieqn.γ * ieqn.I,
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

        g = graph(sir)
        l = collect(labels(g))
        el = collect(edge_labels(g))

        @test sort(l) == sort([:seqn, :ieqn, :reqn])
        @test length(el) == 3
    end
end

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

function couple(s::B, f::A)
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
couple(sf::B, emis::C) = operator_compose(sf, emis, Dict(
    sf.sys.NO2 => emis.sys.NO2 => uu,
))

@testset "Composed System Permutations" begin
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