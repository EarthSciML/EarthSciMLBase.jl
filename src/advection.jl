export Advection, MeanWind, ConstantWind

struct MeanWind <: EarthSciMLODESystem
    sys::ODESystem
    MeanWind(t, ndims) = new(ODESystem(Equation[], t, collect((@variables u[1:ndims](t))[1]), []; name=:meanwind))
end

struct Advection end

function advection(vars, icbc::ICBC)
    iv = ivar(icbc)
    pvs = pvars(icbc)
    uvars = (@variables meanwind₊u[1:length(pvs)](..))[1]
    varsdims = Num[v for v ∈ vars]
    udims = Num[ui(pvs..., iv) for ui ∈ uvars]
    eqs = Equation[]
    for var ∈ varsdims
        terms(wind) = sum(((pv) -> -wind * Differential(pv)(var)).(pvs))
        rhs = sum(vcat([terms(wind) for wind in udims]))
        eq = Differential(iv)(var) ~ rhs
        push!(eqs, eq)
    end
    eqs
end

function Base.:(+)(c::ComposedEarthSciMLSystem, a::Advection)::ComposedEarthSciMLSystem
    @assert isa(c.icbc, ICBC) "The system must have initial and boundary conditions to add advection."
    
    # Add in a model component to allow the specification of the wind velocity.
    c += MeanWind(ivar(c.icbc), length(pvars(c.icbc)))
    
    function f(sys::ModelingToolkit.PDESystem)
        # Get the variables we want to advect.
        vars = pde_states(sys) # TODO(CT): Change once MTK can get the states of a PDESystem.
        eqs = advection(vars, c.icbc)
        operator_compose!(sys, eqs)
    end
    ComposedEarthSciMLSystem(c.systems, c.icbc, [c.pdefunctions; f])
end
Base.:(+)(a::Advection, c::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = c + a

function is_variable(var)::Bool
    if haskey(var.metadata, Symbolics.VariableSource)
        return var.metadata[Symbolics.VariableSource][1] == :variables
    elseif haskey(var.metadata.value.metadata, Symbolics.VariableSource)
        return var.metadata.value.metadata[Symbolics.VariableSource][1] == :variables
    else
        error("unknown type")
    end
end

# TODO(CT): Delete once MTK can get the states of a PDESystem.
function pde_states(sys::ModelingToolkit.PDESystem)
    s = Num[]
    for eq ∈ equations(sys)
        vars = Symbolics.get_variables(eq)
        for var ∈ vars
            var
            if is_variable(var)
                push!(s, var)
            end
        end
    end
    s
end

struct ConstantWind <: EarthSciMLODESystem
    sys::ODESystem

    function ConstantWind(t, vals...)
        @assert length(vals) > 0 "Must specify at least one wind component speed."
        @variables u[1:length(vals)](t)
        eqs = Symbolics.scalarize(u .~ collect(vals))
        new(ODESystem(eqs, t, u, []; name=:constantwind))
    end
end
function Base.:(+)(mw::MeanWind, w::ConstantWind)::ComposedEarthSciMLSystem
    @assert length(mw.sys.u) == length(w.sys.u) "The number of wind components must match the number of mean wind components."
    ComposedEarthSciMLSystem(ConnectorSystem(
        Symbolics.scalarize(mw.sys.u .~ w.sys.u),
        mw, w,
    ))
end
Base.:(+)(w::ConstantWind, mw::MeanWind)::ComposedEarthSciMLSystem = mw + w

using DomainSets

@parameters t x y

icbc = ICBC(
    periodicBC(x ∈ Interval(0, 1.0)),
    constBC(1.0, y ∈ Interval(0, 1.0)),
    constIC(0.0, t ∈ Interval(0, 1.0)),
)

scalar = @variables c1(..) c2(..)
xxx = advection(scalar, icbc)

@variables u[1:length(pvars(icbc))](..)
xxy = PDESystem(
    [equations(xxx); [u[1](x,y,t) ~ 1.0, u[2](x,y,t) ~ -1.0]],
    xxx.bcs,
    domains(icbc),
    xxx.ivs,
    xxx.dvs;
    name=:advection,
)


N = 10
using MethodOfLines, DifferentialEquations

# Integers for x and y are interpreted as number of points. Use a Float to directly specify stepsizes dx and dy.
discretization = MOLFiniteDifference([x=>N, y=>N], t, approx_order=2)
@time prob = discretize(xxy,discretization)
@time sol = solve(prob, Tsit5(), saveat=0.1)

discrete_x = sol[x]
discrete_y = sol[y]
discrete_t = sol[t]

solc1 = sol[c1(x, y, t)]
solc2 = sol[c2(x, y, t)]

using Plots
anim = @animate for k in 1:length(discrete_t)
    heatmap(solc1[1:end, 1:end, k]', title="$(discrete_t[k])")
end
gif(anim, fps = 8)


######

@parameters t, x

struct ExampleSys <: EarthSciMLODESystem
    sys::ODESystem

    function ExampleSys(t; name)
        @variables x(t)
        @parameters p=1.0
        D = Differential(t)
        new(ODESystem([D(x) ~ p], t; name))
    end
end

@named sys1 = ExampleSys(t)
@named sys2 = ExampleSys(t)
icbc = ICBC(constBC(1.0, x ∈ Interval(0, 1.0)), constIC(0.0, t ∈ Interval(0, 1.0)))

combined = operator_compose(sys1, sys2)

combined_pde = combined + icbc + ConstantWind(t, 1.0) + Advection()

combined_mtk = get_mtk(combined_pde)

display(equations(combined_mtk))

combined_mtk.bcs

structural_simplify(combined_mtk)

using MethodOfLines, DifferentialEquations

discretization = MOLFiniteDifference([x=>10], t, approx_order=2)
@time prob = discretize(combined_mtk, discretization)
@time sol = solve(prob, Tsit5(), saveat=0.1)

discrete_x = sol[x]
discrete_t = sol[t]

solc1 = sol[c1(x, y, t)]
solc2 = sol[c2(x, y, t)]

using Plots
anim = @animate for k in 1:length(discrete_t)
    heatmap(solc1[1:end, 1:end, k]', title="$(discrete_t[k])")
end
gif(anim, fps = 8)



######

@parameters x
@variables t u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#2D PDE
C=1
eq  = Dtt(u(t,x)) ~ C^2*Dxx(u(t,x))

# Initial and boundary conditions
bcs = [u(t,0) ~ 0.,# for all t > 0
       u(t,1) ~ 0.,# for all t > 0
       u(0,x) ~ x*(1. - x), #for all 0 < x < 1
       Dt(u(0,x)) ~ 0. ] #for all  0 < x < 1]

# Space and time domains
domains = [t ∈ (0.0,1.0),
           x ∈ (0.0,1.0)]

@named pde_system = PDESystem(eq,bcs,domains,[t,x],[u])

structural_simplify(pde_system)