export Advection, MeanWind, ConstantWind

function meanwind_vars(t, pvars; prefix="", multidim=false)
    uvars = []
    denominator = ModelingToolkit.get_unit(t)
    for pv ∈ pvars
        sym = Symbol("$(prefix)v_$(pv)")
        if multidim # Multi-dimensional variable
            uv = (@variables $sym(..))[1]
        else
            uv = (@variables $sym(t))[1]
        end

        # set metadata
        uv = add_metadata(uv, pv)
        numerator = ModelingToolkit.get_unit(uv)
        uv = Symbolics.setmetadata(uv, ModelingToolkit.VariableUnit, numerator / denominator)
        uv = Symbolics.setmetadata(uv, ModelingToolkit.VariableDescription,
            "Mean wind speed in the $(pv) direction.")
        push!(uvars, uv)
    end
    uvars
end

"""
$(SIGNATURES)

A model component that represents the mean wind velocity, where `t` is the independent variable,
`iv` is the independent variable,
and `pvars` is the partial dependent variables for the domain.
"""
struct MeanWind <: EarthSciMLODESystem
    sys::ODESystem
    function MeanWind(t, pvars)
        uvars = meanwind_vars(t, pvars)
        new(ODESystem(Equation[], t, uvars, []; name=:meanwind))
    end
end

"""
$(SIGNATURES)

Apply advection to a model.

"""
struct Advection end

# Create a system of equations that apply advection to the variables in `vars`, 
# using the given initial and boundary conditions to determine which directions
# to advect in.
function advection(vars, di::DomainInfo)
    iv = ivar(di)
    pvs = pvars(di)
    uvars = meanwind_vars(iv, pvs; prefix="meanwind₊", multidim=true)
    varsdims = Num[v for v ∈ vars]
    udims = Num[ui(iv, pvs...) for ui ∈ uvars]
    δs = di.partial_derivative_func(pvs) # get partial derivative operators. May contain coordinate transforms.
    eqs = Equation[]
    for var ∈ varsdims
        rhs = sum(vcat([-wind * δs[i](var) for (i, wind) ∈ enumerate(udims)]))
        eq = Differential(iv)(var) ~ rhs
        push!(eqs, eq)
    end
    eqs
end

function Base.:(+)(c::ComposedEarthSciMLSystem, _::Advection)::ComposedEarthSciMLSystem
    @assert isa(c.domaininfo, DomainInfo) "The system must have initial and boundary conditions (i.e. DomainInfo) to add advection."

    # Add in a model component to allow the specification of the wind velocity.
    push!(c.systems, MeanWind(ivar(c.domaininfo), pvars(c.domaininfo)))

    function f(sys::ModelingToolkit.PDESystem)
        eqs = advection(sys.dvs, c.domaininfo)
        operator_compose!(sys, eqs)
    end
    push!(c.pdefunctions, f)
    c
end
Base.:(+)(a::Advection, c::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = c + a

"""
$(SIGNATURES)

Construct a constant wind velocity model component with the given wind speed(s), which
should include units. For example, `ConstantWind(t, 1u"m/s", 2u"m/s")`.
"""
struct ConstantWind <: EarthSciMLODESystem
    sys::ODESystem
    ndims

    function ConstantWind(t, vals...)
        counts = ["st", "nd", "rd", "th", "th", "th", "th"]
        uvars = []
        for (i, val) ∈ enumerate(vals)
            sym = Symbol("v_$i")
            uv = (@variables $sym(t))[1]
            uv = Symbolics.setmetadata(uv, ModelingToolkit.VariableUnit, unit(val))
            uv = Symbolics.setmetadata(uv, ModelingToolkit.VariableDescription,
                "Constant wind speed in the $(i)$(counts[i]) direction.")
            push!(uvars, uv)
        end
        uvals = []
        for i in eachindex(vals)
            u = unit(vals[i])
            v = ustrip(vals[i])
            sym = Symbol("c_v$i")
            c = (@constants $sym = v [unit = u])[1]
            push!(uvals, c)
        end
        eqs = Symbolics.scalarize(uvars .~ uvals)
        new(ODESystem(eqs, t, uvars, []; name=:constantwind), length(vals))
    end
end
function couple(mw::MeanWind, w::ConstantWind)
    # Create new systems so that the variables are correctly scoped.
    @named a = ODESystem(Equation[], ModelingToolkit.get_iv(mw.sys), [], [], systems=[mw.sys])
    @named b = ODESystem(Equation[], ModelingToolkit.get_iv(w.sys), [], [], systems=[w.sys])
    ConnectorSystem(
        Symbolics.scalarize(states(a) .~ states(b)),
        mw, w,
    )
end