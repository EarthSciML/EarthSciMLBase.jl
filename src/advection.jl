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

A model component that represents the mean wind velocity, where
`pvars` is the partial dependent variables for the domain.
"""
function MeanWind(pvars...)
    uvars = meanwind_vars(t, pvars)
    ODESystem(Equation[], t, uvars, []; name=systemname(:meanwind))
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

function couple(c::CoupledSystem, _::Advection)::CoupledSystem
    @assert isa(c.domaininfo, DomainInfo) "The system must have initial and boundary conditions (i.e. DomainInfo) to add advection."

    # Add in a model component to allow the specification of the wind velocity.
    push!(c.systems, MeanWind(pvars(c.domaininfo)...))

    function f(sys::ModelingToolkit.PDESystem)
        eqs = advection(sys.dvs, c.domaininfo)
        operator_compose!(sys, eqs)
    end
    push!(c.pdefunctions, f)
    c
end
couple(a::Advection, c::CoupledSystem)::CoupledSystem = couple(c, a)

"""
$(SIGNATURES)

Construct a constant wind velocity model component with the given wind speed(s), which
should include units. For example, `ConstantWind(t, 1u"m/s", 2u"m/s")`.
"""
function ConstantWind(vals...)
    counts = ["st", "nd", "rd", "th", "th", "th", "th"]
    uvars = Num[]
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
    eqs = convert(Vector{Equation}, Symbolics.scalarize(uvars .~ uvals))
    ODESystem(eqs, t, uvars, []; name=systemname(:constantwind))
end

register_coupling(MeanWind(), ConstantWind()) do mw, w
    # Create new systems so that the variables are correctly scoped.
    @named a = ODESystem(Equation[], ModelingToolkit.get_iv(mw), [], [], systems=[mw])
    @named b = ODESystem(Equation[], ModelingToolkit.get_iv(w), [], [], systems=[w])
    ConnectorSystem(
        Symbolics.scalarize(states(a) .~ states(b)),
        mw, w,
    )
end