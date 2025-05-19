export Advection, MeanWind, ConstantWind

function meanwind_vars(t, domain::DomainInfo; prefix = "", multidim = false)
    uvars = []
    pvs = pvars(domain)
    δs = partialderivatives(domain)
    denominator = ModelingToolkit.get_unit(t)
    for (i, pv) in enumerate(pvs)
        sym = Symbol("$(prefix)v_$(pv)")
        if multidim # Multi-dimensional variable
            uv = (@variables $sym(..))[1]
        else
            uv = (@variables $sym(t))[1]
        end
        δ = δs[i].ts[i] # get the partial derivative operator for the variable
        unit = ModelingToolkit.get_unit(pv) / ModelingToolkit.get_unit(δ) / denominator
        # set metadata
        uv = add_metadata(uv, pv; exclude_default = true)
        uv = Symbolics.setmetadata(uv, ModelingToolkit.VariableUnit, unit)
        uv = Symbolics.setmetadata(uv, ModelingToolkit.VariableDescription,
            "Mean wind speed in the $(pv) direction.")
        push!(uvars, uv)
    end
    uvars
end

struct MeanWindCoupler
    sys::Any
end
"""
$(SIGNATURES)

A model component that represents the mean wind velocity, where
`pvars` is the partial dependent variables for the domain.
"""
function MeanWind(t, domain::DomainInfo)
    uvars = meanwind_vars(t, domain)
    ODESystem(Equation[], t, uvars, []; name = :MeanWind,
        metadata = Dict(:coupletype => MeanWindCoupler))
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
    uvars = meanwind_vars(iv, di; prefix = "MeanWind₊", multidim = true)
    varsdims = Num[v for v in vars]
    udims = Num[ui(iv, pvs...) for ui in uvars]
    δs = partialderivatives(di) # get partial derivative operators. May contain coordinate transforms.
    eqs = Equation[]
    for var in varsdims
        rhs = sum(vcat([-wind * δs[i](var) for (i, wind) in enumerate(udims)]))
        eq = Differential(iv)(var) ~ rhs
        push!(eqs, eq)
    end
    eqs
end

function couple2(c::CoupledSystem, _::Advection)::CoupledSystem
    dom = domain(c)

    # Add in a model component to allow the specification of the wind velocity.
    push!(c.systems, MeanWind(ivar(dom), dom))

    function f(sys::ModelingToolkit.PDESystem)
        eqs = advection(sys.dvs, dom)
        operator_compose!(sys, eqs)
    end
    push!(c.pdefunctions, f)
    c
end

struct ConstantWindCoupler
    sys::Any
end
"""
$(SIGNATURES)

Construct a constant wind velocity model component with the given wind speed(s), which
should include units. For example, `ConstantWind(t, 1u"m/s", 2u"m/s")`.
"""
function ConstantWind(t, vals...; name = :ConstantWind)
    counts = ["st", "nd", "rd", "th", "th", "th", "th"]
    uvars = Num[]
    for (i, val) in enumerate(vals)
        sym = Symbol("v_$i")
        desc = "Constant wind speed in the $(i)$(counts[i]) direction."
        if val isa DynamicQuantities.AbstractQuantity
            uv = only(@variables $sym(t), [unit = val / ustrip(val), description = desc])
        else
            uv = only(@variables $sym(t), [description = desc])
        end
        push!(uvars, uv)
    end
    uvals = []
    for i in eachindex(vals)
        v = ustrip(vals[i])
        sym = Symbol("c_v$i")
        if vals[i] isa DynamicQuantities.AbstractQuantity
            c = only(@constants $sym=v [unit = vals[i] / v])
        else
            c = only(@constants $sym = v)
        end
        push!(uvals, c)
    end
    eqs = convert(Vector{Equation}, Symbolics.scalarize(uvars .~ uvals))
    ODESystem(eqs, t, uvars, []; name,
        metadata = Dict(:coupletype => ConstantWindCoupler))
end

function couple2(mw::MeanWindCoupler, w::ConstantWindCoupler)
    mw, w = mw.sys, w.sys
    # Create new systems so that the variables are correctly scoped.
    @named a = ODESystem(Equation[], ModelingToolkit.get_iv(mw), [], [], systems = [mw])
    @named b = ODESystem(Equation[], ModelingToolkit.get_iv(w), [], [], systems = [w])
    ConnectorSystem(
        Symbolics.scalarize(unknowns(a) .~ unknowns(b)),
        mw, w
    )
end
