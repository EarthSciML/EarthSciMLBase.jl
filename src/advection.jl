export Advection, MeanWind, ConstantWind

"""
$(SIGNATURES)

A model component that represents the mean wind velocity, where `t` is the independent variable
and `ndims` is the number of dimensions that wind is traveling in.
"""
struct MeanWind <: EarthSciMLODESystem
    sys::ODESystem
    function MeanWind(t, ndims) 
        uvars = (@variables u(t) v(t) w(t))[1:ndims]
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
    @assert length(pvs) <= 3 "Advection is only implemented for 3 or fewer dimensions."
    uvars = (@variables meanwind₊u(..) meanwind₊v(..) meanwind₊w(..))[1:length(pvs)]
    varsdims = Num[v for v ∈ vars]
    udims = Num[ui(pvs..., iv) for ui ∈ uvars]
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
    c += MeanWind(ivar(c.domaininfo), length(pvars(c.domaininfo)))
    
    function f(sys::ModelingToolkit.PDESystem)
        eqs = advection(sys.dvs, c.domaininfo)
        operator_compose!(sys, eqs)
    end
    ComposedEarthSciMLSystem(c.systems, c.domaininfo, [c.pdefunctions; f])
end
Base.:(+)(a::Advection, c::ComposedEarthSciMLSystem)::ComposedEarthSciMLSystem = c + a

"""
$(SIGNATURES)

Construct a constant wind velocity model component.
"""
struct ConstantWind <: EarthSciMLODESystem
    sys::ODESystem
    ndims

    function ConstantWind(t, vals...)
        @assert 0 < length(vals) <= 3 "Must specify between one and three wind component speeds."
        uvars = (@variables u(t) v(t) w(t))[1:length(vals)]
        eqs = Symbolics.scalarize(uvars .~ collect(vals))
        new(ODESystem(eqs, t, uvars, []; name=:constantwind), length(vals))
    end
end
function Base.:(+)(mw::MeanWind, w::ConstantWind)::ComposedEarthSciMLSystem
    eqs = [mw.sys.u ~ w.sys.u]
    w.ndims >= 2 ? push!(eqs, mw.sys.v ~ w.sys.v) : nothing
    w.ndims == 3 ? push!(eqs, mw.sys.w ~ w.sys.w) : nothing
    ComposedEarthSciMLSystem(ConnectorSystem(
        eqs,
        mw, w,
    ))
end
Base.:(+)(w::ConstantWind, mw::MeanWind)::ComposedEarthSciMLSystem = mw + w