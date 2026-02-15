module EarthSciMLBaseReactantExt

using EarthSciMLBase
import Reactant
using ModelingToolkit

function EarthSciMLBase.map_closure_to_range(f, range, ::EarthSciMLBase.MapReactant, args...)
    function _map(f, range, args...)
        f2(i) = f(i, args...)
        map(f2, range)
    end
    Reactant.@jit _map(f, range, args...)
end

function EarthSciMLBase.mapreduce_range(f, op, range, ::EarthSciMLBase.MapReactant, args...)
    function _mapreduce(range, args...)
        f2(i) = f(i, args...)
        out = map(f2, range)
        reduce(op, out, init = 0)
    end
    Reactant.@jit _mapreduce(range, args...)
end

function EarthSciMLBase.mtk_grid_func(
        sys_mtk::System, domain::EarthSciMLBase.DomainInfo{T, AT}, u0,
        alg::EarthSciMLBase.MapReactant,
        jac_type::JT = EarthSciMLBase.BlockDiagonalJacobian();
        sparse = false, tgrad = false, vjp = true) where {
        T, AT, JT <: EarthSciMLBase.JacobianType}
    sys_mtk, coord_args = EarthSciMLBase._prepare_coord_sys(sys_mtk, domain)

    mtkf_coord = EarthSciMLBase.build_coord_ode_function(sys_mtk, coord_args, alg)
    jac_coord = EarthSciMLBase.build_coord_jac_function(sys_mtk, coord_args, alg; sparse = sparse)

    nvars = length(unknowns(sys_mtk))
    jac_prototype = EarthSciMLBase.build_jacobian(jac_type, nvars, domain, alg, sparse)

    f,
    jf = let
        f = EarthSciMLBase._mtk_grid_func(sys_mtk, mtkf_coord, domain, alg)
        jf = EarthSciMLBase.mtk_jac_grid_func(sys_mtk, jac_coord, domain, jac_type, alg)
        p = MTKParameters(sys_mtk, ModelingToolkit.initial_conditions(sys_mtk))
        t = zero(eltype(domain))
        du = similar(u0) # TODO(CT): Is this allocation avoidable?
        f_compiled = Reactant.@compile f(du, u0, p, t)
        #jf_compiled = Reactant.@compile jf(jac_prototype, u0, p, t)
        f_compiled, jf #jf_compiled
    end

    kwargs = []
    if tgrad
        tgf = EarthSciMLBase.build_coord_tgrad_function(sys_mtk, coord_args, alg)
        tg = EarthSciMLBase.mtk_tgrad_grid_func(sys_mtk, tgf, domain, alg)
        push!(kwargs, :tgrad => tg)
    end
    if vjp
        vj = EarthSciMLBase.mtk_vjp_grid_func(sys_mtk, jac_coord, domain, alg)
        push!(kwargs, :vjp => vj)
    end
    ODEFunction(f; jac_prototype = jac_prototype, jac = jf, kwargs...), sys_mtk, coord_args
end

end # module
