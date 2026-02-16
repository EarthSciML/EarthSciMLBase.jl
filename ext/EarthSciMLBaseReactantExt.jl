module EarthSciMLBaseReactantExt

using EarthSciMLBase
import Reactant
using ModelingToolkit
import LinearSolve as LS
using LinearAlgebra

function EarthSciMLBase.map_closure_to_range(f, range, ::EarthSciMLBase.MapReactant, args...)
    f2(i) = f(i, args...)
    map(f2, range)
end

function EarthSciMLBase.mapreduce_range(f, op, range, ::EarthSciMLBase.MapReactant, args...)
    f2(i) = f(i, args...)
    mapreduce(f2, op, range; init = 0)
end

# Batched LU factorization using Reactant's native MLIR compilation.
# Instead of iterating over blocks, this calls lu() on the entire 3D array,
# which Reactant compiles to enzymexla.linalg_lu.
function LS.generic_lufact!(
        A::EarthSciMLBase.BlockDiagonal{T, <:Reactant.ConcretePJRTArray},
        pivot::RowMaximum, ipiv; check = false) where {T}
    function _batched_lu(data)
        F = lu(data)
        return (F.factors, F.ipiv, F.perm)
    end
    factors, ipiv_r, perm = Reactant.@jit _batched_lu(A.data)

    # Convert ipiv from Int32 (XLA default) to Int64 for BlockDiagonalLU compatibility
    ipiv_out = Int64.(Array(ipiv_r))

    return EarthSciMLBase.BlockDiagonalLU(factors, ipiv_out, 0, perm)
end

# Batched ldiv! compiled to MLIR via @jit.
# Constructs a Reactant BatchedLU from the stored factors and perm, then uses
# Reactant's batched ldiv! which compiles to @opcall batch(_lu_solve_core, ...)
# with StableHLO triangular_solve operations.
function LinearAlgebra.ldiv!(
        x::AbstractVector,
        A::EarthSciMLBase.BlockDiagonalLU{T, <:Reactant.ConcretePJRTArray},
        b::AbstractVector) where {T}
    n = size(A.factors, 1)
    nblk = size(A.factors, 3)
    b_3d = Reactant.to_rarray(reshape(T.(b), n, 1, nblk))
    info = Reactant.to_rarray(zeros(eltype(A.perm), nblk))

    function _batched_solve(factors, perm, info, b_3d)
        F = Reactant.TracedLinearAlgebra.BatchedLU(factors, perm, perm, info)
        ldiv!(F, b_3d)
        return b_3d
    end

    Reactant.@jit _batched_solve(A.factors, A.perm, info, b_3d)
    x .= reshape(Array(b_3d), :)
    return x
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
