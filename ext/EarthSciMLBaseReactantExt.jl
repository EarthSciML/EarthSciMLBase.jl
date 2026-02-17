module EarthSciMLBaseReactantExt

using EarthSciMLBase
import Reactant
using ModelingToolkit
import LinearSolve as LS
using LinearAlgebra
using ArrayInterface

# Monkey-patch: Reactant's @compile-based fill! fails on SubArray views of
# ConcretePJRTArray because the tracing code can't handle SubArray reindexing.
# Fall back to scalar indexing, which uses the direct CPU buffer pointer path.
function Base.fill!(x::SubArray{T, N, <:Reactant.ConcretePJRTArray}, val) where {T, N}
    v = convert(T, val)
    for I in eachindex(x)
        @inbounds x[I] = v
    end
    return x
end

function EarthSciMLBase.map_closure_to_range(f, range, ::EarthSciMLBase.MapReactant, args...)
    f2(i) = f(i, args...)
    map(f2, range)
end

function EarthSciMLBase.mapreduce_range(f, op, range, ::EarthSciMLBase.MapReactant, args...)
    f2(i) = f(i, args...)
    mapreduce(f2, op, range; init = 0)
end

# Override lu_instance so the LinearSolve cache is created with types matching
# what the Reactant batched LU factorization actually returns:
#   ipiv  → Matrix{Int64}  (converted to CPU in generic_lufact!)
#   perm  → ConcretePJRTArray{Int32}  (from compiled lu)
#   alg   → MapReactant
function ArrayInterface.lu_instance(
        B::EarthSciMLBase.BlockDiagonal{T, <:Reactant.ConcretePJRTArray}) where {T}
    n = size(B.data, 1)
    nblk = size(B.data, 3)
    return EarthSciMLBase.BlockDiagonalLU(
        similar(B.data),
        zeros(Int64, n, nblk),
        0,
        Reactant.to_rarray(zeros(Int32, n, nblk)),
        B.alg
    )
end

# Batched LU factorization using Reactant's native MLIR compilation.
# Instead of iterating over blocks, this calls lu() on the entire 3D array,
# which Reactant compiles to enzymexla.linalg_lu.
# The compiled function is cached in the MapAlgorithm to avoid recompilation.
function LS.generic_lufact!(
        A::EarthSciMLBase.BlockDiagonal{T, <:Reactant.ConcretePJRTArray},
        pivot::RowMaximum, ipiv; check = false) where {T}
    alg = A.alg
    if !haskey(alg.cache, :lu_compiled)
        function _batched_lu(data)
            F = lu(data)
            return (F.factors, F.ipiv, F.perm)
        end
        alg.cache[:lu_compiled] = Reactant.@compile _batched_lu(A.data)
    end
    factors, ipiv_r, perm = alg.cache[:lu_compiled](A.data)

    # Convert ipiv from Int32 (XLA default) to Int64 for BlockDiagonalLU compatibility
    ipiv_out = Int64.(Array(ipiv_r))

    return EarthSciMLBase.BlockDiagonalLU(factors, ipiv_out, 0, perm, alg)
end

# Batched ldiv! compiled to MLIR via @compile.
# Constructs a Reactant BatchedLU from the stored factors and perm, then uses
# Reactant's batched ldiv! which compiles to @opcall batch(_lu_solve_core, ...)
# with StableHLO triangular_solve operations.
# The compiled function is cached in the MapAlgorithm to avoid recompilation.
function LinearAlgebra.ldiv!(
        x::AbstractVector,
        A::EarthSciMLBase.BlockDiagonalLU{T, <:Reactant.ConcretePJRTArray},
        b::AbstractVector) where {T}
    alg = A.alg
    n = size(A.factors, 1)
    nblk = size(A.factors, 3)
    b_3d = Reactant.to_rarray(reshape(T.(b), n, 1, nblk))
    info = Reactant.to_rarray(zeros(eltype(A.perm), nblk))

    if !haskey(alg.cache, :solve_compiled)
        function _batched_solve(factors, perm, info, b_3d)
            F = Reactant.TracedLinearAlgebra.BatchedLU(factors, perm, perm, info)
            ldiv!(F, b_3d)
            return b_3d
        end
        alg.cache[:solve_compiled] = Reactant.@compile _batched_solve(A.factors, A.perm, info, b_3d)
    end

    alg.cache[:solve_compiled](A.factors, A.perm, info, b_3d)
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
