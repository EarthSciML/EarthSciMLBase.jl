using ArrayInterface
using LinearAlgebra
import LinearSolve as LS

abstract type AbstractBlockDiagonal{T} <: AbstractMatrix{T} end

struct BlockDiagonal{T, V <: AbstractArray{T, 3}, MA <: MapAlgorithm} <:
       AbstractBlockDiagonal{T}
    data::V
    n::Int
    alg::MA
end

block(B::BlockDiagonal, i) = view(B.data, :, :, i)
nblocks(B::BlockDiagonal) = size(B.data, 3)

"""
    BlockDiagonal(data)
    BlockDiagonal(data, ::MapAlgorithm)

Creates a block-diagonal matrix given a 3D array of the `data`.
The first two dimensions of `data` must be the same, so that the blocks are square.
"""
function BlockDiagonal(
        data::V, alg::MA) where {T, V <: AbstractArray{T, 3}, MA <: MapAlgorithm}
    rows = size(data, 1)
    cols = size(data, 2)
    @assert cols==rows "The blocks must be square"

    return BlockDiagonal{T, V, MA}(data, rows, alg)
end
BlockDiagonal(data) = BlockDiagonal(data, MapBroadcast())

function Base.size(B::AbstractBlockDiagonal)
    N = nblocks(B) * B.n
    return (N, N)
end
Base.size(B::AbstractBlockDiagonal, ::Int) = nblocks(B) * B.n

_getblock(n, i) = (i - 1) ÷ n + 1
_getr(n, i) = (i - 1) % n + 1

function Base.getindex(B::BlockDiagonal{T}, i::Integer, j::Integer) where {T}
    b = _getblock(B.n, i)
    if b != _getblock(B.n, j) # not in the same block
        return zero(T)
    end
    r = _getr(B.n, i)
    c = _getr(B.n, j)
    B.data[r, c, b]
end

function Base.setindex!(B::BlockDiagonal, v, i::Integer, j::Integer)
    b = _getblock(B.n, i)
    if b != _getblock(B.n, j) && v != 0 # not in the same block
        throw(BoundsError(B, [i, j]))
    end
    r = _getr(B.n, i)
    c = _getr(B.n, j)
    B.data[r, c, b] = v
end

ArrayInterface.fast_scalar_indexing(B::BlockDiagonal) = false

function Base.view(B::BlockDiagonal, r::AbstractRange)
    if r == diagind(B)
        ii = CartesianIndices((B.n, B.n))[1:(B.n + 1):(B.n * B.n)]
        return reshape(view(B.data, ii, :), :)
    end
    error("BlockDiagonal does not support range views of non-diagonal indices")
end

function Base.Matrix(B::BlockDiagonal{T}) where {T}
    A = zeros(T, size(B))
    for i in 1:nblocks(B)
        r = ((i - 1) * B.n + 1):(i * B.n)
        c = ((i - 1) * B.n + 1):(i * B.n)
        @views A[r, c] .= B.data[:, :, i]
    end
    return A
end

for op in (:(Base.inv), :(Base.similar), :(Base.copy), :(Base.deepcopy_internal))
    eval(quote
        $(op)(B::BlockDiagonal) = BlockDiagonal($op(B.data), B.n, B.alg)
    end)
end

for op! in (:(Base.copyto!),)
    eval(quote
        function $op!(A::BlockDiagonal, B::BlockDiagonal)
            @assert size(A.data)==size(B.data) "Number of blocks must match"
            @assert A.n==B.n "Block sizes must match"
            $op!(A.data, B.data)
        end
    end)
end

"""
The result of a LU factorization of a block diagonal matrix.
"""
struct BlockDiagonalLU{T, VF <: AbstractArray{T, 3}, VIP <: AbstractArray{Int64, 2}}
    factors::VF
    ipiv::VIP
    info::Int64
end

function ArrayInterface.lu_instance(B::AbstractBlockDiagonal)
    return BlockDiagonalLU(
        similar(B.data),
        similar(B.data, Int64, size(B.data, 1), size(B.data, 3)),
        0
    )
end

function block(B::BlockDiagonalLU, i)
    return @views(LU(B.factors[:, :, i], B.ipiv[:, i], B.info))
end
function block(factors, ipiv, info, i)
    return @views(LU(factors[:, :, i], ipiv[:, i], info))
end
function block(factors, ipiv, i)
    return @views(LU(factors[:, :, i], ipiv[:, i], 0))
end

nblocks(B::BlockDiagonalLU) = size(B.factors, 3)
Base.size(B::BlockDiagonalLU, ::Int) = nblocks(B) * size(B.factors, 1)

function LinearAlgebra.issuccess(F::BlockDiagonalLU; kwargs...)
    for b in 1:nblocks(F)
        if !LinearAlgebra.issuccess(block(F, b); kwargs...)
            return false
        end
    end
    return true
end

function LinearAlgebra.lu!(B::AbstractBlockDiagonal, args...; kwargs...)
    o = BlockDiagonalLU(
        B.data,
        similar(B.data, Int64, size(B.data, 1), size(B.data, 3)),
        0
    )
    o2 = _lu!(o, B, args...; kwargs...)
    return o2
end

function LinearAlgebra.lu(B::AbstractBlockDiagonal, args...; kwargs...)
    o = ArrayInterface.lu_instance(B)
    o2 = _lu!(o, B, args...; kwargs...)
    return o2
end

function _lu!(o::BlockDiagonalLU, B::AbstractBlockDiagonal, args...; kwargs...)
    function flu(i, args...; kwargs...)
        b = lu(view(B.data, :, :, i), args...; kwargs...)
        o.factors[:, :, i] .= b.factors
        o.ipiv[:, i] .= b.ipiv
        b.info
    end
    reduce_info(info1, info2) = abs(info1) > abs(info2) ? info1 : info2
    info = mapreduce(i -> flu(i, args...; kwargs...), reduce_info, 1:nblocks(B), init = 0)
    return BlockDiagonalLU(o.factors, o.ipiv, info)
end

# Fused LU factorization
function LS.generic_lufact!(
        A::BlockDiagonal, pivot::Union{NoPivot, RowMaximum, RowNonZero},
        ipiv; check = false)
    data = A.data
    function lufunc!(b)
        ll = LS.generic_lufact!(
            @view(data[:, :, b]), pivot, @view(ipiv[:, b]); check = check)
        view(ipiv, :, b) .+= (b - 1) * size(ipiv, 1) # Adjust pivot locations
        return ll.info
    end
    # Check for nonzero return codes
    reduce_info(info1, info2) = abs(info1) > abs(info2) ? info1 : info2

    info = AK.mapreduce(lufunc!, reduce_info, 1:nblocks(A), AK.get_backend(data);
        init = 0, neutral = 0)

    return BlockDiagonalLU(A.data, ipiv, info)
end
function LinearAlgebra.generic_lufact!(
        A::BlockDiagonal, pivot::Union{NoPivot, RowMaximum, RowNonZero}; check = false)
    ipiv = similar(A.data, Int64, size(A.data, 1), size(A.data, 3))
    return LS.generic_lufact!(copy(A), pivot, ipiv; check = check)
end

function LinearAlgebra.ldiv!(x::AbstractVecOrMat, A::BlockDiagonalLU, b::AbstractVecOrMat)
    @assert size(x)==size(b) "dimensions of x and b must match"
    @assert size(A, 1)==size(b, 1) "number of rows must match"
    function fldiv!(i, factors, ipiv, info, x, b)
        blck = block(factors, ipiv, info, i)
        n = size(factors, 1)
        rng = ((i - 1) * n + 1):(i * n)
        _x = view(x, rng, :)
        _b = view(b, rng, :)
        ldiv!(_x, blck, _b)
    end
    map_closure_to_range(
        fldiv!, 1:nblocks(A), MapKernel(), A.factors, A.ipiv, A.info, x, b)
    x
end

if Sys.isapple()
    using Metal
    # Copy GPU arrays to the CPU for ldiv! owing to the lack of a batch-fused ldiv! operator.
    function LinearAlgebra.ldiv!(x::T2, A::BlockDiagonalLU{T, <:MtlArray},
            b::T2) where {T, T2 <: AbstractVecOrMat}
        A2 = BlockDiagonalLU(Array(A.factors), Array(A.ipiv), A.info)
        b2 = Array(b)
        x2 = Array(x)
        LinearAlgebra.ldiv!(x2, A2, b2)
        x .= T2(x2)
        x
    end
end

function LinearAlgebra.:\(A::BlockDiagonalLU, b::T) where {T <: AbstractVector}
    @assert size(A, 1)==size(b, 1) "number of rows must match"
    o = similar(b, size(A, 1))
    function backslashf(i, factors, ipiv, info, b, o)
        blck = block(factors, ipiv, info, i)
        n = size(factors, 1)
        rng = ((i - 1) * n + 1):(i * n)
        _b = view(b, rng)
        o[rng] .= blck \ _b
    end
    map_closure_to_range(
        backslashf, 1:nblocks(A), MapBroadcast(), A.factors, A.ipiv, A.info, b, o)
    vcat(o...)
end

function Base.:+(B::BlockDiagonal, M::UniformScaling)
    plusf(i) = block(B, i) + M
    o = map_closure_to_range(plusf, 1:nblocks(B), B.alg)
    BlockDiagonal(cat(o..., dims = 3), B.n, B.alg)
end

function LinearAlgebra.mul!(C::AbstractVecOrMat, A::BlockDiagonal, B::AbstractVecOrMat)
    @assert size(A, 2)==size(B, 1) "Sizes must match"
    function mulf(i, C, A, B)
        blck = view(A, :, :, i)
        n = size(blck, 1)
        rng = ((i - 1) * n + 1):(i * n)
        mul!(view(C, rng, :), blck, view(B, rng, :))
    end
    map_closure_to_range(
        mulf, 1:nblocks(A), A.alg, C, A.data, B)
    return C
end

function LinearAlgebra.:*(A::BlockDiagonal, B::AbstractVector)
    C = similar(B)
    LinearAlgebra.mul!(C, A, B)
end
function LinearAlgebra.:*(A::BlockDiagonal, B::AbstractMatrix)
    C = similar(B)
    LinearAlgebra.mul!(C, A, B)
end
