using ArrayInterface
using LinearAlgebra

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
struct BlockDiagonalLU{T, VF <: AbstractArray{T, 3}, VIP <: AbstractArray{Int64, 2},
    VI <: AbstractArray{Int64, 1}}
    factors::VF
    ipiv::VIP
    info::VI
end

function ArrayInterface.lu_instance(B::AbstractBlockDiagonal)
    return BlockDiagonalLU(
        similar(B.data),
        similar(B.data, Int64, size(B.data, 1), size(B.data, 3)),
        zeros(Int64, size(B.data, 3))
    )
end

function block(B::BlockDiagonalLU, i)
    return @views(LU(B.factors[:, :, i], B.ipiv[:, i], B.info[i]))
end
function block(factors, ipiv, info, i)
    return @views(LU(factors[:, :, i], ipiv[:, i], info[i]))
end
function block(factors, ipiv, i)
    return @views(LU(factors[:, :, i], ipiv[:, i], 0))
end

nblocks(B::BlockDiagonalLU) = size(B.factors, 3)
Base.size(B::BlockDiagonalLU, ::Int) = nblocks(B) * size(B.factors, 1)

"""
Set block i of B equal to lu.
"""
function setblock!(B::BlockDiagonalLU, i, lu::LU)
    @views begin
        B.factors[:, :, i] .= lu.factors
        B.ipiv[:, i] .= lu.ipiv
        B.info[i] = lu.info
    end
end

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
        zeros(Int64, size(B.data, 3))
    )
    _lu!(o, B, args...; kwargs...)
    return o
end

function LinearAlgebra.lu(B::AbstractBlockDiagonal, args...; kwargs...)
    o = ArrayInterface.lu_instance(B)
    _lu!(o, B, args...; kwargs...)
    return o
end

function _lu!(o::BlockDiagonalLU, B::AbstractBlockDiagonal, args...; kwargs...)
    function flu(i, data, factors, ipiv, info, args...; kwargs...)
        b = lu(view(data, :, :, i), args...; kwargs...)
        factors[:, :, i] .= b.factors
        ipiv[:, i] .= b.ipiv
        info[i] = b.info
    end
    map_closure_to_range( # Need to destructure o to work with AcceleratedKernels.jl
        flu, 1:nblocks(B), MapBroadcast(), B.data, o.factors, o.ipiv, o.info, args...; kwargs...)
    return o
end

function LinearAlgebra.ldiv!(
        x::AbstractVecOrMat, A::BlockDiagonalLU, b::AbstractVecOrMat; kwargs...)
    @assert size(x)==size(b) "dimensions of x and b must match"
    @assert size(A, 1)==size(b, 1) "number of rows must match"
    function fldiv!(i, factors, ipiv, info, x, b; kwargs...)
        blck = block(factors, ipiv, info, i)
        n = size(factors, 1)
        rng = ((i - 1) * n + 1):(i * n)
        _x = view(x, rng, :)
        _b = view(b, rng, :)
        ldiv!(_x, blck, _b; kwargs...)
    end
    map_closure_to_range(fldiv!, 1:nblocks(A), MapBroadcast(), A.factors, A.ipiv, A.info, x, b; kwargs...)
    x
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
    map_closure_to_range(backslashf, 1:nblocks(A), MapBroadcast(), A.factors, A.ipiv, A.info, b, o)
    vcat(o...)
end

function Base.:+(B::BlockDiagonal, M::UniformScaling)
    plusf(i) = block(B, i) + M
    o = map_closure_to_range(plusf, 1:nblocks(B), B.alg)
    BlockDiagonal(cat(o..., dims = 3), B.n, B.alg)
end
