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
            $op!.(A.data, B.data)
        end
    end)
end

"""
The result of a LU factorization of a block diagonal matrix.
"""
struct BlockDiagonalLU{T, V <: AbstractVector{T}, MA <: MapAlgorithm} <:
       AbstractBlockDiagonal{T}
    blocks::V
    n::Int
    alg::MA
end
BlockDiagonalLU(blocks, alg) = BlockDiagonalLU(blocks, size(blocks[1], 1), alg)

block(B::BlockDiagonalLU, i) = B.blocks[i]
nblocks(B::BlockDiagonalLU) = length(B.blocks)

function LinearAlgebra.issuccess(F::BlockDiagonalLU; kwargs...)
    for b in blocks(F)
        if !LinearAlgebra.issuccess(b; kwargs...)
            return false
        end
    end
    return true
end

function ArrayInterface.lu_instance(::MtlMatrix{Float32, Metal.PrivateStorage})
    LU{Float32, MtlMatrix{Float32, Metal.PrivateStorage},
        MtlVector{UInt32, Metal.PrivateStorage}}
end

function ArrayInterface.lu_instance(B::AbstractBlockDiagonal)
    return BlockDiagonalLU(
        [ArrayInterface.lu_instance(block(B, i)) for i in 1:nblocks(B)], B.n, B.alg)
end

function LinearAlgebra.lu!(B::AbstractBlockDiagonal, args...; kwargs...)
    o = BlockDiagonalLU(
        Vector{typeof(ArrayInterface.lu_instance(block(B, 1)))}(undef, nblocks(B)),
        B.n, B.alg)
    flu!(i) = o.blocks[i] = lu!(block(B, i), args...; kwargs...)
    map_closure_to_range(flu!, 1:nblocks(B), B.alg)
    o
end

function LinearAlgebra.lu(B::AbstractBlockDiagonal, args...; kwargs...)
    o = BlockDiagonalLU(
        Vector{typeof(ArrayInterface.lu_instance(block(B, 1)))}(
            undef, nblocks(B)),
        B.n, B.alg)
    flu(i) = o.blocks[i] = lu(block(B, i), args...; kwargs...)
    map_closure_to_range(flu, 1:nblocks(B), B.alg)
    o
end

function LinearAlgebra.ldiv!(
        x::AbstractVecOrMat, A::BlockDiagonalLU, b::AbstractVecOrMat; kwargs...)
    @assert size(x)==size(b) "dimensions of x and b must match"
    @assert size(A, 1)==size(b, 1) "number of rows must match"
    function fldiv!(i)
        blck = block(A, i)
        rng = ((i - 1) * A.n + 1):(i * A.n)
        _x = view(x, rng, :)
        _b = view(b, rng, :)
        ldiv!(_x, blck, _b; kwargs...)
    end
    map_closure_to_range(fldiv!, 1:nblocks(A), A.alg)
    x
end

function LinearAlgebra.:\(A::BlockDiagonalLU, b::T) where {T <: AbstractVector}
    @assert size(A, 1)==size(b, 1) "number of rows must match"
    o = Vector{T}(undef, length(A.blocks))
    function backslashf(i)
        blck = block(A, i)
        rng = ((i - 1) * A.n + 1):(i * A.n)
        _b = view(b, rng)
        o[i] = blck \ _b
    end
    map_closure_to_range(backslashf, 1:nblocks(A), A.alg)
    vcat(o...)
end

function Base.:+(B::BlockDiagonal, M::UniformScaling)
    plusf(i) = block(B, i) + M
    o = map_closure_to_range(plusf, 1:nblocks(B), B.alg)
    BlockDiagonal(cat(o..., dims = 3), B.n, B.alg)
end
