using ArrayInterface
using LinearAlgebra

abstract type AbstractBlockDiagonal{T} <: AbstractMatrix{T} end

blocks(B::AbstractBlockDiagonal) = B.blocks

struct BlockDiagonal{T,V<:AbstractMatrix{T}} <: AbstractBlockDiagonal{T}
    blocks::Vector{V}
    n::Int
end

"""
    BlockDiagonal(blocks)

Creates a block-diagonal matrix given a vector of the `blocks`.
The blocks must all be the same size and square.
"""
function BlockDiagonal(blocks::Vector{V}) where {T,V<:AbstractMatrix{T}}
    @assert length(blocks) > 0 "blocks must be non-empty"
    rows = size.(blocks, (1,))
    @assert all(diff(rows) .== 0) "All blocks must have the same number of rows"
    cols = size.(blocks, (2,))
    @assert all(diff(cols) .== 0) "All blocks must have the same number of columns"
    @assert cols[1] == rows[1] "All blocks must be square"

    return BlockDiagonal{T,V}(blocks, rows[1])
end

function Base.size(B::AbstractBlockDiagonal)
    N = length(B.blocks) * B.n
    return (N, N)
end
Base.size(B::AbstractBlockDiagonal, ::Int) = length(B.blocks) * B.n

_getblock(n, i) = (i - 1) ÷ n + 1
_getr(n, i) = (i - 1) % n + 1

function Base.getindex(B::BlockDiagonal{T}, i::Integer, j::Integer) where {T}
    b = _getblock(B.n, i)
    if b != _getblock(B.n, j) # not in the same block
        return zero(T)
    end
    r = _getr(B.n, i)
    c = _getr(B.n, j)
    B.blocks[b][r, c]
end

function Base.setindex!(B::BlockDiagonal, v, i::Integer, j::Integer)
    b = _getblock(B.n, i)
    if b != _getblock(B.n, j) && v != 0 # not in the same block
        throw(BoundsError(B, [i, j]))
    end
    r = _getr(B.n, i)
    c = _getr(B.n, j)
    B.blocks[b][r, c] = v
end

function Base.Matrix(B::BlockDiagonal{T}) where {T}
    A = zeros(T, size(B))
    for i in 1:length(B.blocks)
        A[(i-1)*B.n+1:i*B.n, (i-1)*B.n+1:i*B.n] .= B.blocks[i]
    end
    return A
end

for op in (:(Base.inv), :(Base.similar), :(Base.copy), :(Base.deepcopy_internal))
    eval(quote
        $(op)(B::BlockDiagonal) = BlockDiagonal($op.(blocks(B)), B.n)
    end)
end

for op! in (:(Base.copyto!),)
    eval(quote
        function $op!(A::BlockDiagonal, B::BlockDiagonal)
            @assert length(A.blocks) == length(B.blocks) "Number of blocks must match"
            @assert A.n == B.n "Block sizes must match"
            $op!.(blocks(A), blocks(B))
        end
    end)
end

"""
The result of a LU factorization of a block diagonal matrix.
"""
struct BlockDiagonalLU{T} <: AbstractBlockDiagonal{T}
    blocks::Vector{T}
    n::Int
end
BlockDiagonalLU{T}(blocks::Vector{T}) where {T} = BlockDiagonalLU{T}(blocks, size(blocks[1], 1))

function LinearAlgebra.issuccess(F::BlockDiagonalLU; kwargs...)
    for b in blocks(F)
        if !LinearAlgebra.issuccess(b; kwargs...)
            return false
        end
    end
    return true
end

function ArrayInterface.lu_instance(B::AbstractBlockDiagonal)
    return BlockDiagonalLU([ArrayInterface.lu_instance(b) for b in blocks(B)], B.n)
end

function LinearAlgebra.lu!(B::AbstractBlockDiagonal, args...; kwargs...)
    o = BlockDiagonalLU(Vector{typeof(ArrayInterface.lu_instance(blocks(B)[1]))}(undef, length(blocks(B))), B.n)
    Threads.@threads for i in eachindex(blocks(B))
        o.blocks[i] = lu!(blocks(B)[i], args...; kwargs...)
    end
    o
end

function LinearAlgebra.lu(B::AbstractBlockDiagonal, args...; kwargs...)
    o = BlockDiagonalLU(Vector{typeof(ArrayInterface.lu_instance(blocks(B)[1]))}(undef, length(blocks(B))), B.n)
    Threads.@threads for i in eachindex(blocks(B))
        o.blocks[i] = lu(blocks(B)[i], args...; kwargs...)
    end
    o
end

function LinearAlgebra.ldiv!(x::AbstractVecOrMat, A::BlockDiagonalLU, b::AbstractVecOrMat; kwargs...)
    @assert size(x) == size(b) "dimensions of x and b must match"
    @assert size(A, 1) == size(b, 1) "number of rows must match"
    Threads.@threads for i in eachindex(blocks(A))
        block = blocks(A)[i]
        rng = ((i-1)*A.n+1):(i*A.n)
        _x = view(x, rng, :)
        _b = view(b, rng, :)
        ldiv!(_x, block, _b; kwargs...)
    end
    x
end

function LinearAlgebra.:\(A::BlockDiagonalLU, b::T) where {T<:AbstractVector}
    @assert size(A, 1) == size(b, 1) "number of rows must match"
    o = Vector{T}(undef, length(blocks(A)))
    Threads.@threads for i in eachindex(blocks(A))
        block = blocks(A)[i]
        rng = ((i-1)*A.n+1):(i*A.n)
        _b = view(b, rng)
        o[i] = block \ _b
    end
    vcat(o...)
end

function Base.:+(B::BlockDiagonal, M::UniformScaling)
    return BlockDiagonal([block + M for block in blocks(B)])
end
