function _mtk_grid_func(sys_mtk, mtkf, setp!)
    nrows = length(unknowns(sys_mtk))
    function f(du::AbstractVector, u::AbstractVector, p, t) # In-place
        umat = reshape(u, nrows, :)
        dumat = reshape(du, nrows, :)
        @inbounds for j ∈ 1:size(umat, 2)
            col = view(umat, :, j)
            ddu = view(dumat, :, j)
            setp!(p, j)
            mtkf(ddu, col, p, t)
        end
    end
    function f(du::AbstractMatrix, u::AbstractMatrix, p, t) # In-place
        @inbounds for j ∈ 1:size(u, 2)
            col = view(u, :, j)
            ddu = view(du, :, j)
            setp!(p, j)
            mtkf(ddu, col, p, t)
        end
    end
    # function f(u, p, t) # Out-of-place (Commented out because not tested.)
    #     umat = reshape(u, nrows, :)
    #     function ff(u, p, t, j)
    #         setp!(p, j)
    #         mtkf(u, p, t)
    #     end
    #     @inbounds @views mapreduce(jcol -> ff(jcol[2], p, t, jcol[1]), hcat, enumerate(eachcol(umat)))
    # end

    return f
end

# Create a SciMLOperator for the MTK system.
function _mtk_scimlop(sys_mtk, mtkf, setp!, u0, p)
    f = _mtk_grid_func(sys_mtk, mtkf, setp!)
    nrows = length(unknowns(sys_mtk))
    u0mat = reshape(u0, nrows, :)
    f_op = FunctionOperator(f, u0mat, batch = true, p = p)
    A = I(length(u0) ÷ nrows)
    t = TensorProductOperator(A, f_op)
    cache_operator(t, u0)
end

# Return a function to apply the MTK system to each column of u after reshaping to a matrix.
function mtk_grid_func(sys_mtk::ODESystem, domain::DomainInfo{T}, u0, p;
        sparse=true, tgrad=true, scimlop=false) where {T}
    setp! = coord_setter(sys_mtk, domain)

    mtkf = ODEFunction(sys_mtk, tgrad=tgrad, jac=true, sparse=sparse)

    f = scimlop ? _mtk_scimlop(sys_mtk, mtkf, setp!, u0[:], p) : _mtk_grid_func(sys_mtk, mtkf, setp!)

    ncells = reduce(*, length.(grid(domain)))
    nvars = size(u0, 1)
    single_jac_prototype = mtkf.jac_prototype
    if isnothing(single_jac_prototype)
        single_jac_prototype = Matrix{eltype(u0)}(undef, nvars, nvars)
    end
    jac_prototype = BlockDiagonal([similar(single_jac_prototype) for _ in 1:ncells])
    jf = mtk_jac_grid_func(sys_mtk, mtkf, setp!)

    kwargs = []
    if tgrad
        tg = mtk_tgrad_grid_func(sys_mtk, mtkf, setp!, ncells)
        push!(kwargs, :tgrad=>tg)
    end
    ODEFunction(f; jac_prototype=jac_prototype, jac=jf, kwargs...)
end

# Create a function to calculate the gridded Jacobian.
# ngrid is the number of grid cells.
function mtk_jac_grid_func(sys_mtk, mtkf, setp!)
    nvar = length(unknowns(sys_mtk))
    function jac(out, u, p, t) # In-place
        u = reshape(u, nvar, :)
        blks = blocks(out)
        for r ∈ 1:size(u, 2)
            _u = view(u, :, r)
            setp!(p, r)
            mtkf.jac(blks[r], _u, p, t)
        end
    end
end

# Create a function to calculate the gridded time gradient.
# ngrid is the number of grid cells.
function mtk_tgrad_grid_func(sys_mtk, mtkf, setp!, ngrid)
    nvar = length(unknowns(sys_mtk))
    function jac(out, u, p, t) # In-place
        u = reshape(u, nvar, :)
        blks = blocks(out)
        for r ∈ 1:size(u, 2)
            _u = view(u, :, r)
            setp!(p, r)
            mtkf.tgrad(blks[r], _u, p, t)
        end
    end
end

# TODO: Remove below after adding to BlockDiagonals.jl
struct BlockDiagonalLU{T}
    blocks::Vector{T}
end

function LinearAlgebra.issuccess(F::BlockDiagonalLU; kwargs...)
    for b in F.blocks
        if !LinearAlgebra.issuccess(b; kwargs...)
            return false
        end
    end
    return true
end

function ArrayInterface.lu_instance(A::BlockDiagonal)
    return BlockDiagonalLU([ArrayInterface.lu_instance(b) for b in blocks(A)])
end

function LinearAlgebra.lu!(B::BlockDiagonal, args...; kwargs...)
    BlockDiagonalLU([lu!(blk, args...; kwargs...) for blk in blocks(B)])
end

function LinearAlgebra.lu(B::BlockDiagonal, args...; kwargs...)
    BlockDiagonalLU([lu(blk, args...; kwargs...) for blk in blocks(B)])
end

function LinearAlgebra.ldiv!(x::AbstractVecOrMat, A::BlockDiagonalLU, b::AbstractVecOrMat; kwargs...)
    row_i = 1
    @assert size(x) == size(b) "dimensions of x and b must match"
    @assert mapreduce(a -> size(a, 1), +, A.blocks) == size(b, 1) "number of rows must match"
    for block in A.blocks
        nrow = size(block, 1)
        _x = view(x, row_i:(row_i + nrow - 1), :)
        _b = view(b, row_i:(row_i + nrow - 1), :)
        ldiv!(_x, block, _b; kwargs...)
        row_i += nrow
    end
    x
end

Base.deepcopy_internal(x::BlockDiagonal, dict::IdDict) =
     BlockDiagonal([copy(block) for block in x.blocks])

Base.copyto!(x::BlockDiagonal, y::BlockDiagonal) = copyto!.(x.blocks, y.blocks)
