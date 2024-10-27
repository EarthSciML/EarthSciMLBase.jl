function _mtk_grid_func(sys_mtk, setp!)
    mtkf = eval(generate_function(sys_mtk)[2])
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
        @inbounds for j ∈ 1:size(umat, 2)
            col = view(umat, :, j)
            ddu = view(dumat, :, j)
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
function _mtk_scimlop(sys_mtk, setp!, u0, p)
    f = _mtk_grid_func(sys_mtk, setp!)
    nrows = length(unknowns(sys_mtk))
    u0mat = reshape(u0, nrows, :)
    f_op = FunctionOperator(f, u0mat, batch = true, p = p)
    A = I(nrows ÷ length(u0))
    t = TensorProductOperator(A, f_op)
    cache_operator(t, u0)
end

# Return a function to apply the MTK system to each column of u after reshaping to a matrix.
function mtk_grid_func(sys_mtk::ODESystem, domain::DomainInfo{T}, u0, p; jac=true,
        sparse=true, tgrad=jac, scimlop=false) where {T}
    setp! = coord_setter(sys_mtk, domain)

    f = scimlop ? _mtk_scimlop(sys_mtk, setp!, u0, p) : _mtk_grid_func(sys_mtk, setp!)

    ncells = length(grid(domain))
    kwargs = []
    if sparse
        b = repeat([length(unknowns(sys_mtk))], ncells)
        j = BlockBandedMatrix{T}(undef, b, b, (0, 0)) # Jacobian prototype
        push!(kwargs, :jac_prototype => j)
    end
    if jac
        jf = mtk_jac_grid_func(sys_mtk, setp!, ncells)
        push!(kwargs, :jac=>jf)
    end
    if tgrad
        tg = mtk_tgrad_grid_func(sys_mtk, setp!, ncells)
        push!(kwargs, :tgrad=>tg)
    end
    ODEFunction(f; kwargs...)
end

# Create a function to calculate the gridded Jacobian.
# ngrid is the number of grid cells.
function mtk_jac_grid_func(sys_mtk, setp!, ngrid)
    # Sparse is always dense because the block-banded-matrix is dense within the block.
    nrows = length(unknowns(sys_mtk))^2
    jac1 = eval(generate_jacobian(sys_mtk, sparse=false)[2])
    function jac(out, u, p, t) # In-place
        @inbounds for r ∈ 0:(ngrid-1)
            rng = r * nrows + 1
            rng = rng:(rng+nrows)
            _u = view(u, rng, rng)
            _out = view(out, rng, rng)
            setp!(p, r+1)
            jac1(_out, _u, p, t)
        end
    end
end

# Create a function to calculate the gridded time gradient.
# ngrid is the number of grid cells.
function mtk_tgrad_grid_func(sys_mtk, setp!, ngrid)
    # Sparse is always dense because the block-banded-matrix is dense within the block.
    nrows = length(unknowns(sys_mtk))
    tgrad1 = eval(generate_tgrad(sys_mtk, sparse=false)[2])
    function jac(out, u, p, t) # In-place
        @inbounds for r ∈ 0:ngrid
            rng = r * nrows + 1
            rng = rng:(rng+nrows)
            _u = view(u, rng)
            _out = view(out, rng)
            setp!(p, r+1)
            @inline tgrad1(_out, _u, p, t)
        end
    end
end