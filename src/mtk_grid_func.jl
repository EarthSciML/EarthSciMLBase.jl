function _mtk_grid_func(sys_mtk, mtkf, setp!)
    nrows = length(unknowns(sys_mtk))
    function f(du::AbstractVector, u::AbstractVector, p, t) # In-place
        u = reshape(u, nrows, :)
        du = reshape(du, nrows, :)
        for j ∈ 1:size(u, 2)
            col = view(u, :, j)
            ddu = view(du, :, j)
            setp!(p, j)
            mtkf(ddu, col, p, t)
        end
        nothing
    end
    function f(u, p, t) # Out-of-place
        u = reshape(u, nrows, :)
        function ff(u, p, t, j)
            setp!(p, j)
            mtkf(u, p, t)
        end
        du = @views mapreduce(jcol -> ff(jcol[2], p, t, jcol[1]), hcat, enumerate(eachcol(u)))
        reshape(du, :)
    end

    return f
end

# Return a function to apply the MTK system to each column of u after reshaping to a matrix.
function mtk_grid_func(sys_mtk::ODESystem, domain::DomainInfo{T}, u0, p;
    sparse=true, tgrad=true, vjp=true) where {T}
    setp! = coord_setter(sys_mtk, domain)

    mtkf = ODEFunction(sys_mtk, tgrad=tgrad, jac=true, sparse=sparse)

    f = _mtk_grid_func(sys_mtk, mtkf, setp!)

    ncells = reduce(*, length.(grid(domain)))
    nvars = length(unknowns(sys_mtk))
    single_jac_prototype = mtkf.jac_prototype
    if isnothing(single_jac_prototype)
        single_jac_prototype = Matrix{eltype(u0)}(undef, nvars, nvars)
    end
    jac_prototype = BlockDiagonal([similar(single_jac_prototype) for _ in 1:ncells])
    jf = mtk_jac_grid_func(sys_mtk, mtkf, setp!)

    kwargs = []
    if tgrad
        tg = mtk_tgrad_grid_func(sys_mtk, mtkf, setp!)
        push!(kwargs, :tgrad => tg)
    end
    if vjp
        vj = mtk_vjp_grid_func(sys_mtk, mtkf, setp!)
        push!(kwargs, :vjp => vj)
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
        nothing
    end
    function jac(u, p, t) # Out-of-place
        u = reshape(u, nvar, :)
        BlockDiagonal([
            begin
                _u = view(u, :, r)
                setp!(p, r)
                mtkf.jac(_u, p, t)
            end for r ∈ 1:size(u, 2)
        ])
    end
end

# Create a function to calculate the gridded time gradient.
# ngrid is the number of grid cells.
function mtk_tgrad_grid_func(sys_mtk, mtkf, setp!)
    nvar = length(unknowns(sys_mtk))
    function tgrad(out, u, p, t) # In-place
        u = reshape(u, nvar, :)
        # FIXME(CT): I believe the tgrad output should be a vector and not a clone of the Jacobian.
        # This may be a bug in the DifferentialEquations.jl interface.
        blks = blocks(out)
        for r ∈ 1:size(u, 2)
            _u = view(u, :, r)
            setp!(p, r)
            mtkf.tgrad(blks[r], _u, p, t)
        end
    end
end

function mtk_vjp_grid_func(sys_mtk, mtkf, setp!)
    nvar = length(unknowns(sys_mtk))
    function vjp(vJ, v, u, p, t)
        u = reshape(u, nvar, :)
        J = Matrix{eltype(u)}(undef, nvar, nvar)
        for r ∈ 1:size(u, 2)
            _u = view(u, :, r)
            vJr = view(vJ, (r-1)*nvar+1:r*nvar, :)
            vr = view(v, (r-1)*nvar+1:r*nvar, :)
            setp!(p, r)
            mtkf.jac(J, _u, p, t)
            mul!(vJr, J', vr)
        end
    end
    function vjp(v, u, p, t)
        u = reshape(u, nvar, :)
        vcat([
            begin
                _u = view(u, :, r)
                vr = view(v, (r-1)*nvar+1:r*nvar, :)
                setp!(p, r)
                J = mtkf.jac(_u, p, t)
                J' * vr
            end for r ∈ 1:size(u, 2)
        ]...)
    end
    return vjp
end
