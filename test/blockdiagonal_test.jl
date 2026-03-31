using EarthSciMLBase: BlockDiagonal, block, MapKernel, MapReactant, ldiv_factors!
using LinearAlgebra
import SciMLOperators as SMO
using Test
using LinearSolve

@testset "LU" begin
    x = BlockDiagonal(rand(3, 3, 2))
    @test size(x) == (6, 6)
    lux1 = lu(Matrix(x))
    lux2 = lu(x)
    @test lux1.L[1:3, 1:3] ≈ block(lux2, 1).L
    @test lux1.U[1:3, 1:3] ≈ block(lux2, 1).U
    @test lux1.L[4:6, 4:6] ≈ block(lux2, 2).L
    @test lux1.U[4:6, 4:6] ≈ block(lux2, 2).U
    lux3 = lu!(x)
    @test lux1.L[1:3, 1:3] ≈ block(lux3, 1).L
    @test lux1.U[1:3, 1:3] ≈ block(lux3, 1).U
    @test lux1.L[4:6, 4:6] ≈ block(lux3, 2).L
    @test lux1.U[4:6, 4:6] ≈ block(lux3, 2).U

    @test LinearAlgebra.issuccess(lux1) == true

    x = BlockDiagonal(zeros(3, 3, 2))
    @test LinearAlgebra.issuccess(lu(x; check = false)) == false
end

@testset "ldiv!" begin
    @testset "Vector" begin
        x = BlockDiagonal(rand(3, 3, 2))
        y = rand(6)
        z1 = LinearAlgebra.ldiv!(similar(y), lu(x), y)
        z2 = LinearAlgebra.ldiv!(similar(y), lu(Matrix(x)), y)
        @test z1 ≈ z2
    end
end

@testset "backslash" begin
    @testset "Vector" begin
        x = BlockDiagonal(rand(3, 3, 2))
        y = rand(6)
        z1 = lu(x) \ y
        z2 = lu(Matrix(x)) \ y
        @test z1 ≈ z2
    end
end

@testset "Indexing" begin
    x = BlockDiagonal(reshape(1.0:18, 3, 3, 2))

    @testset "getindex" begin
        @test all(x[1:3, 1:3] .== reshape(1:9, 3, 3))
        @test all(x[4:6, 4:6] .== reshape(10:18, 3, 3))
        @test all(x[1:3, 4:6] .== 0.0)
        @test all(x[4:6, 1:3] .== 0.0)
    end

    @testset "setindex!" begin
        x = BlockDiagonal(rand(3, 3, 2))
        x[1:3, 1:3] .= reshape(1:9, 3, 3)
        x[4:6, 4:6] .= reshape(10:18, 3, 3)
        @test all(x.data .≈ reshape(1.0:18, 3, 3, 2))
    end
end

@testset "plus" begin
    x = BlockDiagonal(zeros(3, 3, 2))
    y = x + UniformScaling(1.0)
    @test y isa BlockDiagonal
    @test y ≈ I(6)
end

@testset "Diagonal View" begin
    x = BlockDiagonal(reshape(1:18, 3, 3, 2))
    y = Matrix(x)
    @test @view(x[diagind(x)]) == @view(y[diagind(y)])
end

@testset "Multiplication" begin
    x = BlockDiagonal(reshape(1:18, 3, 3, 2))

    @testset "Vector" begin
        b = rand(6)
        o = Matrix(x) * b
        o2 = similar(b)
        mul!(o2, x, b)
        o3 = x * b

        @test o ≈ o2
        @test o ≈ o3
        # Make sure the correct methods are being used.
        @test occursin("BlockDiagonal", string(which(mul!, typeof.((o2, x, b)))))
        @test occursin("BlockDiagonal", string(which(*, typeof.((x, b)))))
    end
    @testset "Matrix" begin
        b = rand(6, 3)

        o = Matrix(x) * b
        o2 = similar(b)
        mul!(o2, x, b)
        @test occursin("BlockDiagonal", string(which(mul!, typeof.((o2, x, b)))))
        o3 = x * b
        @test occursin("BlockDiagonal", string(which(*, typeof.((x, b)))))

        @test o ≈ o2
        @test o ≈ o3
    end
end

@testset "SciMLOperators" begin
    import SciMLOperators as SMO
    opfunc!(w, v, u, p, t) = mul!(w, u, v)
    opfunc!(v, u, p, t) = u * v
    x = BlockDiagonal(rand(3, 3, 2))
    b = rand(6)
    o = similar(b)
    op = SMO.FunctionOperator(opfunc!, b, o, u = x, p = nothing, t = 0.0, islinear = true)
    @test op(o, b, x, nothing, 0.0) ≈ mul!(o, x, b)
end

@testset "generic_lufact!" begin
    x = BlockDiagonal(rand(3, 3, 2))
    y = Matrix(x)
    ipiv = zeros(Int64, size(x.data, 1), size(x.data, 3))
    lx = LinearSolve.generic_lufact!(x, RowMaximum(), ipiv)

    ly = lu(y)
    @test ly.factors ≈ Matrix(BlockDiagonal(lx.factors))
    lx.ipiv[4:6] .+= 3 # The generic LU implementation indexes pivots based on each block.
    @test ly.ipiv == lx.ipiv[:]

    @testset "Singular" begin
        x = BlockDiagonal(zeros(3, 3, 2))
        lx = LinearSolve.generic_lufact!(x, RowMaximum(), ipiv; check = false)
        @test lx.info == 1
    end
end

if Sys.isapple()
    @testset "Metal" begin
        using Metal
        d = rand(Float32, 3, 3, 2)
        x = BlockDiagonal(MtlArray(d), MapKernel())
        y = Array(BlockDiagonal(d))

        ipiv = MtlArray(zeros(Int64, size(x.data, 1), size(x.data, 3)))
        lx = LinearSolve.generic_lufact!(x, RowMaximum(), ipiv)
        lx.ipiv[4:6] .+= 3 # The generic LU implementation indexes pivots based on each block.

        ly = lu(y)
        @test ly.factors ≈ Matrix(BlockDiagonal(Array(lx.factors)))
        @test ly.ipiv == Array(lx.ipiv)[:]

        @testset "ldiv!" begin
            d = rand(Float32, 3, 3, 2)
            x = BlockDiagonal(MtlArray(d))
            x2 = BlockDiagonal(Array(d))
            y = MtlArray(rand(Float32, 6))
            y2 = Array(y)
            z1 = LinearAlgebra.ldiv!(similar(y), lu(x), y)
            z2 = LinearAlgebra.ldiv!(similar(y2), lu(x2), y2)
            @test Array(z1) ≈ z2
        end
    end
end

# @testset "Reactant" begin
#     using Reactant
#     d = rand(Float32, 3, 3, 2)
#     x = BlockDiagonal(Reactant.to_rarray(d), MapReactant())
#     y = Array(BlockDiagonal(d))

#     ipiv = Reactant.to_rarray(zeros(Int64, size(x.data, 1), size(x.data, 3)))
#     lx = LinearSolve.generic_lufact!(x, RowMaximum(), ipiv)
#     lx.ipiv[4:6] .+= 3 # The generic LU implementation indexes pivots based on each block.

#     ly = lu(y)
#     @test ly.factors ≈ Matrix(BlockDiagonal(Array(lx.factors)))
#     @test ly.ipiv == Array(lx.ipiv)[:]

#     @testset "ldiv!" begin
#         d = rand(Float32, 3, 3, 2)
#         x = BlockDiagonal(MtlArray(d))
#         x2 = BlockDiagonal(Array(d))
#         y = MtlArray(rand(Float32, 6))
#         y2 = Array(y)
#         z1 = LinearAlgebra.ldiv!(similar(y), lu(x), y)
#         z2 = LinearAlgebra.ldiv!(similar(y2), lu(x2), y2)
#         @test Array(z1) ≈ z2
#     end
# end

@testset "ldiv_factors!" begin
    A = rand(3, 3)
    b = rand(3)

    F = lu(A)
    x = similar(b)
    ldiv_factors!(x, F.factors, F.ipiv, b)
    @test norm(A * x - b)≈0.0 atol=1e-10
    x2 = similar(b)
    LinearAlgebra.ldiv!(x2, F, b)
    @test x ≈ x2
end
