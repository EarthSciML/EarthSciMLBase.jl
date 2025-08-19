using EarthSciMLBase: BlockDiagonal, block
using LinearAlgebra
using Test

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
    @test LinearAlgebra.issuccess(lu(x; check=false)) == false
end

@testset "ldiv!" begin
    @testset "Vector" begin
        x = BlockDiagonal(rand(3, 3, 2))
        y = rand(6)
        z1 = LinearAlgebra.ldiv!(similar(y), lu(x), y)
        z2 = LinearAlgebra.ldiv!(similar(y), lu(Matrix(x)), y)
        @test z1 ≈ z2
    end

    @testset "Matrix" begin
        x = BlockDiagonal(rand(3, 3, 2))
        y = rand(6, 2)
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
