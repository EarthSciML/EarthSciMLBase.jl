using EarthSciMLBase
using LinearAlgebra
using Test

@testset "LU" begin
    x = BlockDiagonal([rand(3, 3), rand(3, 3)])
    lux1 = lu(Matrix(x))
    lux2 = lu(x)
    lux3 = lu!(x)
    @test all([b1.L ≈ b2.L && b1.U ≈ b2.U for (b1, b2) in zip(lux2.blocks, lux3.blocks)])
    @test lux1.L[1:3, 1:3] ≈ lux2.blocks[1].L
    @test lux1.U[1:3, 1:3] ≈ lux2.blocks[1].U
    @test lux1.L[4:6, 4:6] ≈ lux2.blocks[2].L
    @test lux1.U[4:6, 4:6] ≈ lux2.blocks[2].U
end

@testset "ldiv!" begin
    @testset "Vector" begin
        x = BlockDiagonal([rand(3, 3), rand(3, 3)])
        y = rand(6)
        z1 = LinearAlgebra.ldiv!(similar(y), lu(x), y)
        z2 = LinearAlgebra.ldiv!(similar(y), lu(Matrix(x)), y)
        @test z1 ≈ z2
    end

    @testset "Matrix" begin
        x = BlockDiagonal([rand(3, 3), rand(3, 3)])
        y = rand(6, 2)
        z1 = LinearAlgebra.ldiv!(similar(y), lu(x), y)
        z2 = LinearAlgebra.ldiv!(similar(y), lu(Matrix(x)), y)
        @test z1 ≈ z2
    end
end

@testset "Indexing" begin
    x = BlockDiagonal([reshape(1:9, 3, 3), reshape(10:18, 3, 3)])

    @testset "getindex" begin
        @test all(x[1:3, 1:3] .== reshape(1:9, 3, 3))
        @test all(x[4:6, 4:6] .== reshape(10:18, 3, 3))
        @test all(x[1:3, 4:6] .== 0.0)
        @test all(x[4:6, 1:3] .== 0.0)
    end

    @testset "setindex!" begin
        x = BlockDiagonal([rand(3, 3), rand(3, 3)])
        x[1:3, 1:3] .= reshape(1:9, 3, 3)
        x[4:6, 4:6] .= reshape(10:18, 3, 3)
        @test all(x.blocks .≈ [reshape(1:9, 3, 3), reshape(10:18, 3, 3)])
    end
end
