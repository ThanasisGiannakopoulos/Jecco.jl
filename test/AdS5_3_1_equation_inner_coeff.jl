
using Test
using BenchmarkTools

using Jecco
using Jecco.AdS5_3_1

ABCS  = zeros(4)
AA    = zeros(2,2)
BB    = zeros(2,2)
CC    = zeros(2,2)
SS    = zeros(2)

svars    = Jecco.AdS5_3_1.SVars(ones(11)...)
fvars    = Jecco.AdS5_3_1.FVars(ones(37)...)
sdvars   = Jecco.AdS5_3_1.SdVars(ZeroPotential(), ones(70)...)
bdgvars  = Jecco.AdS5_3_1.BdGVars(ones(71)...)
phidvars = Jecco.AdS5_3_1.phidVars(ZeroPotential(), ones(72)...)
avars    = Jecco.AdS5_3_1.AVars(ZeroPotential(), ones(76)...)


@testset "S equation inner grid coefficients:" begin
    Jecco.AdS5_3_1.S_eq_coeff!(ABCS, svars, Inner())

    @test all(ABCS .≈ [6.0, 48.0, 133.42988060987634, 130.85976121975267])
end
@btime Jecco.AdS5_3_1.S_eq_coeff!($ABCS, $svars, Inner())

@testset "Fxy equations inner grid coefficients:" begin
    Jecco.AdS5_3_1.Fxy_eq_coeff!(AA, BB, CC, SS, fvars, Inner())

    @test all( AA .≈ [48.92907291226281 0.0; 0.0 18.0] )
    @test all( BB .≈ [811.6226411252798 -151.92523101186953; 119.40115727043062 41.42071634074196] )
    @test all( CC .≈ [9249.882276773546 -1318.4601133442727; 2930.339069731014 256.1897346300231] )
    @test all( SS .≈ [9577.64621761626, 3981.7746343640238])
end
@btime Jecco.AdS5_3_1.Fxy_eq_coeff!($AA, $BB, $CC, $SS, $fvars, Inner())

@testset "Sd equation inner grid coefficients:" begin
    Jecco.AdS5_3_1.Sd_eq_coeff!(ABCS, sdvars, Inner())

    @test all( ABCS .≈ [0.0, -880.7233124207306, -2544.311791437666, 279203.2839670398] )
end
@btime Jecco.AdS5_3_1.Sd_eq_coeff!($ABCS, $sdvars, Inner())

@testset "B2d equation inner grid coefficients:" begin
    Jecco.AdS5_3_1.B2d_eq_coeff!(ABCS, bdgvars, Inner())

    @test all( ABCS .≈ [0.0, -2642.169937262192, -9687.956436628037, -23245.124131500335] )
end
@btime Jecco.AdS5_3_1.B2d_eq_coeff!($ABCS, $bdgvars, Inner())

@testset "B1dGd equations inner grid coefficients:" begin
    Jecco.AdS5_3_1.B1dGd_eq_coeff!(AA, BB, CC, SS, bdgvars, Inner())

    @test all( AA .≈ [0.0 0.0; 0.0 0.0] )
    @test all( BB .≈ [-2642.169937262192 0.0; 0.0 -2642.169937262192] )
    @test all( CC .≈ [-15724.739986410723 -6036.783549782685; 14374.17230438983 -9687.956436628037] )
    @test all( SS .≈ [-143418.3032738444, -58153.507064913196] )
end
@btime Jecco.AdS5_3_1.B1dGd_eq_coeff!($AA, $BB, $CC, $SS, $bdgvars, Inner())

@testset "phid equations inner grid coefficients:" begin
    Jecco.AdS5_3_1.phid_eq_coeff!(ABCS, phidvars, Inner())

    @test all( ABCS .≈ [0.0, -587.1488749471532, -1565.7303331924084, 12974.144452661105] )
end
@btime Jecco.AdS5_3_1.phid_eq_coeff!($ABCS, $phidvars, Inner())

@testset "A equations inner grid coefficients:" begin
    Jecco.AdS5_3_1.A_eq_coeff!(ABCS, avars, Inner())

    @test all( ABCS .≈ [1321.084968631096, 7926.5098117865755, 7926.5098117865755, -832872.1280995568] )
end
@btime Jecco.AdS5_3_1.A_eq_coeff!($ABCS, $avars, Inner())

nothing