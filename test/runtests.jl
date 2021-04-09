using Test

using Jecco
using Jecco.AdS5_3_1
import Jecco.AdS5_3_1.getA
import Jecco.AdS5_3_1.getxi

@testset "All tests:" begin

    @time @testset "derivative operators tests:" begin include("deriv_operators_tests.jl") end

    @time @testset "grid tests:" begin include("grid_tests.jl") end

    @time @testset "output tests:" begin include("output_tests.jl") end

    @time @testset "filtering tests:" begin include("filtering_tests.jl") end

    @time @testset "AdS5_3_1 inner grid coefficients tests:" begin
        include("AdS5_3_1_equation_inner_coeff.jl")
    end

    @time @testset "AdS5_3_1 outer grid coefficients tests:" begin
        include("AdS5_3_1_equation_outer_coeff.jl")
    end

    @time @testset "AdS5_3_1 homogeneous black brane tests:" begin
        include("AdS5_3_1_homog_Bbrane.jl")
    end

end

nothing
