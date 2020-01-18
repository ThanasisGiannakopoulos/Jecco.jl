
using Dates
using Printf

mutable struct TimeInfo{T}
    it  :: Int
    t   :: T
    dt  :: T
end
TimeInfo(it::Int, t::Real, dt::Real) = TimeInfo{typeof(t)}(it, t, dt)
TimeInfo() = TimeInfo(0, 0.0, 0.0)

function out_info(it::Integer, t::Real, f, label::String, info_every::Integer,
                  header_every::Integer)

    if it % header_every == 0
        println("--------------------------------------------------------------------------------")
        println("Iteration      Time |              \t $label")
        println("                    |   minimum      maximum")
        println("--------------------------------------------------------------------------------")
    end

    if it % info_every == 0
        @printf "%9d %9.3f | %9.4g    %9.4g\n" it t minimum(f) maximum(f)
    end

    nothing
end
