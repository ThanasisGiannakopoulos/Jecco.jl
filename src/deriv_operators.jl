
import Base: *, copyto!
import LinearAlgebra: mul!
using SparseArrays

abstract type AbstractDerivOperator{T,N} end

D1_21_weights() = [-1.0,  0.0, 1.0] ./ 2.0
D2_21_weights() = [ 1.0, -2.0, 1.0]

D1_42_weights() = [ 1.0, -8.0,   0.0,  8.0, -1.0] ./ 12.0
D2_42_weights() = [-1.0, 16.0, -30.0, 16.0, -1.0] ./ 12.0


struct FiniteDiffDeriv{T<:Real,N,T2,S} <: AbstractDerivOperator{T,N}
    derivative_order        :: Int
    approximation_order     :: Int
    dx                      :: T2
    len                     :: Int
    stencil_length          :: Int
    stencil_coefs           :: S
end

struct SpectralDeriv{T<:Real,N,S} <: AbstractDerivOperator{T,N}
    derivative_order        :: Int
    len                     :: Int
    D                       :: S
end

struct CenteredDiff{N} end

function CenteredDiff{N}(derivative_order::Int,
                         approximation_order::Int, dx::T,
                         len::Int) where {T<:Real,N}
    @assert approximation_order > 1 "approximation_order must be greater than 1."

    stencil_length = derivative_order + approximation_order - 1 +
        (derivative_order+approximation_order)%2

    # TODO: this can all be improved...

    if approximation_order == 2
        if derivative_order == 1
            weights = D1_21_weights()
        elseif derivative_order == 2
            weights = D2_21_weights()
        else
            error("derivative_order not implemented yet")
        end
    elseif approximation_order == 4
        if derivative_order == 1
            weights = D1_42_weights()
        elseif derivative_order == 2
            weights = D2_42_weights()
        else
            error("derivative_order not implemented yet")
        end
    else
        error("approximation_order not implemented yet")
    end

    stencil_coefs = (1/dx^derivative_order) .* weights

    FiniteDiffDeriv{T,N,T,typeof(stencil_coefs)}(derivative_order, approximation_order,
                                                 dx, len, stencil_length, stencil_coefs)
end

CenteredDiff(args...) = CenteredDiff{1}(args...)


struct ChebDeriv{N} end

function ChebDeriv{N}(derivative_order::Int, xmin::T, xmax::T, len::Int) where {T<:Real,N}
    x, Dx, Dxx = cheb(xmin, xmax, len)

    if derivative_order == 1
        D = Dx
    elseif derivative_order == 2
        D = Dxx
    else
        error("derivative_order not implemented")
    end

    SpectralDeriv{T,N,typeof(D)}(derivative_order, len, D)
end

ChebDeriv(args...) = ChebDeriv{1}(args...)


@inline function Base.getindex(A::FiniteDiffDeriv, i::Int, j::Int)
    N      = A.len
    coeffs = A.stencil_coefs
    mid    = div(A.stencil_length, 2) + 1

    # note: without the assumption of periodicity this needs to be adapted
    idx  = 1 + mod(j - i + mid - 1, N)

    if idx < 1 || idx > A.stencil_length
        return 0.0
    else
        return coeffs[idx]
    end
end

@inline Base.getindex(A::FiniteDiffDeriv, i::Int, ::Colon) =
    [A[i,j] for j in 1:A.len]

@inline Base.getindex(A::SpectralDeriv, i, j) = A.D[i,j]


# make FiniteDiffDeriv a callable struct, to compute derivatives at a given point
function (A::FiniteDiffDeriv{T,N,T2,S})(f::AbstractArray{T,M},
                                        idx::Vararg{Int,M}) where {T<:Real,N,T2,S,M}

    # make sure axis of differentiation is contained in the dimensions of f
    @assert N <= M

    coeffs = A.stencil_coefs
    mid = div(A.stencil_length, 2) + 1
    i   = idx[N] # point where derivative will be taken (with respect to the N-axis)

    sum_i = zero(T)

    if mid <= i <= (A.len-mid+1)
        @fastmath @inbounds for aa in 1:A.stencil_length
            i_circ = i - (mid - aa)
            I = Base.setindex(idx, i_circ, N)

            sum_i += coeffs[aa] * f[I...]
        end
    else
        @fastmath @inbounds for aa in 1:A.stencil_length
            # imposing periodicity
            i_circ = 1 + mod(i - (mid-aa) - 1, A.len)
            I = Base.setindex(idx, i_circ, N)

            sum_i += coeffs[aa] * f[I...]
        end
    end

    sum_i
end

function LinearAlgebra.mul!(df::AbstractVector, A::FiniteDiffDeriv, f::AbstractVector)
    @fastmath @inbounds for idx in eachindex(f)
        df[idx] = A(f,idx)
    end
    nothing
end

function LinearAlgebra.mul!(df::AbstractArray, A::FiniteDiffDeriv, f::AbstractArray)
    @fastmath @inbounds for idx in CartesianIndices(f)
        df[idx] = A(f,idx.I...)
    end
    nothing
end



# mul! done by standard matrix multiplication for Chebyshev differentiation
# matrices. This can also be done through FFT, but after some testing the FFT
# route actually seemed to be performing slower, so let's stick with this
function LinearAlgebra.mul!(df::AbstractVector, A::SpectralDeriv, f::AbstractVector)
    mul!(df, A.D, f)
end

# and now for Arrays
function LinearAlgebra.mul!(df::AbstractArray{T}, A::SpectralDeriv{T,N,S},
                            f::AbstractArray{T}) where {T,N,S}
    Rpre  = CartesianIndices(axes(f)[1:N-1])
    Rpost = CartesianIndices(axes(f)[N+1:end])

    _mul_loop!(df, A, f, Rpre, size(f,N), Rpost)
end

# this works as follows. suppose we are taking the derivative of a 4-dimensional
# array g, with coordinates w,x,y,z, and size
#
#   size(g) = (nw, nx, ny, nz)
#
# let's further suppose that we want the derivative along the x-direction (here,
# the second entry). this fixes
#
#   N = 2
#
# and then
#
#   Rpre  = CartesianIndices((nw,))
#   Rpost = CartesianIndices((ny,nz))
#
# now, the entry [Ipre,:,Ipost]
#
# slices *only* along the x-direction, for fixed Ipre and Ipost. Ipre and Ipost
# loop along Rpre and Rpost (respectively) without touching the x-direction. we
# further take care to loop in memory-order, since julia's arrays are faster in
# the first dimension.
#
# adapted from http://julialang.org/blog/2016/02/iteration
#
@noinline function _mul_loop!(df::AbstractArray{T}, A, f, Rpre, n, Rpost) where {T}
    @fastmath @inbounds for Ipost in Rpost
        @inbounds for i in 1:n
            @inbounds for Ipre in Rpre
                sum_i = zero(T)
                @inbounds @simd for j in 1:n
                    sum_i += A.D[i,j] * f[Ipre,j,Ipost]
                end
                df[Ipre,i,Ipost] = sum_i
            end
        end
    end
    nothing
end

function *(A::AbstractDerivOperator, x::AbstractArray)
    y = similar(x)
    mul!(y, A, x)
    y
end


# make SpectralDeriv a callable struct, to compute derivatives only at a given point.
function (A::SpectralDeriv{T,N,S})(f::AbstractArray{T,M},
                                   idx::Vararg{Int,M}) where {T<:Real,N,M,S}
    # make sure axis of differentiation is contained in the dimensions of f
    @assert N <= M

    i  = idx[N] # point where derivative will be taken (with respect to the N-axis)

    sum_i = zero(T)
    @fastmath @inbounds for ii in 1:A.len
        I = Base.setindex(idx, ii, N)
        sum_i += A.D[i,ii] * f[I...]
    end
    sum_i
end


# now for cross-derivatives. we assume that A acts on the first and B on the
# second axis of the x Matrix.

function (A::FiniteDiffDeriv{T,N1,T2,S})(B::FiniteDiffDeriv{T,N2,T2,S}, x::AbstractMatrix{T},
                                         i::Int, j::Int) where {T<:Real,T2,S,N1,N2}
    NA   = A.len
    NB   = B.len
    qA   = A.stencil_coefs
    qB   = B.stencil_coefs
    midA = div(A.stencil_length, 2) + 1
    midB = div(B.stencil_length, 2) + 1

    @assert( (NA, NB) == size(x) )
    @assert( N2 > N1 )

    sum_ij = zero(T)

    if midA <= i <= (NA-midA+1) && midB <= j <= (NB-midB+1)
        @fastmath @inbounds for jj in 1:B.stencil_length
            j_circ = j - (midB-jj)
            sum_i  = zero(T)
            @inbounds for ii in 1:A.stencil_length
                i_circ = i - (midA-ii)
                sum_i += qA[ii] * qB[jj] * x[i_circ,j_circ]
            end
            sum_ij += sum_i
        end
    else
        @fastmath @inbounds for jj in 1:B.stencil_length
            # imposing periodicity
            j_circ = 1 + mod(j - (midB-jj) - 1, NB)
            sum_i  = zero(T)
            @inbounds for ii in 1:A.stencil_length
                # imposing periodicity
                i_circ = 1 + mod(i - (midA-ii) - 1, NA)
                sum_i += qA[ii] * qB[jj] * x[i_circ,j_circ]
            end
            sum_ij += sum_i
        end
    end

    sum_ij
end

# and now for any Array.

function (A::FiniteDiffDeriv{T,N1,T2,S})(B::FiniteDiffDeriv{T,N2,T2,S},
                                         f::AbstractArray{T,M},
                                         idx::Vararg{Int,M}) where {T<:Real,T2,S,N1,N2,M}
    NA   = A.len
    NB   = B.len
    qA   = A.stencil_coefs
    qB   = B.stencil_coefs
    midA = div(A.stencil_length, 2) + 1
    midB = div(B.stencil_length, 2) + 1

    # make sure axes of differentiation are contained in the dimensions of f
    @assert N1 < N2 <= M

    # points where derivative will be taken (along their respective axes)
    i  = idx[N1]
    j  = idx[N2]

    sum_ij = zero(T)

    if midA <= i <= (NA-midA+1) && midB <= j <= (NB-midB+1)
        @fastmath @inbounds for jj in 1:B.stencil_length
            j_circ = j - (midB-jj)
            Itmp   = Base.setindex(idx, j_circ, N2)
            sum_i  = zero(T)
            @inbounds for ii in 1:A.stencil_length
                i_circ = i - (midA-ii)
                I      = Base.setindex(Itmp, i_circ, N1)
                sum_i += qA[ii] * qB[jj] * f[I...]
            end
            sum_ij += sum_i
        end
    else
        @fastmath @inbounds for jj in 1:B.stencil_length
            # imposing periodicity
            j_circ = 1 + mod(j - (midB-jj) - 1, NB)
            Itmp   = Base.setindex(idx, j_circ, N2)
            sum_i  = zero(T)
            @inbounds for ii in 1:A.stencil_length
                # imposing periodicity
                i_circ = 1 + mod(i - (midA-ii) - 1, NA)
                I      = Base.setindex(Itmp, i_circ, N1)
                sum_i += qA[ii] * qB[jj] * f[I...]
            end
            sum_ij += sum_i
        end
    end

    sum_ij
end


# Casting to matrix types

copyto!(M::AbstractMatrix{T}, A::SpectralDeriv) where {T<:Real} =
    copyto!(M, A.D)

function copyto!(M::AbstractMatrix{T}, A::FiniteDiffDeriv) where {T<:Real}
    for idx in CartesianIndices(M)
        M[idx] = A[idx.I...]
    end
    M
end

LinearAlgebra.Array(A::AbstractDerivOperator{T,N}) where {T,N} =
    copyto!(zeros(T, A.len, A.len), A)

SparseArrays.SparseMatrixCSC(A::FiniteDiffDeriv{T,N,T2,S}) where {T,N,T2,S} =
    copyto!(spzeros(T, A.len, A.len), A)
