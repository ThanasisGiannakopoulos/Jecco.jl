
abstract type Filter{T,N} end

function KO_kernel(order::Int, sigma_diss::T) where {T<:Real}
    if order == 3
        kernel = -T[1, -4, 6, -4, 1]
        fac = sigma_diss / 16
    elseif order == 5
        kernel = T[1, -6, 15, -20, 15, -6, 1]
        fac = sigma_diss / 64
    else
        error("order = $order Kreiss-Oliger filter not yet implemented.")
    end

    rmul!(kernel, fac)

    # adding one to the middle entry so that it becomes a low-pass filter
    mid = div(length(kernel) + 1, 2)
    kernel[mid] += 1

    kernel
end

function exp_kernel!(f::Vector{T}, γ::T, α::T) where {T<:Real}
    M  = length(f) - 1
    f .= [ exp( -α * ( (i-1)/M )^(γ*M) ) for i in eachindex(f) ]
    nothing
end
function exp_kernel(dim::Int, γ::T, α::T) where {T<:Real}
    f  = Array{T}(undef, dim)
    exp_kernel!(f, γ, α)
    f
end


struct ConvFilter{T<:Real,N,A} <: Filter{T,N}
    kernel :: Vector{T}
    _cache :: A
end

struct FftFilter{T<:Real,N,A,FT<:FFTW.r2rFFTWPlan} <: Filter{T,N}
    kernel   :: Vector{T}
    fft_plan :: FT
    _cache   :: A
end

struct KO_Filter{N} end

function KO_Filter{N}(order::Int, sigma_diss::T, Nxx...) where {T<:Real,N}
    kernel = KO_kernel(order, sigma_diss)
    _cache = Array{T}(undef, Nxx...)
    ConvFilter{T,N,typeof(_cache)}(kernel, _cache)
end

KO_Filter(args...) = KO_Filter{1}(args...)


struct Exp_Filter{N} end

# α = -log(ϵ) where ϵ is the machine epsilon.
# for the standard choice of ϵ = 2^-52, α = 36.0437
function Exp_Filter{N}(γ::T, Nxx...; α::T=36.0437) where {T<:Real,N}
    kernel = exp_kernel(Nxx[N], γ, α)
    _cache = Array{T}(undef, Nxx...)

    # REDFT00 is the DCT-I, see http://www.fftw.org/fftw3_doc/Real-even_002fodd-DFTs-_0028cosine_002fsine-transforms_0029.html#Real-even_002fodd-DFTs-_0028cosine_002fsine-transforms_0029

    # let's use only one thread per FFTW so that they don't trample over each other
    FFTW.set_num_threads(1)

    # we want to use the DCT-I since this basis is precisely the one we have by
    # using the Gauss-Lobatto grid points
    fft_plan = FFTW.plan_r2r(_cache, FFTW.REDFT00, N)

    FftFilter{T,N,typeof(_cache),typeof(fft_plan)}(kernel, fft_plan, _cache)
end

Exp_Filter(args...) = Exp_Filter{1}(args...)


function convolution!(fout::AbstractVector{T}, f::AbstractVector{T},
                      g::AbstractVector{T}) where {T}
    f_len = length(f)
    g_len = length(g)
    mid   = div(g_len + 1, 2)

    @fastmath @inbounds for i in eachindex(f)
        sum_i = zero(T)

        if mid <= i <= (f_len - mid + 1)
            @inbounds for aa in 1:g_len
                i_circ = i - (mid - aa)
                sum_i += g[aa] * f[i_circ]
            end
        else
            @inbounds for aa in 1:g_len
                # imposing periodicity
                i_circ = 1 + mod(i - (mid-aa) - 1, f_len)
                sum_i += g[aa] * f[i_circ]
            end
        end
        fout[i] = sum_i
    end

    fout
end

# convolution along N axis
function convolution!(fout::AbstractArray{T,M}, f::AbstractArray{T,M},
                      g::AbstractVector{T}, N::Integer) where {T,M}
    # make sure axis of convolution is contained in the dimensions of f
    @assert N <= M

    f_len = size(f,N)
    g_len = length(g)
    mid   = div(g_len + 1, 2)

    @fastmath @inbounds for idx in CartesianIndices(f)
        i     = idx.I[N] # convolution to be done along this direction
        sum_i = zero(T)

        if mid <= i <= (f_len - mid + 1)
            @inbounds for aa in 1:g_len
                i_circ = i - (mid - aa)
                I = Base.setindex(idx.I, i_circ, N)
                sum_i += g[aa] * f[I...]
            end
        else
            @inbounds for aa in 1:g_len
                # imposing periodicity
                i_circ = 1 + mod(i - (mid-aa) - 1, f_len)
                I = Base.setindex(idx.I, i_circ, N)
                sum_i += g[aa] * f[I...]
            end
        end
        fout[idx] = sum_i
    end

    fout
end

function (filter::ConvFilter)(f::AbstractVector)
    @assert length(filter._cache) == length(f)
    convolution!(filter._cache, f, filter.kernel)
    copyto!(f, filter._cache)
end

function (filter::ConvFilter{T,M})(f::AbstractArray{T}) where {T,M}
    @assert length(filter._cache) == length(f)
    convolution!(filter._cache, f, filter.kernel, M)
    copyto!(f, filter._cache)
end


# see, eg, "Idempotent filtering in spectral and spectral element methods",
# Journal of Computational Physics 220 (2006) 41-58
function (filter::FftFilter)(f::AbstractVector)
    M = length(f) - 1

    copyto!(filter._cache, f)

    # compute the DCT-I of f
    mul!(f, filter.fft_plan, filter._cache)

    # in momentum space, act with the filter kernel [eq (2.5) of the paper above]
    # filter.kernel .* f
    @inbounds @simd for i in eachindex(f)
        # the division by 2*(N-1) is to invert the FFT, see below
        filter._cache[i] = 0.5/M * filter.kernel[i] * f[i]
    end

    # now go back to position space using the DCT-I. the division by 2*(N-1)
    # above comes from the normalization used. DCT-I is its own inverse up to a
    # constant, and FFTW defines it as such, cf:
    # http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029
    mul!(f, filter.fft_plan, filter._cache)
end

# filter along N axis
function (filter::FftFilter{T,N})(f::AbstractArray{T,M}) where {T,N,M}
    # make sure axis of filtering is contained in the dimensions of f
    @assert N <= M
    copyto!(filter._cache, f)

    Rpre  = CartesianIndices(size(f)[1:N-1])
    Rpost = CartesianIndices(size(f)[N+1:end])

    # use a function barrier here to separate the type-unstable portion
    _fftfilter!(f, filter._cache, filter.kernel, filter.fft_plan, Rpre,
                size(f, N), Rpost)
end

# Multidimensional algorithm adapted from
# http://julialang.org/blog/2016/02/iteration
function _fftfilter!(fout, f, kernel, fft_plan::FFTW.Plan, Rpre, n, Rpost)
    # compute the DCT-I of f
    mul!(fout, fft_plan, f)

    @fastmath @inbounds for Ipost in Rpost
        @inbounds for i in 1:n
            @inbounds @simd for Ipre in Rpre
                f[Ipre,i,Ipost] = 0.5/(n-1) * kernel[i] * fout[Ipre,i,Ipost]
            end
        end
    end

    # go back to position space using the DCT-I
    mul!(fout, fft_plan, f)
end
