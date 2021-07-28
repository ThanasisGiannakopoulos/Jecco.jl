#This script will set up and solve the equations, in fourier space, for the gravitational wave
#perturbation h as a function of time. We solve (Dtt+k^2)h(t,k)=T(t,k).

#TODO: The integrator does not evolve in time, always spits dh = 0, fix it.

import Base.Threads.@threads
using FFTW
using LinearAlgebra
using Interpolations
using OrdinaryDiffEq
#Transform of only real functions. Gives back a matrix of 2 component vector for
# the momenta k and the matrix for the mode coefficients fk.
abstract type Parameters end

struct param{T<:Interpolations.GriddedInterpolation, TP<:Real} <: Parameters
    px  :: T
    pxy :: T
    py  :: T
    pz  :: T
    x   :: Array{TP,1}
    y   :: Array{TP,1}
    kx  :: Array{TP,2}
    ky  :: Array{TP,2}
    dt  :: TP
    tol :: TP
end

function get_evol_variable(u::Array{T,3}, u_t::Array{T,3}) where {T<:Complex}
    N      = length(u)
    u_evol = im.*zeros(N)
    u_evol = reshape(u, N)
    append!(u_evol, reshape(u_t, N))

    u_evol
end

function get_tensors(u_evol::Array{T,1}, Nkx::Int, Nky::Int) where {T<:Complex}
    Ntot     = length(u_evol)
    idxsplit = Int(Ntot/2)
    h        = reshape(u_evol[1:idxsplit],Nkx,Nky,4)
    h_t      = reshape(u_evol[idxsplit+1:end],Nkx,Nky,4)

    h, h_t
end

function Fourier_Transform_2D(f::Array{T,2}) where {T<:Real}
    Nx, Ny = size(f)
    plan = plan_rfft(f)

    fk   = 1/(Nx*Ny).*(plan * f)
end

function Inverse_Fourier_Transform_2D(u::Array{T,1}, Nkx::Int, Nky::Int, Nx::Int) where {T<:Complex}
    fk, fk_t   = get_tensors(u, Nkx, Nky)
    _, Ny, n = size(fk)
    if n != 4
        @warn "There should be 4 components for the h and hd tensors"
        return
    elseif Ny != Nky
        @warn "Missmatch between the y grid and the number of ky"
    end
    plan = plan_irfft(fk[:,:,1], Nx)
    f    = zeros(Nx, Ny, n)
    f_t  = zeros(Nx, Ny, n)
    for i in 1:n
        f[:,:,i] = Nx*Ny .*(plan * @view fk[:,:,i])
        f_t[:,:,i] = Nx*Ny .*(plan * @view fk_t[:,:,i])
    end

    f, f_t
end

#We can print the physical grid and the kx ky plane should be easy computable from it
function output_GW(outdir::String, h::Array{T,3}, h_t::Array{T,3}, chart2D::Chart,
                                    tinfo::Jecco.TimeInfo) where{T<:Real}
    if tinfo.it == 0
        try run(`mkdir $outdir`) catch end
    end

    perturbation = (
            Jecco.Field("hxx", view(h, :, :, 1), chart2D),
            Jecco.Field("hxy", view(h, :, :, 2), chart2D),
            Jecco.Field("hyy", view(h, :, :, 3), chart2D),
            Jecco.Field("hzz", view(h, :, :, 4), chart2D),
            Jecco.Field("hdxx", view(h_t, :, :, 1), chart2D),
            Jecco.Field("hdxy", view(h_t, :, :, 2), chart2D),
            Jecco.Field("hdyy", view(h_t, :, :, 3), chart2D),
            Jecco.Field("hdzz", view(h_t, :, :, 4), chart2D),
    )
    pert_writer  = Jecco.Output(outdir, "perturbation_", tinfo)
    pert_writer(perturbation)
end

function initial_conditions(ff::VEVTimeSeries, kx::Array{T,2}, ky::Array{T,2}) where {T<:Real}
    Nkx = length(kx[:,1])
    Nky = length(ky[1,:])
    h   = im.*zeros(Nkx, Nky, 4)
    h_t = im.*zeros(Nkx, Nky, 4)

    get_evol_variable(h,h_t)
end


function rhs(h_evol::Array{T,1}, param::Parameters, t::TP) where {T<:Complex, TP<:Real}
    tol       = param.tol
    px_inter  = param.px(t,param.x,param.y)
    pxy_inter = param.pxy(t,param.x,param.y)
    py_inter  = param.py(t,param.x,param.y)
    pz_inter  = param.pz(t,param.x,param.y)
    kx        = param.kx
    ky        = param.ky
    Nkx       = length(kx[:,1])
    Nky       = length(ky[1,:])
    px        = Fourier_Transform_2D(px_inter)
    pxy       = Fourier_Transform_2D(pxy_inter)
    py        = Fourier_Transform_2D(py_inter)
    pz        = Fourier_Transform_2D(pz_inter)
    h, h_t    = get_tensors(h_evol, Nkx, Nky)
    dh        = similar(h)
    dh_t      = similar(h_t)

#TODO: @threads gives me instability here for some reason...
    @time @fastmath @inbounds for j in 1:Nky
        for i in 1:Nkx
            kkx  = kx[i,j]
            kky  = ky[i,j]
            kkx2 = kkx^2
            kky2 = kky^2
            k2   = kkx^2+kky^2
            ppx  = px[i,j]
            ppxy = pxy[i,j]
            ppy  = py[i,j]
            ppz  = pz[i,j]

            M = 0.5/k2*(kky2*ppx+kkx2*ppy+2*kkx*kky*ppxy-k2*ppz).*[kky^2/k2, kkx*kky/k2, kkx^2/k2, -1]
            indices = findall(abs.(M) .< tol)
            for i in indices
                M[i] = 0.0
            end
            dh[i,j,:]   = h_t[i,j,:]
            dh_t[i,j,:] = -M -k2.*h[i,j,:]
        end
    end
    ppx         = px[1,1]
    ppxy        = pxy[1,1]
    ppy         = py[1,1]
    ppz         = pz[1,1]
    trT         = ppx + ppy + ppz
    M           = [ppx, ppxy, ppy, ppz] -1/3*trT.*[1,0,1,1]
    indices     = findall(abs.(M) .< tol)
    for i in indices
        M[i]    = 0.0
    end
    dh_t[1,1,:] = -M

    get_evol_variable(dh,dh_t)
end

#Computes the component associated to field in the bounbdary stress tensor.
#E.g. field=energy gives us h00 as solution.
function solve_GW(outdir::String, dirname::String; dt::T = 0.0, dt_output::T = 0.0,
                                    alg, option::String="static", tol::T=10^-10) where {T<:Real}
    px       = VEVTimeSeries(dirname, :px)
    pxy      = VEVTimeSeries(dirname, :pxy)
    py       = VEVTimeSeries(dirname, :py)
    pz       = VEVTimeSeries(dirname, :pz)
    tt, x, y = get_coords(px,:,:,:)
    tspan    = (tt[1],tt[end])

    if dt == 0.0 dt = tt[2]-tt[1] end
    if dt_output == 0.0 dt_output = tt[2]-tt[1] end

    dx       = x[2]-x[1]
    dy       = y[2]-y[1]
    kxx      = 2π.*rfftfreq(length(x), 1/dx)
    kyy      = 2π.*fftfreq(length(y), 1/dy)
    Nkx      = length(kxx)
    Nky      = length(kyy)
    kx       = zeros(Nkx,Nky)
    ky       = zeros(Nkx,Nky)

    @fastmath @inbounds @threads for j in eachindex(kyy)
        for i in eachindex(kxx)
            kx[i,j] = kxx[i]
            ky[i,j] = kyy[j]
        end
    end

    Nx, Ny       = (length(x), length(y))
    chart2D      = Chart(["Cartesian","Cartesian"], ["x","y"], [x[1],y[1]], [x[end],y[end]], [Nx, Ny])
    h0_evol      = initial_conditions(px,kx,ky)
    h, h_t       = Inverse_Fourier_Transform_2D(h0_evol, Nkx, Nky, Nx)
    px_inter     = interpolate((tt,x,y,), px[:,:,:], Gridded(Linear()))
    pxy_inter    = interpolate((tt,x,y,), pxy[:,:,:], Gridded(Linear()))
    py_inter     = interpolate((tt,x,y,), py[:,:,:], Gridded(Linear()))
    pz_inter     = interpolate((tt,x,y,), pz[:,:,:], Gridded(Linear()))
    ff_inter     = [px_inter, pxy_inter, py_inter, pz_inter]
    prm          = param{typeof(px_inter),typeof(x[1])}(px_inter,pxy_inter,py_inter,pz_inter,
                                                    x,y,kx,ky,dt,tol)
    problem      = ODEProblem(rhs, h0_evol, tspan, prm)
    integrator   = init(problem, alg, save_everystep=false, dt=dt, adaptive=false)
    tstart       = time()
    tinfo        = Jecco.TimeInfo(0, 0.0, 0.0, 0.0)
    output_GW(outdir, h, h_t, chart2D, tinfo)
    last_printed = tspan[1]
    for (u, t) in tuples(integrator)
        tinfo.it     += 1
        tinfo.dt      = integrator.dt
        tinfo.t       = t
        tinfo.runtime = time()-tstart
        h, h_t        = Inverse_Fourier_Transform_2D(u, Nkx, Nky, Nx)
        umax          = maximum(abs.(@view u[2:Nkx*Nky]))
        dumax         = maximum(abs.(@view u[Nkx*Nky+2:end]))
        indices       = findfirst(abs.(@view u[Nkx*Nky+2:end]) .== dumax)

        if t-last_printed-1e-12 >= dt_output
            last_printed = t
            output_GW(outdir, h, h_t, chart2D, tinfo)
            println("File printed")
        end

        println((t-tspan[1])%dt_output)
        println("t = $t")
        println("\n")
        println("max |hk| = $umax")
        println("max |hk_t| = $dumax")
        println("index = $indices")
        println("------------------------------------------------------------")
        println("\n")
    end
end
