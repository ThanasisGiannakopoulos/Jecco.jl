# DEBUG=0 to supress debug info
DEBUG = 1
using HDF5

function compute_coeffs_AH!(sigma::Array, gauge::Gauge, cache::HorizonCache,
                            sys::System{Outer})
    _, Nx, Ny = size(sys)

    Dx  = sys.Dx
    Dxx = sys.Dxx
    Dy  = sys.Dy
    Dyy = sys.Dyy

    B1_uAH      = cache.bulkhorizon.B1_uAH
    B2_uAH      = cache.bulkhorizon.B2_uAH
    G_uAH       = cache.bulkhorizon.G_uAH
    S_uAH       = cache.bulkhorizon.S_uAH
    Fx_uAH      = cache.bulkhorizon.Fx_uAH
    Fy_uAH      = cache.bulkhorizon.Fy_uAH
    Sd_uAH      = cache.bulkhorizon.Sd_uAH

    Du_B1_uAH   = cache.bulkhorizon.Du_B1_uAH
    Du_B2_uAH   = cache.bulkhorizon.Du_B2_uAH
    Du_G_uAH    = cache.bulkhorizon.Du_G_uAH
    Du_S_uAH    = cache.bulkhorizon.Du_S_uAH
    Du_Fx_uAH   = cache.bulkhorizon.Du_Fx_uAH
    Du_Fy_uAH   = cache.bulkhorizon.Du_Fy_uAH
    Du_Sd_uAH   = cache.bulkhorizon.Du_Sd_uAH

    Duu_B1_uAH  = cache.bulkhorizon.Duu_B1_uAH
    Duu_B2_uAH  = cache.bulkhorizon.Duu_B2_uAH
    Duu_G_uAH   = cache.bulkhorizon.Duu_G_uAH
    Duu_S_uAH   = cache.bulkhorizon.Duu_S_uAH
    Duu_Fx_uAH  = cache.bulkhorizon.Duu_Fx_uAH
    Duu_Fy_uAH  = cache.bulkhorizon.Duu_Fy_uAH

    axx         = cache.axx
    ayy         = cache.ayy
    axy         = cache.axy
    bx          = cache.bx
    by          = cache.by
    cc          = cache.cc
    b_vec       = cache.b_vec

    ind2D  = LinearIndices(B1_uAH[1,:,:])

    # coefficients of the derivative operators
    @fastmath @inbounds Threads.@threads for j in 1:Ny
        @inbounds for i in 1:Nx
            idx   = ind2D[i,j]

            uAH   = sigma[1,i,j]
            u2    = uAH * uAH
            u3    = uAH * uAH * uAH
            u4    = uAH * uAH * uAH * uAH

            xi    = gauge.xi[1,i,j]
            xi_x  = Dx(gauge.xi, 1,i,j)
            xi_y  = Dy(gauge.xi, 1,i,j)
            xi_xx = Dxx(gauge.xi, 1,i,j)
            xi_yy = Dyy(gauge.xi, 1,i,j)
            xi_xy = Dx(Dy, gauge.xi, 1,i,j)

            sigma0    = sigma[1,i,j]
            sigma0_x  = Dx(sigma, 1,i,j)
            sigma0_y  = Dy(sigma, 1,i,j)
            sigma0_xx = Dxx(sigma, 1,i,j)
            sigma0_yy = Dyy(sigma, 1,i,j)
            sigma0_xy = Dx(Dy, sigma, 1,i,j)

            B1    = B1_uAH[1,i,j]
            B2    = B2_uAH[1,i,j]
            G     = G_uAH[1,i,j]
            S     = S_uAH[1,i,j]
            Fx    = Fx_uAH[1,i,j]
            Fy    = Fy_uAH[1,i,j]
            Sd    = Sd_uAH[1,i,j]

            # r derivatives

            B1p        = -u2 * Du_B1_uAH[1,i,j]
            B2p        = -u2 * Du_B2_uAH[1,i,j]
            Gp         = -u2 * Du_G_uAH[1,i,j]
            Sp         = -u2 * Du_S_uAH[1,i,j]
            Fxp        = -u2 * Du_Fx_uAH[1,i,j]
            Fyp        = -u2 * Du_Fy_uAH[1,i,j]
            Sdp        = -u2 * Du_Sd_uAH[1,i,j]

            B1pp       = 2*u3 * Du_B1_uAH[1,i,j]  + u4 * Duu_B1_uAH[1,i,j]
            B2pp       = 2*u3 * Du_B2_uAH[1,i,j]  + u4 * Duu_B2_uAH[1,i,j]
            Gpp        = 2*u3 * Du_G_uAH[1,i,j]   + u4 * Duu_G_uAH[1,i,j]
            Spp        = 2*u3 * Du_S_uAH[1,i,j]   + u4 * Duu_S_uAH[1,i,j]
            Fxpp       = 2*u3 * Du_Fx_uAH[1,i,j]  + u4 * Duu_Fx_uAH[1,i,j]
            Fypp       = 2*u3 * Du_Fy_uAH[1,i,j]  + u4 * Duu_Fy_uAH[1,i,j]

            # x and y derivatives

            B1_x    = Dx(B1_uAH,   1,i,j) - Du_B1_uAH[1,i,j] * sigma0_x
            B2_x    = Dx(B2_uAH,   1,i,j) - Du_B2_uAH[1,i,j] * sigma0_x
            G_x     = Dx(G_uAH,    1,i,j) -  Du_G_uAH[1,i,j] * sigma0_x
            S_x     = Dx(S_uAH,    1,i,j) -  Du_S_uAH[1,i,j] * sigma0_x
            Fx_x    = Dx(Fx_uAH,   1,i,j) - Du_Fx_uAH[1,i,j] * sigma0_x
            Fy_x    = Dx(Fy_uAH,   1,i,j) - Du_Fy_uAH[1,i,j] * sigma0_x
            Sd_x    = Dx(Sd_uAH,   1,i,j) - Du_Sd_uAH[1,i,j] * sigma0_x
            # if j == 1
            #     @show Sd_x
            # end

            B1_y    = Dy(B1_uAH,   1,i,j) - Du_B1_uAH[1,i,j] * sigma0_y
            B2_y    = Dy(B2_uAH,   1,i,j) - Du_B2_uAH[1,i,j] * sigma0_y
            G_y     = Dy(G_uAH,    1,i,j) -  Du_G_uAH[1,i,j] * sigma0_y
            S_y     = Dy(S_uAH,    1,i,j) -  Du_S_uAH[1,i,j] * sigma0_y
            Fx_y    = Dy(Fx_uAH,   1,i,j) - Du_Fx_uAH[1,i,j] * sigma0_y
            Fy_y    = Dy(Fy_uAH,   1,i,j) - Du_Fy_uAH[1,i,j] * sigma0_y
            Sd_y    = Dy(Sd_uAH,   1,i,j) - Du_Sd_uAH[1,i,j] * sigma0_y

            B1p_x   = -u2 * Dx(Du_B1_uAH, 1,i,j) + u2 * sigma0_x * Duu_B1_uAH[1,i,j]
            B2p_x   = -u2 * Dx(Du_B2_uAH, 1,i,j) + u2 * sigma0_x * Duu_B2_uAH[1,i,j]
            Gp_x    = -u2 * Dx(Du_G_uAH,  1,i,j) + u2 * sigma0_x *  Duu_G_uAH[1,i,j]
            Sp_x    = -u2 * Dx(Du_S_uAH,  1,i,j) + u2 * sigma0_x *  Duu_S_uAH[1,i,j]
            Fxp_x   = -u2 * Dx(Du_Fx_uAH, 1,i,j) + u2 * sigma0_x * Duu_Fx_uAH[1,i,j]
            Fyp_x   = -u2 * Dx(Du_Fy_uAH, 1,i,j) + u2 * sigma0_x * Duu_Fy_uAH[1,i,j]

            B1p_y   = -u2 * Dy(Du_B1_uAH, 1,i,j) + u2 * sigma0_y * Duu_B1_uAH[1,i,j]
            B2p_y   = -u2 * Dy(Du_B2_uAH, 1,i,j) + u2 * sigma0_y * Duu_B2_uAH[1,i,j]
            Gp_y    = -u2 * Dy(Du_G_uAH,  1,i,j) + u2 * sigma0_y *  Duu_G_uAH[1,i,j]
            Sp_y    = -u2 * Dy(Du_S_uAH,  1,i,j) + u2 * sigma0_y *  Duu_S_uAH[1,i,j]
            Fxp_y   = -u2 * Dy(Du_Fx_uAH, 1,i,j) + u2 * sigma0_y * Duu_Fx_uAH[1,i,j]
            Fyp_y   = -u2 * Dy(Du_Fy_uAH, 1,i,j) + u2 * sigma0_y * Duu_Fy_uAH[1,i,j]

            vars = (
                sigma0, sigma0_x, sigma0_y, sigma0_xx, sigma0_yy, sigma0_xy,
                xi    , xi_x    , xi_y    , xi_xx    , xi_yy    , xi_xy,
                B1   , B2   , G   ,  S    , Fx    , Fy    , Sd ,
                B1p  , B2p  , Gp  ,  Sp   , Fxp   , Fyp   , Sdp,
                B1pp , B2pp , Gpp ,  Spp  , Fxpp  , Fypp  ,
                B1_x , B2_x , G_x ,  S_x  , Fx_x  , Fy_x  , Sd_x,
	        B1_y , B2_y , G_y ,  S_y  , Fx_y  , Fy_y  , Sd_y,
                B1p_x, B2p_x, Gp_x,  Sp_x , Fxp_x , Fyp_x ,
                B1p_y, B2p_y, Gp_y,  Sp_y , Fxp_y , Fyp_y ,
            )

            a11, a22, a12, b1, b2, c, res = AH_eq_coeff(vars, sys.gridtype)

            axx[idx]   = a11
            ayy[idx]   = a22
            axy[idx]   = a12
            bx[idx]    = b1
            by[idx]    = b2
            cc[idx]    = c
            b_vec[idx] = res
        end
    end

    nothing
end


#= Finding the Apparent Horizon

this is a 2D (non-linear) PDE of the type

  αxx f_xx + αyy f_yy + αxy f_xy + βxx (f_x)^2 + βyy (f_y)^2 + βxy (f_x) (f_y)
   + γx f_x + γy f_y + S = 0

we solve this equation with a Newton-Kantorovich method where, starting with a
guess, we solve the associated linear problem (using the same strategy as in
compute_xi_t!) to improve our guess until a sufficiently precise solution is
reached.
=#
function find_AH!(sigma::Array, bulkconstrain::BulkConstrained,
                  bulkevol::BulkEvolved, deriv::BulkDeriv, gauge::Gauge,
                  cache::HorizonCache, sys::System{Outer}, ahf::AHF)
    _, Nx, Ny = size(sys)
    bulk = Bulk(bulkevol, bulkconstrain)

    Du  = sys.Du
    Duu = sys.Duu
    Dx  = sys.Dx
    Dxx = sys.Dxx
    Dy  = sys.Dy
    Dyy = sys.Dyy

    interp = sys.uinterp

    B1_uAH      = cache.bulkhorizon.B1_uAH
    B2_uAH      = cache.bulkhorizon.B2_uAH
    G_uAH       = cache.bulkhorizon.G_uAH
    S_uAH       = cache.bulkhorizon.S_uAH
    Fx_uAH      = cache.bulkhorizon.Fx_uAH
    Fy_uAH      = cache.bulkhorizon.Fy_uAH
    Sd_uAH      = cache.bulkhorizon.Sd_uAH

    Du_B1_uAH   = cache.bulkhorizon.Du_B1_uAH
    Du_B2_uAH   = cache.bulkhorizon.Du_B2_uAH
    Du_G_uAH    = cache.bulkhorizon.Du_G_uAH
    Du_S_uAH    = cache.bulkhorizon.Du_S_uAH
    Du_Fx_uAH   = cache.bulkhorizon.Du_Fx_uAH
    Du_Fy_uAH   = cache.bulkhorizon.Du_Fy_uAH
    Du_Sd_uAH   = cache.bulkhorizon.Du_Sd_uAH

    Duu_B1_uAH  = cache.bulkhorizon.Duu_B1_uAH
    Duu_B2_uAH  = cache.bulkhorizon.Duu_B2_uAH
    Duu_G_uAH   = cache.bulkhorizon.Duu_G_uAH
    Duu_S_uAH   = cache.bulkhorizon.Duu_S_uAH
    Duu_Fx_uAH  = cache.bulkhorizon.Duu_Fx_uAH
    Duu_Fy_uAH  = cache.bulkhorizon.Duu_Fy_uAH

    axx         = cache.axx
    ayy         = cache.ayy
    axy         = cache.axy
    bx          = cache.bx
    by          = cache.by
    cc          = cache.cc
    b_vec       = cache.b_vec

    Dx_2D       = cache.Dx_2D
    Dxx_2D      = cache.Dxx_2D
    Dy_2D       = cache.Dy_2D
    Dyy_2D      = cache.Dyy_2D
    Dxy_2D      = cache.Dxy_2D
    _Dx_2D      = cache._Dx_2D
    _Dxx_2D     = cache._Dxx_2D
    _Dy_2D      = cache._Dy_2D
    _Dyy_2D     = cache._Dyy_2D
    _Dxy_2D     = cache._Dxy_2D

    # take all u-derivatives. we assume that they've not been computed beforehand
    @sync begin
        @spawn mul!(deriv.Du_B1,  Du,  bulk.B1)
        @spawn mul!(deriv.Du_B2,  Du,  bulk.B2)
        @spawn mul!(deriv.Du_G,   Du,  bulk.G)
        @spawn mul!(deriv.Du_S,   Du,  bulk.S)
        @spawn mul!(deriv.Du_Fx,  Du,  bulk.Fx)
        @spawn mul!(deriv.Du_Fy,  Du,  bulk.Fy)
        @spawn mul!(deriv.Du_Sd,  Du,  bulk.Sd)

        @spawn mul!(deriv.Duu_B1, Duu, bulk.B1)
        @spawn mul!(deriv.Duu_B2, Duu, bulk.B2)
        @spawn mul!(deriv.Duu_G,  Duu, bulk.G)
        @spawn mul!(deriv.Duu_S,  Duu, bulk.S)
        @spawn mul!(deriv.Duu_Fx, Duu, bulk.Fx)
        @spawn mul!(deriv.Duu_Fy, Duu, bulk.Fy)
        # @spawn mul!(deriv.Duu_Sd, Duu, bulk.Sd)
    end

    f0  = similar(b_vec)

    println("INFO (AH): Looking for the apparent horizon...")
    println("    it \t max_res")
    # start relaxation method
    it = 0
    while true
        
        ##########################################################################
        sigma0      = zeros(1,Nx,Ny)
        sigma0_x    = zeros(1,Nx,Ny)
        sigma0_y    = zeros(1,Nx,Ny)
        sigma0_xx   = zeros(1,Nx,Ny)
        sigma0_yy   = zeros(1,Nx,Ny)
        sigma0_xy   = zeros(1,Nx,Ny)
        xi          = zeros(1,Nx,Ny)
        xi_x        = zeros(1,Nx,Ny)
        xi_y        = zeros(1,Nx,Ny)
        xi_xx       = zeros(1,Nx,Ny)
        xi_yy       = zeros(1,Nx,Ny)
        xi_xy       = zeros(1,Nx,Ny)
        B1          = zeros(1,Nx,Ny)
        B2          = zeros(1,Nx,Ny)
        G           = zeros(1,Nx,Ny)
        S           = zeros(1,Nx,Ny)
        Fx          = zeros(1,Nx,Ny)
        Fy          = zeros(1,Nx,Ny)
        Sd          = zeros(1,Nx,Ny)
        B1p         = zeros(1,Nx,Ny)
        B2p         = zeros(1,Nx,Ny)
        Gp          = zeros(1,Nx,Ny)
        Sp          = zeros(1,Nx,Ny)
        Fxp         = zeros(1,Nx,Ny)
        Fyp         = zeros(1,Nx,Ny)
        Sdp         = zeros(1,Nx,Ny)
        B1pp        = zeros(1,Nx,Ny)
        B2pp        = zeros(1,Nx,Ny)
        Gpp         = zeros(1,Nx,Ny)
        Spp         = zeros(1,Nx,Ny)
        Fxpp        = zeros(1,Nx,Ny)
        Fypp        = zeros(1,Nx,Ny)
        B1_x        = zeros(1,Nx,Ny)
        B2_x        = zeros(1,Nx,Ny)
        G_x         = zeros(1,Nx,Ny)
        S_x         = zeros(1,Nx,Ny)
        Fx_x        = zeros(1,Nx,Ny)
        Fy_x        = zeros(1,Nx,Ny)
        Sd_x        = zeros(1,Nx,Ny)
        B1_y        = zeros(1,Nx,Ny)
        B2_y        = zeros(1,Nx,Ny)
        G_y         = zeros(1,Nx,Ny)
        S_y         = zeros(1,Nx,Ny)
        Fx_y        = zeros(1,Nx,Ny)
        Fy_y        = zeros(1,Nx,Ny)
        Sd_y        = zeros(1,Nx,Ny)
        B1p_x       = zeros(1,Nx,Ny)
        B2p_x       = zeros(1,Nx,Ny)
        Gp_x        = zeros(1,Nx,Ny)
        Sp_x        = zeros(1,Nx,Ny)
        Fxp_x       = zeros(1,Nx,Ny)
        Fyp_x       = zeros(1,Nx,Ny)
        B1p_y       = zeros(1,Nx,Ny)
        B2p_y       = zeros(1,Nx,Ny)
        Gp_y        = zeros(1,Nx,Ny)
        Sp_y        = zeros(1,Nx,Ny)
        Fxp_y       = zeros(1,Nx,Ny)
        Fyp_y       = zeros(1,Nx,Ny)
        #########################################################################    

        # interpolate bulk functions (and u-derivatives) to the u = 1/r = sigma surface
        @inbounds Threads.@threads for j in 1:Ny
            @inbounds for i in 1:Nx
                uAH = sigma[1,i,j]
                u2  = uAH * uAH
                u3  = uAH * uAH * uAH
                u4  = uAH * uAH * uAH * uAH

                B1_uAH[1,i,j]       = interp(view(bulk.B1,  :,i,j))(uAH)
                B2_uAH[1,i,j]       = interp(view(bulk.B2,  :,i,j))(uAH)
                G_uAH[1,i,j]        = interp(view(bulk.G,   :,i,j))(uAH)
                S_uAH[1,i,j]        = interp(view(bulk.S,   :,i,j))(uAH)
                Fx_uAH[1,i,j]       = interp(view(bulk.Fx,  :,i,j))(uAH)
                Fy_uAH[1,i,j]       = interp(view(bulk.Fy,  :,i,j))(uAH)
                Sd_uAH[1,i,j]       = interp(view(bulk.Sd,  :,i,j))(uAH)

                Du_B1_uAH[1,i,j]    = interp(view(deriv.Du_B1,  :,i,j))(uAH)
                Du_B2_uAH[1,i,j]    = interp(view(deriv.Du_B2,  :,i,j))(uAH)
                Du_G_uAH[1,i,j]     = interp(view(deriv.Du_G,   :,i,j))(uAH)
                Du_S_uAH[1,i,j]     = interp(view(deriv.Du_S,   :,i,j))(uAH)
                Du_Fx_uAH[1,i,j]    = interp(view(deriv.Du_Fx,  :,i,j))(uAH)
                Du_Fy_uAH[1,i,j]    = interp(view(deriv.Du_Fy,  :,i,j))(uAH)
                Du_Sd_uAH[1,i,j]    = interp(view(deriv.Du_Sd,  :,i,j))(uAH)

                Duu_B1_uAH[1,i,j]   = interp(view(deriv.Duu_B1,  :,i,j))(uAH)
                Duu_B2_uAH[1,i,j]   = interp(view(deriv.Duu_B2,  :,i,j))(uAH)
                Duu_G_uAH[1,i,j]    = interp(view(deriv.Duu_G,   :,i,j))(uAH)
                Duu_S_uAH[1,i,j]    = interp(view(deriv.Duu_S,   :,i,j))(uAH)
                Duu_Fx_uAH[1,i,j]   = interp(view(deriv.Duu_Fx,  :,i,j))(uAH)
                Duu_Fy_uAH[1,i,j]   = interp(view(deriv.Duu_Fy,  :,i,j))(uAH)

            end
        end

        @inbounds Threads.@threads for j in 1:Ny
            @inbounds for i in 1:Nx

                uAH = sigma[1,i,j]
                u2    = uAH * uAH
                u3    = uAH * uAH * uAH
                u4    = uAH * uAH * uAH * uAH

                xi[1,i,j]    = gauge.xi[1,i,j]
                xi_x[1,i,j]  = Dx(gauge.xi, 1,i,j)
                xi_y[1,i,j]  = Dy(gauge.xi, 1,i,j)
                xi_xx[1,i,j] = Dxx(gauge.xi, 1,i,j)
                xi_yy[1,i,j] = Dyy(gauge.xi, 1,i,j)
                xi_xy[1,i,j] = Dx(Dy, gauge.xi, 1,i,j)

                sigma0[1,i,j]   = sigma[1,i,j]
                sigma0_x[1,i,j]  = Dx(sigma, 1,i,j)
                sigma0_y[1,i,j]  = Dy(sigma, 1,i,j)
                sigma0_xx[1,i,j] = Dxx(sigma, 1,i,j)
                sigma0_yy[1,i,j] = Dyy(sigma, 1,i,j)
                sigma0_xy[1,i,j] = Dx(Dy, sigma, 1,i,j)

                B1[1,i,j]    = B1_uAH[1,i,j]
                B2[1,i,j]    = B2_uAH[1,i,j]
                G[1,i,j]     = G_uAH[1,i,j]
                S[1,i,j]     = S_uAH[1,i,j]
                Fx[1,i,j]    = Fx_uAH[1,i,j]
                Fy[1,i,j]    = Fy_uAH[1,i,j]
                Sd[1,i,j]    = Sd_uAH[1,i,j]

                # r derivatives

                B1p[1,i,j]        = -u2 * Du_B1_uAH[1,i,j]
                B2p[1,i,j]        = -u2 * Du_B2_uAH[1,i,j]
                Gp[1,i,j]         = -u2 * Du_G_uAH[1,i,j]
                Sp[1,i,j]         = -u2 * Du_S_uAH[1,i,j]
                Fxp[1,i,j]        = -u2 * Du_Fx_uAH[1,i,j]
                Fyp[1,i,j]        = -u2 * Du_Fy_uAH[1,i,j]
                Sdp[1,i,j]        = -u2 * Du_Sd_uAH[1,i,j]

                B1pp[1,i,j]       = 2*u3 * Du_B1_uAH[1,i,j]  + u4 * Duu_B1_uAH[1,i,j]
                B2pp[1,i,j]       = 2*u3 * Du_B2_uAH[1,i,j]  + u4 * Duu_B2_uAH[1,i,j]
                Gpp[1,i,j]        = 2*u3 * Du_G_uAH[1,i,j]   + u4 * Duu_G_uAH[1,i,j]
                Spp[1,i,j]        = 2*u3 * Du_S_uAH[1,i,j]   + u4 * Duu_S_uAH[1,i,j]
                Fxpp[1,i,j]       = 2*u3 * Du_Fx_uAH[1,i,j]  + u4 * Duu_Fx_uAH[1,i,j]
                Fypp[1,i,j]       = 2*u3 * Du_Fy_uAH[1,i,j]  + u4 * Duu_Fy_uAH[1,i,j]

                # x and y derivatives

                B1_x[1,i,j]    = Dx(B1_uAH,   1,i,j) - Du_B1_uAH[1,i,j] * sigma0_x[1,i,j]
                B2_x[1,i,j]    = Dx(B2_uAH,   1,i,j) - Du_B2_uAH[1,i,j] * sigma0_x[1,i,j]
                G_x[1,i,j]     = Dx(G_uAH,    1,i,j) -  Du_G_uAH[1,i,j] * sigma0_x[1,i,j]
                S_x[1,i,j]     = Dx(S_uAH,    1,i,j) -  Du_S_uAH[1,i,j] * sigma0_x[1,i,j]
                Fx_x[1,i,j]    = Dx(Fx_uAH,   1,i,j) - Du_Fx_uAH[1,i,j] * sigma0_x[1,i,j]
                Fy_x[1,i,j]    = Dx(Fy_uAH,   1,i,j) - Du_Fy_uAH[1,i,j] * sigma0_x[1,i,j]
                Sd_x[1,i,j]    = Dx(Sd_uAH,   1,i,j) - Du_Sd_uAH[1,i,j] * sigma0_x[1,i,j]

                # if j==1
                #     @show Sd_x[1,i,j] 
                # end
                
                B1_y[1,i,j]    = Dy(B1_uAH,   1,i,j) - Du_B1_uAH[1,i,j] * sigma0_y[1,i,j]
                B2_y[1,i,j]    = Dy(B2_uAH,   1,i,j) - Du_B2_uAH[1,i,j] * sigma0_y[1,i,j]
                G_y[1,i,j]     = Dy(G_uAH,    1,i,j) -  Du_G_uAH[1,i,j] * sigma0_y[1,i,j]
                S_y[1,i,j]     = Dy(S_uAH,    1,i,j) -  Du_S_uAH[1,i,j] * sigma0_y[1,i,j]
                Fx_y[1,i,j]    = Dy(Fx_uAH,   1,i,j) - Du_Fx_uAH[1,i,j] * sigma0_y[1,i,j]
                Fy_y[1,i,j]    = Dy(Fy_uAH,   1,i,j) - Du_Fy_uAH[1,i,j] * sigma0_y[1,i,j]
                Sd_y[1,i,j]    = Dy(Sd_uAH,   1,i,j) - Du_Sd_uAH[1,i,j] * sigma0_y[1,i,j]

                B1p_x[1,i,j]   = -u2 * Dx(Du_B1_uAH, 1,i,j) + u2 * sigma0_x[1,i,j] * Duu_B1_uAH[1,i,j]
                B2p_x[1,i,j]   = -u2 * Dx(Du_B2_uAH, 1,i,j) + u2 * sigma0_x[1,i,j] * Duu_B2_uAH[1,i,j]
                Gp_x[1,i,j]    = -u2 * Dx(Du_G_uAH,  1,i,j) + u2 * sigma0_x[1,i,j] *  Duu_G_uAH[1,i,j]
                Sp_x[1,i,j]    = -u2 * Dx(Du_S_uAH,  1,i,j) + u2 * sigma0_x[1,i,j] *  Duu_S_uAH[1,i,j]
                Fxp_x[1,i,j]   = -u2 * Dx(Du_Fx_uAH, 1,i,j) + u2 * sigma0_x[1,i,j] * Duu_Fx_uAH[1,i,j]
                Fyp_x[1,i,j]   = -u2 * Dx(Du_Fy_uAH, 1,i,j) + u2 * sigma0_x[1,i,j] * Duu_Fy_uAH[1,i,j]

                B1p_y[1,i,j]   = -u2 * Dy(Du_B1_uAH, 1,i,j) + u2 * sigma0_y[1,i,j] * Duu_B1_uAH[1,i,j]
                B2p_y[1,i,j]   = -u2 * Dy(Du_B2_uAH, 1,i,j) + u2 * sigma0_y[1,i,j] * Duu_B2_uAH[1,i,j]
                Gp_y[1,i,j]    = -u2 * Dy(Du_G_uAH,  1,i,j) + u2 * sigma0_y[1,i,j] *  Duu_G_uAH[1,i,j]
                Sp_y[1,i,j]    = -u2 * Dy(Du_S_uAH,  1,i,j) + u2 * sigma0_y[1,i,j] *  Duu_S_uAH[1,i,j]
                Fxp_y[1,i,j]   = -u2 * Dy(Du_Fx_uAH, 1,i,j) + u2 * sigma0_y[1,i,j] * Duu_Fx_uAH[1,i,j]
                Fyp_y[1,i,j]   = -u2 * Dy(Du_Fy_uAH, 1,i,j) + u2 * sigma0_y[1,i,j] * Duu_Fy_uAH[1,i,j]

            end
        end
        

        # compute axx, ayy, axy, bx, by, cc and res coefficients of the
        # linearized equation (stored in the cache struct)
        compute_coeffs_AH!(sigma, gauge, cache, sys)

        # the residual is stored in b_vec
        max_res = maximum(abs.(b_vec))
        println("    $it \t $max_res")

if DEBUG==1

    
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/f0", f0)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/sigma",sigma)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/axx",axx)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/ayy",ayy)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/axy",axy)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/bx",bx)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/by",by)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/cc",cc)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/b_vec",b_vec)

    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1_uAH", B1_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2_uAH", B2_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/G_uAH", G_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/S_uAH", S_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fx_uAH", Fx_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fy_uAH", Fy_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sd_uAH", Sd_uAH)

    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Du_B1_uAH", Du_B1_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Du_B2_uAH", Du_B2_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Du_G_uAH", Du_G_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Du_S_uAH", Du_S_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Du_Fx_uAH", Du_Fx_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Du_Fy_uAH", Du_Fy_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Du_Sd_uAH", Du_Sd_uAH)
    
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Duu_B1_uAH", Duu_B1_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Duu_B2_uAH", Duu_B2_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Duu_G_uAH", Duu_G_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Duu_S_uAH", Duu_S_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Duu_Fx_uAH", Duu_Fx_uAH)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Duu_Fy_uAH", Duu_Fy_uAH)

    ##########################################################################
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/sigma0", sigma0)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/sigma0_x", sigma0_x)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/sigma0_y", sigma0_y)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/sigma0_xx", sigma0_xx)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/sigma0_yy", sigma0_yy)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/sigma0_xy", sigma0_xy)

    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/xi", xi)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/xi_x", xi_x)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/xi_y", xi_y)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/xi_xx", xi_xx)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/xi_yy", xi_yy)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/xi_xy", xi_xy)

    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1", B1)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2", B2)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/G", G)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/S", S)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fx", Fx)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fy", Fy)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sd", Sd)

    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1p", B1p)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2p", B2p)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Gp", Gp)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sp", Sp)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fxp", Fxp)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fyp", Fyp)
    h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sdp", Sdp)

h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1pp", B1pp)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2pp", B2pp)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Gpp", Gpp)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Spp", Spp)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fxpp", Fxpp)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fypp", Fypp)

h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1_x", B1_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2_x", B2_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/G_x", G_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/S_x", S_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fx_x", Fx_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fy_x", Fy_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sd_x", Sd_x)

h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1_y", B1_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2_y", B2_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/G_y", G_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/S_y", S_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fx_y", Fx_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fy_y", Fy_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sd_y", Sd_y)

h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1p_x", B1p_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2p_x", B2p_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Gp_x", Gp_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sp_x", Sp_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fxp_x", Fxp_x)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fyp_x", Fyp_x)

h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B1p_y", B1p_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/B2p_y", B2p_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Gp_y", Gp_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Sp_y", Sp_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fxp_y", Fxp_y)
h5write("/home/thanasis/repos/Jecco.jl/debug/AH_debug_it_$(it).h5","AH/Fyp_y", Fyp_y)

#########################################################################    

        end
        
        if max_res < ahf.epsilon
            break
        end

        if it >= ahf.itmax
            println("INFO (AH): maximum iteration reached.")
            println("INFO (AH): giving up")
            break
        end

        # each time the mul_col! routine is called (below), the operators Dx_2D,
        # Dxx_2D, etc, are overwritten. so restore them here from _Dx_2D,
        # _Dxx_2D, etc, which are never overwritten.
        copyto!(Dx_2D,  _Dx_2D)
        copyto!(Dxx_2D, _Dxx_2D)
        copyto!(Dy_2D,  _Dy_2D)
        copyto!(Dyy_2D, _Dyy_2D)
        copyto!(Dxy_2D, _Dxy_2D)

        # overwrite the operators with the coefficients computed above
        mul_col!(axx, Dxx_2D)
        mul_col!(ayy, Dyy_2D)
        mul_col!(axy, Dxy_2D)
        mul_col!(bx,  Dx_2D)
        mul_col!(by,  Dy_2D)
        ccId = Diagonal(cc)

        # build actual operator to be inverted
        A_mat = Dxx_2D + Dyy_2D + Dxy_2D + Dx_2D + Dy_2D + ccId

        # since we're using periodic boundary conditions, the operator A_mat
        # (just like the Dx and Dxx operators) is strictly speaking not
        # invertible (it has zero determinant) since the solution is not unique.
        # indeed, its LU decomposition shouldn't even be defined. for some
        # reason, however, the call to "lu" does in fact factorize the matrix.
        # in any case, to be safer, let's instead call "factorize", which uses
        # fancy algorithms to determine which is the best way to factorize (and
        # which performs a QR decomposition if the LU fails). the inverse that
        # is performed probably returns the minimum norm least squares solution,
        # or something similar. in any case, for our purposes here we mostly
        # care about getting a solution (not necessarily the minimum norm least
        # squares one).
        A_fact = factorize(A_mat)
        ldiv!(f0, A_fact, b_vec)

# update solution
#check
        @inbounds for idx in eachindex(sigma)
           # @show idx
           # @show sigma[idx]
           # @show f0[idx]
            sigma[idx] -= f0[idx]
           # @show sigma[idx]
        end
        it += 1
        
        # check if sigma inside domain
        min_uAH = minimum(sigma)
        max_uAH = maximum(sigma)
        if ( min_uAH < sys.ucoord[1] || max_uAH > sys.ucoord[end] )
            println("INFO (AH): guess outside domain, min_uAH = $min_uAH, max_uAH = $max_uAH")
            println("INFO (AH): giving up")
            break
        end

    end # end relaxation loop

    nothing
end
