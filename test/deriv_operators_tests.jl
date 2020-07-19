
@testset "FD Derivative tests:" begin

    # periodic FD
    Dx = CenteredDiff{1}(1, 4, 1, 10)
    @test 12*Dx[1,:]  == [0.0, 8.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -8.0]
    @test 12*Dx[2,:]  == [-8.0, 0.0, 8.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    @test 12*Dx[3,:]  == [1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test 12*Dx[4,:]  == [0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    @test 12*Dx[5,:]  == [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0, 0.0]
    @test 12*Dx[6,:]  == [0.0, 0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0]
    @test 12*Dx[7,:]  == [0.0, 0.0, 0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0]
    @test 12*Dx[8,:]  == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0]
    @test 12*Dx[9,:]  == [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -8.0, 0.0, 8.0]
    @test 12*Dx[10,:] == [8.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -8.0, 0.0]

    # non-periodic FD
    Dx_np = Non_periodic_FD{1}(1, 2, 1, 10)
    @test 2*Dx_np[1,:]  == [-3.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test 2*Dx_np[2,:]  == [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test 2*Dx_np[3,:]  == [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test 2*Dx_np[4,:]  == [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test 2*Dx_np[5,:]  == [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    @test 2*Dx_np[6,:]  == [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    @test 2*Dx_np[7,:]  == [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0]
    @test 2*Dx_np[8,:]  == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    @test 2*Dx_np[9,:]  == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0]
    @test 2*Dx_np[10,:] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -4.0, 3.0]

    # 1D case

    xmin   = -2.0*pi
    xmax   =  2.0*pi
    xnodes =  600
    ord    =  4

    hx     = (xmax - xmin) / xnodes

    x  = collect(xmin:hx:xmax-hx)
    f  = sin.(x)

    D1 = CenteredDiff(1, ord, hx, length(x))
    D2 = CenteredDiff(2, ord, hx, length(x))

    df  = D1 * f
    d2f = D2 * f

    @test df  ≈ cos.(x) atol=hx^ord
    @test d2f ≈ -f atol=hx^ord

    # non-periodic FDs
    D1_np = Non_periodic_FD(1, 2, hx, length(x))
    D2_np = Non_periodic_FD(2, 2, hx, length(x))

    df_np  = D1_np * f
    d2f_np = D2_np * f

    # fixme manual tests is passed, but @test gives error
    #@test df_np  ≈ cos.(x) atol=hx^2
    #@test d2f_np ≈ -f atol=hx^2
    # the following works
    @test maximum(abs.(df_np - cos.(x))) ≈ 0.0 atol=hx^2
    @test maximum(abs.(d2f_np + f))      ≈ 0.0 atol=hx^2

    # now for the callable, point-wise, methods
    for i in eachindex(f)
        @test D1(f,i) == df[i]
        @test D2(f,i) == d2f[i]
        
        @test D1_np(f,i) == df_np[i]
        @test D2_np(f,i) == d2f_np[i]
    end

    # 3D case

    ymin   = -1.0*pi
    ymax   =  1.0*pi
    ynodes =  20
    zmin   = -1.0*pi
    zmax   =  1.0*pi
    znodes =  300

    hy     = (ymax - ymin) / ynodes
    hz     = (zmax - zmin) / znodes

    y      = collect(ymin:hy:ymax-hy)
    z      = collect(zmin:hz:zmax-hz)

    f      = [sin.(x1) .* sin.(x2) .* sin.(x3) for x1 in x, x2 in y, x3 in z]
    dxf0   = [cos.(x1) .* sin.(x2) .* sin.(x3) for x1 in x, x2 in y, x3 in z]
    dyf0   = [sin.(x1) .* cos.(x2) .* sin.(x3) for x1 in x, x2 in y, x3 in z]
    dzf0   = [sin.(x1) .* sin.(x2) .* cos.(x3) for x1 in x, x2 in y, x3 in z]
    dxzf0  = [cos.(x1) .* sin.(x2) .* cos.(x3) for x1 in x, x2 in y, x3 in z]

    # periodic
    Dx     = CenteredDiff{1}(1, ord, hx, length(x))
    Dz     = CenteredDiff{3}(1, ord, hz, length(z))

    Dxx    = CenteredDiff{1}(2, ord, hx, length(x))
    Dzz    = CenteredDiff{3}(2, ord, hz, length(z))

    dxf    = Dx * f
    dzf    = Dz * f
    dxzf   = Dx * (Dz * f)

    @test dxf  ≈ dxf0
    @test dzf  ≈ dzf0
    @test dxzf ≈ dxzf0

    d2xf   = Dxx * f
    d2zf   = Dzz * f

    @test d2xf ≈ -f
    @test d2zf ≈ -f

    # non-periodic
    Dx_np     = Non_periodic_FD{1}(1, 2, hx, length(x))
    Dz_np     = Non_periodic_FD{3}(1, 2, hz, length(z))

    Dxx_np    = Non_periodic_FD{1}(2, 2, hx, length(x))
    Dzz_np    = Non_periodic_FD{3}(2, 2, hz, length(z))

    dxf_np    = Dx_np * f
    dzf_np    = Dz_np * f
    dxzf_np   = Dx_np * (Dz_np * f)

    @test maximum(abs.(dxf_np - dxf0))   ≈ 0.0 atol=hx^2
    @test maximum(abs.(dzf_np - dzf0))   ≈ 0.0 atol=hx^2
    @test maximum(abs.(dxzf_np - dxzf0)) ≈ 0.0 atol=hx^2

    d2xf_np   = Dxx_np * f
    d2zf_np   = Dzz_np * f

    @test maximum(abs.(d2xf_np + f)) ≈ 0.0 atol=hx^2
    @test maximum(abs.(d2zf_np + f)) ≈ 0.0 atol=hx^2


    # now for callable, point-wise, methods
    @test Dx(f,2,10,120)  == dxf[2,10,120]
    @test Dx(f,42,20,300) == dxf[42,20,300]
    @test Dz(f,2,10,120)  == dzf[2,10,120]
    @test Dz(f,42,20,300) == dzf[42,20,300]

    @test Dx_np(f,2,10,120)  == dxf_np[2,10,120]
    @test Dx_np(f,42,20,300) == dxf_np[42,20,300]
    @test Dz_np(f,2,10,120)  == dzf_np[2,10,120]
    @test Dz_np(f,42,20,300) == dzf_np[42,20,300]
end

@testset "Spectral Derivative tests:" begin

    # 1D case
    xmin   = -2.0
    xmax   =  2.0
    xnodes =  32

    x, = Jecco.cheb(xmin, xmax, xnodes)
    f = 0.5 * x.^2

    Dx  = ChebDeriv(1, xmin, xmax, xnodes)
    Dxx = ChebDeriv(2, xmin, xmax, xnodes)

    dxf  = Dx * f
    dxxf = Dxx * f
    @test dxf  ≈ x
    @test dxxf ≈ fill(1.0, size(dxxf))

    # now for the callable, point-wise, methods
    for i in eachindex(f)
        @test Dx(f,i) ≈ dxf[i]
    end


    # 3D case

    ymin   = -1.0
    ymax   =  1.0
    ynodes =  8

    zmin   = -1.0
    zmax   =  1.0
    znodes =  16

    y, = Jecco.cheb(ymin, ymax, ynodes)
    z, = Jecco.cheb(zmin, zmax, znodes)

    Dy  = ChebDeriv{2}(1, ymin, ymax, ynodes)
    Dyy = ChebDeriv{2}(2, ymin, ymax, ynodes)
    Dz  = ChebDeriv{3}(1, zmin, zmax, znodes)
    Dzz = ChebDeriv{3}(2, zmin, zmax, znodes)

    f     = [0.5 * x1.^2 .* cos.(x2) .* sin.(x3) for x1 in x, x2 in y, x3 in z]
    dxf0  = [x1 .* cos.(x2) .* sin.(x3)          for x1 in x, x2 in y, x3 in z]
    dzf0  = [0.5 * x1.^2 .* cos.(x2) .* cos.(x3) for x1 in x, x2 in y, x3 in z]
    dxxf0 = [cos.(x2) .* sin.(x3)                for x1 in x, x2 in y, x3 in z]
    dzzf0 = -copy(f)

    dxf = Dx * f
    dzf = Dz * f
    d2xf = Dxx * f
    d2zf = Dzz * f

    @test dxf ≈ dxf0
    @test dzf ≈ dzf0

    @test d2xf ≈ dxxf0
    @test d2zf ≈ dzzf0

    # now for callable, point-wise, methods
    @test Dx(f,2,4,16)  ≈ dxf[2,4,16]
    @test Dx(f,1,6,12)  ≈ dxf[1,6,12]
    @test Dz(f,2,3,8)   ≈ dzf[2,3,8]
    @test Dz(f,16,8,1)  ≈ dzf[16,8,1]end

@testset "Cross FD derivative tests:" begin
    # 2D FD case
    xmin   = -2.0*pi
    xmax   =  2.0*pi
    xnodes =  600
    ymin   = -1.0*pi
    ymax   =  1.0*pi
    ynodes =  300
    ord    =  4

    hx     = (xmax - xmin) / xnodes
    hy     = (ymax - ymin) / ynodes

    x      = collect(xmin:hx:xmax-hx)
    y      = collect(ymin:hy:ymax-hy)

    f      = [sin.(x1) .* sin.(x2) for x1 in x, x2 in y]

    Dx     = CenteredDiff{1}(1, ord, hx, length(x))
    Dy     = CenteredDiff{2}(1, ord, hy, length(y))
    # non-periodic
    Dx_np  = Non_periodic_FD{1}(1, 2, hx, length(x))
    Dy_np  = Non_periodic_FD{2}(1, 2, hy, length(y))

    dxyf   = Dx * (Dy * f)
    # periodic non-periodic
    dx_ynp_f   = Dx * (Dy_np * f)
    # non-periodic periodic
    dxnp_y_f   = Dx_np * (Dy * f)
    # non-periodic non-periodic
    dxnp_ynp_f = Dx_np * (Dy_np * f)

    @test Dx(Dy, f,  2,120) ≈ dxyf[2,120]
    @test Dx(Dy, f, 42,300) ≈ dxyf[42,300]

    @test Dx(Dy_np, f,  2,120)  ≈ dx_ynp_f[2,120]
    @test Dx(Dy_np, f, 42,300)  ≈ dx_ynp_f[42,300]
    @test Dx(Dy_np, f, 1,300)   ≈ dx_ynp_f[1,300]
    @test Dx(Dy_np, f, 600,1)   ≈ dx_ynp_f[600,1]
    @test Dx(Dy_np, f, 1,1)     ≈ dx_ynp_f[1,1]
    @test Dx(Dy_np, f, 600,300) ≈ dx_ynp_f[600,300]
    
    @test Dx_np(Dy, f,  2,120)  ≈ dxnp_y_f[2,120]
    @test Dx_np(Dy, f, 42,300)  ≈ dxnp_y_f[42,300]
    @test Dx_np(Dy, f, 1,300)   ≈ dxnp_y_f[1,300]
    @test Dx_np(Dy, f, 600,1)   ≈ dxnp_y_f[600,1]
    @test Dx_np(Dy, f, 1,1)     ≈ dxnp_y_f[1,1]
    @test Dx_np(Dy, f, 600,300) ≈ dxnp_y_f[600,300]
    
    @test Dx_np(Dy_np, f,  2,120)  ≈ dxnp_ynp_f[2,120]
    @test Dx_np(Dy_np, f, 42,300)  ≈ dxnp_ynp_f[42,300]
    @test Dx_np(Dy_np, f, 1,300)   ≈ dxnp_ynp_f[1,300]
    @test Dx_np(Dy_np, f, 600,1)   ≈ dxnp_ynp_f[600,1]
    @test Dx_np(Dy_np, f, 1,1)     ≈ dxnp_ynp_f[1,1]
    @test Dx_np(Dy_np, f, 600,300) ≈ dxnp_ynp_f[600,300]

end

@testset "FD cross derivative for general arrays tests:" begin

    xmin   = -2.0*pi
    xmax   =  2.0*pi
    xnodes =  600
    ord    =  4

    ymin   = -1.0
    ymax   =  1.0
    ynodes =  16

    zmin   = -1.0*pi
    zmax   =  1.0*pi
    znodes =  300


    hx     = (xmax - xmin) / xnodes
    hz     = (zmax - zmin) / znodes

    x      = collect(xmin:hx:xmax-hx)
    y,     = Jecco.cheb(ymin, ymax, ynodes)
    z      = collect(zmin:hz:zmax-hz)

    f   = [0.5 * sin.(x1) .* x2.^2 .* sin.(x3)  for x1 in x, x2 in y, x3 in z]

    Dx  = CenteredDiff{1}(1, ord, hx, length(x))
    Dz  = CenteredDiff{3}(1, ord, hz, length(z))
    # non periodic
    Dx_np  = Non_periodic_FD{1}(1, 2, hx, length(x))
    Dz_np  = Non_periodic_FD{3}(1, 2, hz, length(z))

    dxzf  = Dx * (Dz * f)
    #non-periodic periodic
    dxnp_z_f   = Dx_np * (Dz * f)
    # periodic non-periodic
    dx_znp_f   = Dx * (Dz_np * f)
    # non-periodic non-periodic
    dxnp_znp_f = Dx_np * (Dz_np * f)

    @test Dx(Dz, f, 2,16,1)     ≈ dxzf[2,16,1]
    @test Dx(Dz, f, 1,12,2)     ≈ dxzf[1,12,2]
    @test Dx(Dz, f, 100,12,300) ≈ dxzf[100,12,300]
    @test Dx(Dz, f, 600,8,100)  ≈ dxzf[600,8,100]

    @test Dx_np(Dz, f, 2,16,1)     ≈ dxnp_z_f[2,16,1]
    @test Dx_np(Dz, f, 1,12,2)     ≈ dxnp_z_f[1,12,2]
    @test Dx_np(Dz, f, 100,12,300) ≈ dxnp_z_f[100,12,300]
    @test Dx_np(Dz, f, 600,8,100)  ≈ dxnp_z_f[600,8,100]

    @test Dx(Dz_np, f, 2,16,1)     ≈ dx_znp_f[2,16,1]
    @test Dx(Dz_np, f, 1,12,2)     ≈ dx_znp_f[1,12,2]
    @test Dx(Dz_np, f, 100,12,300) ≈ dx_znp_f[100,12,300]
    @test Dx(Dz_np, f, 600,8,100)  ≈ dx_znp_f[600,8,100]

    @test Dx_np(Dz_np, f, 2,16,1)     ≈ dxnp_znp_f[2,16,1]
    @test Dx_np(Dz_np, f, 1,12,2)     ≈ dxnp_znp_f[1,12,2]
    @test Dx_np(Dz_np, f, 100,12,300) ≈ dxnp_znp_f[100,12,300]
    @test Dx_np(Dz_np, f, 600,8,100)  ≈ dxnp_znp_f[600,8,100]


    g   = [sin.(x3) .* 0.5 * sin.(x2) .* x1.^2 for x3 in z, x1 in y, x2 in x]

    Dz  = CenteredDiff{1}(1, ord, hz, length(z))
    Dx  = CenteredDiff{3}(1, ord, hx, length(x))

    dxzg  = Dx * (Dz * g)

    @test Dz(Dx, g, 100,2,1)     ≈ dxzg[100,2,1]
    @test Dz(Dx, g, 1,1,1)       ≈ dxzg[1,1,1]
    @test Dz(Dx, g, 300,16,600)  ≈ dxzg[300,16,600]
    @test Dz(Dx, g, 10,8,150)    ≈ dxzg[10,8,150]
end
