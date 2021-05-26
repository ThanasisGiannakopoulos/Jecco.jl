using Jecco, Jecco.AdS5_3_1
using FFTW
using Plots
gr()

dirname   = "/home/mikel/Documents/Jecco.jl/data/end_data/"
outdir    = "/home/mikel/Documents/Jecco.jl/data/new_data/"
A_dir     = "/home/mikel/Documents/Jecco.jl/data/bubbles/phiM_0.85_phiQ_10/state_A_e_1.318/"
B_dir     = A_dir
PS_dir    = "/home/mikel/Documents/Jecco.jl/data/bubbles/phiM_0.85_phiQ_10/phase_separated/"

grid = SpecCartGrid3D(
    x_min            = -50.,
    x_max            =  50.,
    x_nodes          =  400,
    y_min            = -50.,
    y_max            =  50.,
    y_nodes          =  400,
    u_outer_min      =  0.1,
    u_outer_max      =  1.005,
    u_outer_domains  =  1,
    u_outer_nodes    =  48,
    u_inner_nodes    =  12,
    fd_order         =  4,
    sigma_diss       =  0.2,
)


potential = AdS5_3_1.Phi8Potential(
    #alpha   = -0.7,
    #beta    = 0.16,
    #gamma   = 0.0,
    oophiM2 = -1.38408,
    oophiQ  = 0.1,
)


io = InOut(recover_dir = dirname, out_dir = outdir, checkpoint_dir = outdir,
           out_boundary_every=1, out_gauge_every=1,out_bulk_every=1,remove_existing = true,)

new_center = (10.,10.)
e_new      = 1.3

#=
parameters = AdS5_3_1.new_parameters(
    e_new   = 1.1,
#    a4_ampy = 1.0,
#    a4_ky   = 2,
#    boostx = true,
    #fx20   = 1.0,
    #u_AH   = 0.9,
)
=#

#=
parameters_collision =AdS5_3_1.new_parameters_coll(
    dirname1  = dirname,
    dirname2  = dirname,
    x1_center = -15.0,
    y1_center = 0.0,
    x2_center = 15.0,
    y2_center = 0.0,
    fx21      = -0.1,
    fy21      = 0.0,
    fx22      = 0.1,
    fy22      = 0.0,
    u_AH      = 1.0,
)
=#

AdS5_3_1.create_checkpoint(io, potential)
#AdS5_3_1.shift(io, potential, new_center=new_center)
#AdS5_3_1.new_box(grid, io, potential)
#AdS5_3_1.change_energy(io, e_new, potential)
#AdS5_3_1.create_new_data(grid, io, parameters, potential)
#AdS5_3_1.design_collision(grid, io, parameters_collision)
#AdS5_3_1.bubble_expansion(grid, io, potential, A_dir, B_dir, PS_dir)

#=
phi11 = BulkTimeSeries(dirname,:phi,1)
phi12 = BulkTimeSeries(dirname,:phi,2)
phi21 = BulkTimeSeries(outdir,:phi,1)
phi22 = BulkTimeSeries(outdir,:phi,2)
=#



e     = VEVTimeSeries(outdir, :energy)
px    = VEVTimeSeries(outdir, :px)
e_A   = VEVTimeSeries(A_dir, :energy)
#e_B   = VEVTimeSeries(B_dir, :energy)
e_PS  = VEVTimeSeries(PS_dir, :energy)
#Jx = VEVTimeSeries(outdir,:Jx)
#Jy = VEVTimeSeries(outdir,:Jy)

t,x,y = get_coords(e,:,:,:)
plan  = plan_rfft(e[1,:,:])
Nx    = length(x)
Ny    = length(y)
e0    = real(1/(Nx*Ny) * (plan * e[1,:,:]))[1]
#Jx0   = real(1/(Nx*Ny) * (plan * Jx[1,:,:]))[1]
#Jy0   = real(1/(Nx*Ny) * (plan * Jy[1,:,:]))[1]

#=
plan = plan_fft(e_old[end,:,:]);
Nx = length(x)
Ny = length(y)
dx = x[2]-x[1]
dy = y[2]-y[1]
ek = 1/(Nx*Ny)*(plan*e_old[end,:,:])
kx = 2*pi*fftfreq(Nx,1/dx)
ky = 2*pi*fftfreq(Ny,1/dy)

x0 = x[1]
y0 = y[1]

sum = 0.0
for j in 1:length(ky)
    for i in 1:length(kx)
        global sum += ek[i,j]*exp(im*(kx[i]*x0+ky[j]*y0))
    end
end
=#
println("New set up is:")
println("(xmin, xmax) = ($(x[1]),$(x[end]+x[2]-x[1]))")
println("(ymin, ymax) = ($(y[1]),$(y[end]+y[2]-y[1]))")
println("Nx, Ny = $(length(x)), $(length(y))")
println("Average Energy Density = $e0")
println("Maximum energy = $(maximum(e[end,:,:]))")
println("Minimum energy = $(minimum(e[end,:,:]))")
println("A energy = $(e_A[end,1,1])")
println("B energy = $(minimum(e_PS[end,:,:]))")
println("Maximum px = $(maximum(px[end,:,:]))")
println("Minimum px = $(minimum(px[end,:,:]))")
#println("Average x momenta = $Jx0")
#println("Average y momenta = $Jy0")
#println("Maximum x momenta = $(maximum(abs.(Jx[end,:,:])))")
#println("Maximum y momenta = $(maximum(abs.(Jy[end,:,:])))")

#plot(x, e[end,:,1], lw=3)
#=
plot(x,y,e[1,:,:],st=:surface, camera=(50,65))
xlabel!("x")
ylabel!("y")
=#
