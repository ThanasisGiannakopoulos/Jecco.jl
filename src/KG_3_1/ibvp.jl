
using DifferentialEquations
using Vivi

function unpack_dom(ucoord)
    Nsys = length(ucoord)
    Nus  = [ucoord[i].nodes for i in 1:Nsys]

    Nu_lims = zeros(typeof(Nsys), Nsys + 1)
    for i in 1:Nsys
        Nu_lims[i+1] = Nu_lims[i] + Nus[i]
    end

    function (f)
        [f[Nu_lims[i]+1:Nu_lims[i+1],:,:] for i in 1:Nsys]
    end
end

function write_out(out, fieldnames, coordss)
    Nsys   = length(fieldnames)
    ucoord = [coordss[i][1] for i in 1:Nsys]
    unpack = unpack_dom(ucoord)

    function (u)
        phis = unpack(u)
        Vivi.output(out, fieldnames, phis, coordss)
        nothing
    end
end

function ibvp(p::Param, gpar::ParamGrid, idpar::ParamID)

    systems = Jecco.KG_3_1.create_sys(gpar)
    Nsys    = length(systems)

    ucoords = [systems[i].coords[1] for i in 1:Nsys]
    unpack  = unpack_dom(ucoords)

    phi0s = initial_data(systems, idpar)
    ID    = vcat(phi0s...)

    rhs! = Jecco.KG_3_1.setup_rhs(phi0s, systems, unpack)

    dt0 = p.dt

    tspan = (0.0, p.tmax)

    prob  = ODEProblem(rhs!, ID, tspan, systems)
    # http://docs.juliadiffeq.org/latest/basics/integrator.html
    integrator = init(prob, RK4(), save_everystep=false, dt=dt0, adaptive=false)
    # integrator = init(prob, AB3(), save_everystep=false, dt=dt0, adaptive=false)

    tinfo  = Vivi.TimeInfo()

    # write initial data
    Jecco.out_info(tinfo.it, tinfo.t, ID, "phi", 1, 200)

    fieldnames = ["phi c=$i" for i in 1:Nsys]
    fields     = phi0s
    coordss    = [systems[i].coords for i in 1:Nsys]

    out    = Vivi.Output(p.folder, p.prefix, p.out_every, tinfo)
    output = write_out(out, fieldnames, coordss)
    output(ID)

    for (u,t) in tuples(integrator)
        tinfo.it += 1
        tinfo.dt  = integrator.dt
        tinfo.t   = t

        Jecco.out_info(tinfo.it, tinfo.t, u, "phi", 1, 200)
        output(u)
    end

    nothing
end
