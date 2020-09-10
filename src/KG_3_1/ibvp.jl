
using OrdinaryDiffEq

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

function write_out(out, fields)
    Nsys    = length(fields)
    ucoords = [fields[i].chart.coords[1] for i in 1:Nsys]

    unpack  = unpack_dom(ucoords)

    function (u)
        phis = unpack(u)
        for i in 1:Nsys
            fields[i].data = phis[i]
        end
        out(fields)
        nothing
    end
end

function ibvp(par_grid::ParamGrid, par_id::ParamID,
              par_evol::ParamEvol, par_io::ParamIO)

    systems = Jecco.KG_3_1.create_systems(par_grid)
    Nsys    = length(systems)

    ucoords = [systems[i].ucoord for i in 1:Nsys]
    unpack  = unpack_dom(ucoords)

    phi0s = initial_data(systems, par_id)
    ID    = vcat(phi0s...)

    rhs! = Jecco.KG_3_1.setup_rhs(phi0s, systems, unpack)

    dt0   = par_evol.dt
    tspan = (0.0, par_evol.tmax)

    if par_evol.ODE_method == "RK4"
        alg = RK4()
    elseif par_evol.ODE_method == "AB3"
        alg = AB3()
    else
        error("Unknown ODE_method")
    end

    prob  = ODEProblem(rhs!, ID, tspan, systems)
    # http://docs.juliadiffeq.org/latest/basics/integrator.html
    integrator = init(prob, alg, save_everystep=false, dt=dt0, adaptive=false)

    tinfo  = Jecco.TimeInfo()

    # write initial data

    phis = unpack(ID)
    fieldnames = ["phi c=$i" for i in 1:Nsys]
    charts     = [Chart([sys.ucoord, sys.xcoord, sys.ycoord]) for sys in systems]
    fields     = [Jecco.Field(fieldnames[i], phis[i], charts[i]) for i in 1:Nsys]

    out  = Jecco.Output(par_io.folder, par_io.prefix, par_io.out_every, tinfo;
                        overwrite=par_io.overwrite)

    output = write_out(out, fields)
    output(ID)

    Jecco.out_info(tinfo.it, tinfo.t, 0.0, ID, "phi", 1, 200)

    tstart = time()
    t0     = tinfo.t
    for (u,t) in tuples(integrator)
        tinfo.it += 1
        tinfo.dt  = integrator.dt
        tinfo.t   = t

        output(u)

        telapsed = (time() - tstart) / 3600
        deltat   = t - t0
        Jecco.out_info(tinfo.it, tinfo.t, deltat/telapsed, u, "phi", 1, 200)
    end

    nothing
end
