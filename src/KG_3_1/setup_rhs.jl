
function setup_rhs(bulkconstrains::BulkPartition{Nsys}, boundary::Boundary,
                   bulkderivs::BulkPartition{Nsys},
                   systems::SystemPartition, integration::Integration) where {Nsys}
    # function to solve the nested system
    nested = Nested(systems, bulkconstrains, bulkderivs)

    function (ff_t::EvolVars, ff::EvolVars, evoleq::EvolutionEquations, t)
        bulkevols_t = getbulkevolvedpartition(ff_t)
        bulkevols   = getbulkevolvedpartition(ff)

        # solve nested system for the constrained variables
        nested(bulkevols, boundary, evoleq)


        # first u-domain
        sys           = systems[1]
        bulkevol_t    = bulkevols_t[1]
        bulkevol      = bulkevols[1]
        bulkconstrain = bulkconstrains[1]
        deriv         = bulkderivs[1]
        compute_bulkevolved_t_1st!(bulkevol_t, bulkconstrain, bulkevol, deriv,
                                   sys, evoleq)
        # remaining u-domains
        # If we put a @threads after the @inbound below
        # it seems that there is race condition issue
        # i.e. @threads gives manifestly wrong results
        # (at least for multidomains)
        # TODO: fix the parallelization below
        @inbounds for aa in 2:Nsys
            
            sys           = systems[aa]
            bulkevol_t    = bulkevols_t[aa]
            bulkevol      = bulkevols[aa]
            bulkconstrain = bulkconstrains[aa]
            deriv         = bulkderivs[aa]

            compute_bulkevolved_t!(bulkevol_t, bulkconstrain, bulkevol, deriv,
                                   sys, evoleq)
        end
        sync_bulkevolved!(bulkevols_t, bulkconstrains, systems, evoleq)

        nothing
    end
end
