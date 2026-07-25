/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.sandia.gov
   Steve Plimpton, sjplimp@sandia.gov, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   SPARTA-MSMC V1.0
   An efficient particle solver for continuum-to-rarefied gas flows
   based on the Multiscale Simulation Monte Carlo method.

   Developer: Hao Yang
   Organization: Beihang University
   Email: yang_hao@buaa.edu.cn
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
#include "stdlib.h"
#include "grid.h"
#include "update.h"
#include "particle.h"
#include "react.h"
#include "comm.h"
#include "random_park.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "collide_msmc.h"
#include "grid_comm_macro.h"
#include <cmath>
#include "msmc_math.h"
#include "surf_collide.h"
#include "output.h"
#include "mpi.h"

using namespace SPARTA_NS;
using namespace MathConst;
using msmc_math::Mat3x3;
using msmc_math::Vec3;
using msmc_math::solve_h;

enum{NONE,DISCRETE,SMOOTH};

#define MAXLINE 1024
/* ---------------------------------------------------------------------- */
// Constructor
CollideMSMC::CollideMSMC(SPARTA* sparta, int narg, char** arg) : Collide(sparta, narg, arg)
{
    /* ----------------------------------------------------------------------
       Example input: collide msmc air ar.msmc
       arg[0] = "msmc"         (style name)
       arg[1] = "air"          (mixture ID)
       arg[2] = "ar.msmc"      (parameter file passed to read_param_file)
       arg[3] and beyond       (optional arguments for CollideMSMCModify)
       narg must be at least 3: msmc, mixture ID, and filename
    ------------------------------------------------------------------------- */
    if (narg < 3) error->all(FLERR, "Illegal collide command");

    nmaxconserv = 0;
    conservMacro = NULL;
    nmaxfad = 0;
    fadCoeff = NULL;
    nmaxcrmax = 0;
    crmax = NULL;

    // Rank 0 reads species parameters and broadcasts them to all ranks.
    // Default parameter values
    time_ave_coef = 0.99;
    nparams = particle->nspecies;
    resetWmax = 0.99;
    Pr = 0.666667;
    alpha_Pc = 0.1;
    interpolate_flag = 1;
    solve_mode = MODE_MSMC;
    RKc1 = 0.0; RKc2 = 0.5; RKa21 = 0.5;
    RKw1 = 0.0; RKw2 = 1.0;

    if (nparams == 0)
        error->all(FLERR, "Cannot use collide command with no species defined");
    // Allocate parameter storage
    memory->create(params, nparams, "collide_msmc:params");
    for (int i = 0; i < nparams; i++) {
        params[i].mu_ref = -1.0;
        params[i].omega = 0.0;
        params[i].T_ref = 0.0;
        params[i].d_ref = 0.0;
        params[i].Pr = Pr;
        params[i].Zrot_cont = -1.0;
        params[i].Zvib_cont = -1.0;
        params[i].RotRelNum_model = Z_CONSTANT;
        params[i].VibRelNum_model = Z_CONSTANT;
    }
    // Parse optional input arguments:
    //   mode msmc|dsmc selects the primary solver path.
    //   Other options are forwarded to collide_msmc_modify.
    if (narg > 3) {
        std::vector<char*> modify_args;
        int iarg = 3;
        while (iarg < narg) {
            if (strcmp(arg[iarg], "mode") == 0) {
                if (iarg + 1 >= narg) error->all(FLERR, "Illegal collide command");
                if (strcmp(arg[iarg + 1], "msmc") == 0) solve_mode = MODE_MSMC;
                else if (strcmp(arg[iarg + 1], "dsmc") == 0) solve_mode = MODE_DSMC;
                else error->all(FLERR, "Illegal collide command");
                iarg += 2;
            }
            else {
                modify_args.push_back(arg[iarg]);
                ++iarg;
            }
        }
        if (!modify_args.empty()) {
            CollideMSMCModify msmc_modify = CollideMSMCModify(sparta);
            msmc_modify.command(static_cast<int>(modify_args.size()), modify_args.data());
        }
    }
    // Rank 0 reads the parameter file.
    if (comm->me == 0) read_param_file(arg[2]);
    // Broadcast parameters to all ranks.
    MPI_Bcast(params, nparams * sizeof(Params), MPI_BYTE, 0, world);
    // Initialize diagnostic counters.
    count_try_relaxation = count_done_relaxation = count_fail_relaxation = 0;
    count_msmc_cell = count_lowNp_cell = count_lowtemp_cell = 0;
    count_conserv_clip_cell = 0;

    maxglocal = 0;
    resetWmax_flag = NULL;
    nplocalmax = 0;
    //relax_flag = NULL;
}


/* ---------------------------------------------------------------------- */
// Destructor
CollideMSMC::~CollideMSMC()
{
    // Skip cleanup for copy instances created by SPARTA internals.
    if (copymode) return;
    // Release parameter arrays.
    memory->destroy(params);
    //memory->destroy(prefactor);
    // Release conservation and sampling buffers.
    memory->destroy(conservMacro);
    memory->destroy(fadCoeff);
    memory->destroy(crmax);
    // Release Wmax reset flags.
    memory->destroy(resetWmax_flag);
}

/* ---------------------------------------------------------------------- */
void CollideMSMC::reset_count() {
    // Acceptance-rejection sampling counters.
    count_try_relaxation = count_done_relaxation = count_fail_relaxation = 0;
    // Cell-level diagnostic counters.
    count_msmc_cell = count_lowNp_cell = count_lowtemp_cell = 0;
    count_conserv_clip_cell = 0;
}

/* ----------------------------------------------------------------------
* currently CollideMSMC::init() will do nothing but call Collide::init();
   ---------------------------------------------------------------------- */
void CollideMSMC::init()
{
    Collide::init();
}

/* ----------------------------------------------------------------------
* perform MSMC/DSMC collisions for all owned child cells
* shared preprocessing is followed by either DSMC fallback or MSMC update
------------------------------------------------------------------------- */

void CollideMSMC::collisions()
{
// 1. Evaluate macroscopic quantities.
    computeMacro();
// 2. Prepare cell-local work arrays.
    // nglocal is the number of grid cells owned by this rank.
    // Reset resetWmax_flag.
    if (nglocal > maxglocal) {
        maxglocal = ceil(nglocal * 1.2);
        memory->destroy(resetWmax_flag);
        memory->create(resetWmax_flag, maxglocal, "collideMSMC:resetWmax_flag");
    }
    for (int icell = 0; icell < nglocal; icell++) {
        resetWmax_flag[icell] = 1;
    }
    if (nglocal > nmaxfad) {
        while (nglocal > nmaxfad) nmaxfad += DELTAPART;
        memory->destroy(fadCoeff);
        memory->create(fadCoeff, nmaxfad, "collideMSMC:fadCoeff");
    }
    if (nglocal > nmaxcrmax) {
        while (nglocal > nmaxcrmax) nmaxcrmax += DELTAPART;
        memory->destroy(crmax);
        memory->create(crmax, nmaxcrmax, "collideMSMC:crmax");
    }
    Grid::ChildCell* cells = grid->cells;
    Grid::ChildInfo* cinfo = grid->cinfo;
    Particle::OnePart* particles = particle->particles;
    int* next = particle->next;
// 3. Loop over local grid cells.
    for (int ic = 0; ic < nglocal; ic++) {
        // Check particle count.
        int Np = cinfo[ic].count;
        if (Np < 1) continue;
        // Compute effective collision volume.
        double volume = cinfo[ic].volume / cinfo[ic].weight * cells[ic].dt_weight;
        if (volume == 0.0) error->one(FLERR, "Collision cell volume is zero");
        // Build the particle index list.
        if (Np > npmax) {
            while (Np > npmax) npmax += DELTAPART;
            memory->destroy(plist);
            memory->create(plist, npmax, "collide:plist");
        }
        // Estimate crmax, build plist, and compute vc.
        double* ui = cells[ic].macro.v;
        double mass = particle->species[particles[cinfo[ic].first].ispecies].mass;
        Params& ps = params[particles[cinfo[ic].first].ispecies];
        int ip = cinfo[ic].first;
        int n = 0;
        double Ttr_cell = grid->cinfo[ic].macro.Ttr;
        if (Ttr_cell > 0.0) crmax[ic] = 2.0 * sqrt(update->boltz * Ttr_cell / mass);
        else crmax[ic] = 300.0;
        while (ip >= 0) {
            // Update crmax estimate.
            Particle::OnePart* p = &particles[ip];
            double dv_sqr = (p->v[0] - ui[0]) * (p->v[0] - ui[0]) +
                            (p->v[1] - ui[1]) * (p->v[1] - ui[1]) +
                            (p->v[2] - ui[2]) * (p->v[2] - ui[2]);
            double dv = 2.0 * sqrt(dv_sqr);
            if (crmax[ic] < dv) crmax[ic] = dv;
            // Store particle index.
            plist[n++] = ip;
            ip = next[ip];
        }
        double sigma_T_ref = MY_PI * ps.d_ref * ps.d_ref;
        double crbar_ref = 4.0 * sqrt(update->boltz * ps.T_ref / MY_PI / mass);
        //
        double nrho = grid->cinfo[ic].macro.rho / mass;
        double sigmaT_cr_max = sigma_T_ref * pow(crmax[ic] / crbar_ref, 1.0 - 2.0 * ps.omega) * crmax[ic];
        double vc = nrho * sigmaT_cr_max;
        ExpRKProb epr = build_exprk_prob(ic, vc);
        // ~~~~~ DSMC mode ~~~~~
        // Either explicit DSMC mode, or fallback to DSMC when the cell macro
        // is not reliable enough for MSMC relaxation/compensation.
        if (solve_mode == MODE_DSMC || !grid->cinfo[ic].macro.do_relaxation) {
            if (Np < 2) continue;
            double Nntc = 0.5 * Np * vc * epr.dt;
            int Nattempt = static_cast<int>(Nntc + random->uniform());
            if (Nattempt > 0) {
                std::vector<int> coll_pair(2 * Nattempt);
                choose_collision_pair_DSMC(Np, 2 * Nattempt, coll_pair, random);
                for (int k = 0; k < Nattempt; ++k) {
                    int ip1 = plist[coll_pair[2 * k]];
                    int ip2 = plist[coll_pair[2 * k + 1]];
                    perform_binary_collision(&particles[ip1], &particles[ip2], ic);
                }
            }
            continue;
        }
        // ~~~~~ MSMC mode ~~~~~
        // Compute probabilities for the MSMC particle groups.
        double Pcoll1 = epr.Pcoll1;
        double Pcoll2 = epr.Pcoll2;
        double Prelax = epr.Prelax;
        // Sample particle counts for the MSMC groups.
        int Ncoll1 = 0, Ncoll2 = 0, Nrelax = 0;
        Iround_group(Pcoll1, Pcoll2, Prelax, Np, Ncoll1, Ncoll2, Nrelax);
        // Shuffle particle indices for collision and relaxation groups.
        int Ncoll_tot = 2 * Ncoll1 + 2 * Ncoll2 + Nrelax;
        for (int i = 0; i < Ncoll_tot; i++) {
            int t = i + (Np - i) * random->uniform();
            std::swap(plist[i], plist[t]);
        }
        // Compute MSMC anti-dissipation coefficients.
        Thermo th = updateThermo(particle->species[particles[cinfo[ic].first].ispecies],
                                 ps, cells[ic].macro, cinfo[ic].macro);
        FadCoeff& fad = fadCoeff[ic];
        for (int id = 0; id < 6; id++) fad.yh_s[id] = 0.0;
        fad.yh_etr = fad.yh_erot = fad.yh_evib = 0.0;
        for (int id = 0; id < 3; id++) {
            fad.yh_qtr[id] = fad.yh_qrot[id] = fad.yh_qvib[id] = 0.0;
        }
        // --- MSMC polyatomic ---
        compute_stress_compensation(cinfo[ic].macro, epr, th, fad);
        compute_energy_compensation(particle->species[particles[cinfo[ic].first].ispecies],
                                    cells[ic].macro, cinfo[ic].macro, epr, th, fad);
        compute_heatflux_compensation(particle->species[particles[cinfo[ic].first].ispecies],
                                      cells[ic].macro, cinfo[ic].macro, epr, th, fad);
        // --- MSMC polyatomic ---
        // F1 binary collisions, P(F1,F1)/v.
        int idx = 0;
        for (int k = 0; k < Ncoll1; ++k){
            int ip1 = plist[idx++];
            int ip2 = plist[idx++];
            perform_binary_collision(&particles[ip1], &particles[ip2], ic);
        }
        // F2 binary collisions, P(F2,F2)/v.
        //    F2 = P2null * f + P2collF1 * P(F1,F1)/v + P2relax * fad
        // 7(a) F2 sub-grouping
        double P2c1 = epr.P2c1;
        double P2r = epr.P2r;
        int Nnull = 0, N2c1 = 0, N2r = 0;
        Iround_group(0.0, P2c1, P2r, 2 * Ncoll2, Nnull, N2c1, N2r);
        // 7(b) Generate F2 particles from P(F1,F1)/v.
        int id2 = idx;
        for (int k = 0; k < N2c1; k++){
            int ip1 = plist[id2++];
            int ip2 = plist[id2++];
            perform_binary_collision(&particles[ip1], &particles[ip2], ic);
        }
        // 7(c) Generate F2 particles from fad.
        for (int k = 0; k < N2r; k++){
            int ip = plist[id2++];
            perform_msmc_relaxation(&particles[ip], ic, th);
        }
        // 7(d) Reshuffle F2 particles.
        int N_F2 = 2 * Ncoll2;
        for (int i = 0; i < N_F2; i++){
            int t = i + static_cast<int>((N_F2 - i) * random->uniform());
            std::swap(plist[idx + i], plist[idx + t]);
        }
        // 7(e) Apply P(F2,F2)/v.
        for (int k = 0; k < Ncoll2; k++){
            int ip1 = plist[idx++];
            int ip2 = plist[idx++];
            perform_binary_collision(&particles[ip1], &particles[ip2], ic);
        }
        // Sample particles from fad.
        for (int k = 0; k < Nrelax; k++){
            int ip = plist[idx++];
            perform_msmc_relaxation(&particles[ip], ic, th);
        }
        // Apply conservation correction.
        conserv_correction(ic, plist, Np);
    } // end for ic loop
// 4. Reset Wmax where appropriate.
    for (int ic = 0; ic < nglocal; ic++) {
        if (resetWmax > 0.0 && resetWmax_flag[ic]) {
            cinfo[ic].macro.Wmax *= resetWmax;
        }
    }
//
    print_warning();
}


/**
 * @brief Randomly selects candidate collision pairs within a specific cell.
 * @param Ntot Total number of particles in the current cell.
 * @param Nc Number of indices to generate (2 * Nattempt).
 * @param coll_pair Vector to store the local indices of the selected particles.
 */
void CollideMSMC::choose_collision_pair_DSMC(int Ntot, int Nc, std::vector<int>& coll_pair, RanPark* rng) {
	if (Ntot < 2 || Nc <= 0) return;
	//vector<int> plist(Ntot);
	//for (int i = 0; i < Ntot; ++i) { plist[i] = i; }
	//for (int i = 0; i < Nc; ++i) {
	//	int t = i + (Ntot - i) * rng->uniform();
	//	std::swap(plist[i], plist[t]);
	//}
	//for (int i = 0; i < Nc; ++i) { coll_pair[i] = plist[i]; }
	for (int i = 0; i < Nc / 2; ++i) {
		coll_pair[2 * i] = rng->uniform() * Ntot;
		coll_pair[2 * i + 1] = rng->uniform() * Ntot;
		while (coll_pair[2 * i] == coll_pair[2 * i + 1]) { coll_pair[2 * i + 1] = rng->uniform() * Ntot; }
	}
}


/* ----------------------------------------------------------------------
   Stochastic rounding helper.
   z: expected value.
   type: rounding mode (NONE, EVEN, or ODD).
------------------------------------------------------------------------- */
int CollideMSMC::Iround(double z, RoundType type) {
    if (type == ROUND_NONE) {
        return floor(z + random->uniform());
    }
    else if (type == ROUND_EVEN) {
        return 2 * floor(0.5 * z + random->uniform());
    }
    else if (type == ROUND_ODD) {
       return 2 * floor(0.5 * (z - 1.0) + random->uniform()) + 1;
    }
    else {
        return -1;
    }
}

/* ----------------------------------------------------------------------
   Sample particle counts for MSMC grouping.
   Inputs: probabilities P1, P2, Pr and total particle count N.
   Outputs: Nc1, Nc2 collision-pair counts and Nr relaxation count.
------------------------------------------------------------------------- */
void CollideMSMC::Iround_group(double P1, double P2, double Pr, int N, int& Nc1, int& Nc2, int& Nr) {
    // N1 and N2 must be represented by even particle counts, and the
    // rounded counts must not exceed N.
    Nc1 = Nc2 = Nr = 0;
    double Pleft = 1.0 - P1 - P2 - Pr;
    double Pc = P1 + P2, P_ = Pr + Pleft;
    if (Pc < P_) { // Sample the even collision group first.
        int Nc = Iround(N * Pc, ROUND_EVEN) / 2;
        if (Nc > 0 && Pc > 0.0) {
            // Nc is already the number of collision pairs, so only one
            // stochastic rounding should be applied here. Re-applying
            // ROUND_EVEN would bias the pair-count split.
            if (P1 < P2) {
                Nc1 = Iround((double)Nc * P1 / Pc, ROUND_NONE);
                Nc2 = Nc - Nc1;
            }
            else {
                Nc2 = Iround((double)Nc * P2 / Pc, ROUND_NONE);
                Nc1 = Nc - Nc2;
            }
        }
        int N_ = N - 2 * Nc;
        if (N_ > 0 && P_ > 0.0) {
            if (Pr < Pleft) {
                Nr = Iround(N_ * Pr / P_, ROUND_NONE);
            }
            else {
                int Nleft = Iround(N_ * Pleft / P_, ROUND_NONE);
                Nr = N_ - Nleft;
            }
        }
    }
    else { // Sample Nleft and Nrelax first.
        int N_ = 0;
        if (N % 2 == 0) {
            N_ = Iround(N * P_, ROUND_EVEN);	// Keep N_ even when N is even.
        }
        else {
            N_ = Iround(N * P_, ROUND_ODD);		// Keep N_ odd when N is odd.
        }
        if (N_ > 0 && P_ > 0.0) {
            if (Pr < Pleft) {
                Nr = Iround(N_ * Pr / P_, ROUND_NONE);
            }
            else {
                int Nleft = Iround(N_ * Pleft / P_, ROUND_NONE);
                Nr = N_ - Nleft;
            }
        }
        int Nc = (N - N_) / 2;
        if (Nc > 0 && Pc > 0.0) {
            if (P1 < P2) {
                Nc1 = Iround((double)Nc * P1 / Pc, ROUND_NONE);
                Nc2 = Nc - Nc1;
            }
            else {
                Nc2 = Iround((double)Nc * P2 / Pc, ROUND_NONE);
                Nc1 = Nc - Nc2;
            }
        }
    }
}

/**
 * @brief Enforces momentum and energy conservation after the relaxation step.
 * @details Linearly shifts and scales particle velocities within each cell to
 * match the target macroscopic velocity (ui) and temperature (T).
 */
void CollideMSMC::conserv_correction(int icell, int* plist, int np) {
    ConservMacro& cm = conservMacro[icell];
    if (!cm.done_relaxation || !plist) return;
    // Step 1: collect the actual post-collision moments in this cell.
    Particle::OnePart* particles = particle->particles;
    Particle::Species& species = particle->species[particles[plist[0]].ispecies];
    const double mass = species.mass;
    double sum_v[3] = { 0.0, 0.0, 0.0 };
    double sum_vsq = 0.0;
    double sum_erot = 0.0;
    double sum_evib = 0.0;
    for (int k = 0; k < np; ++k) {
        Particle::OnePart& part = particles[plist[k]];
        for (int id = 0; id < 3; ++id) {
            sum_v[id] += part.v[id];
            sum_vsq += part.v[id] * part.v[id];
        }
        sum_erot += part.erot;
        sum_evib += part.evib;
    }

    // Step 2: convert the moments to post-collision mean energies.
    double u_post[3];
    for (int id = 0; id < 3; ++id) u_post[id] = sum_v[id] / np;
    const double u_post_sqr = u_post[0] * u_post[0] +
                              u_post[1] * u_post[1] +
                              u_post[2] * u_post[2];
    double Etr_post = 0.5 * mass * (sum_vsq - np * u_post_sqr) / np;
    double Erot_post = sum_erot / np;
    double Evib_post = sum_evib / np;
    double Etot_pre = cm.Etr_pre + cm.Erot_pre + cm.Evib_pre;

    // Step 3: if vibration alone exceeds the total pre-collision energy,
    // clip it before redistributing the remaining energy to Etr/Erot.
    if (Evib_post > Etot_pre) {
        ++count_conserv_clip_cell;
        const double Evib0 = update->boltz * species.vibtemp[0];
        if (species.vibdof > 0 && vibstyle == DISCRETE && Evib0 > 0.0) {
            std::vector<int> active;
            for (int k = 0; k < np; ++k) {
                Particle::OnePart& part = particles[plist[k]];
                int iq = static_cast<int>(part.evib / Evib0 + 0.5);
                if (iq > 0) active.push_back(k);
            }
            while (Evib_post > Etot_pre && !active.empty()) {
                int i = static_cast<int>(random->uniform() * active.size());
                if (i >= active.size()) i = active.size() - 1;
                int pick = active[i];

                Particle::OnePart& part = particles[plist[pick]];
                int iq = static_cast<int>(part.evib / Evib0 + 0.5);
                part.evib = static_cast<double>(iq - 1) * Evib0;
                Evib_post -= Evib0 / np;

                if (iq == 1) {
                    active[i] = active.back();
                    active.pop_back();
                }
            }
        } else if (Evib_post > 0.0) {
            const double coef_vib = Etot_pre / Evib_post;
            for (int k = 0; k < np; ++k) particles[plist[k]].evib *= coef_vib;
            Evib_post = Etot_pre;
        }
    }

    // Step 4: put the remaining energy into translation/rotation while
    // preserving the post-collision Etr:Erot ratio.
    double scale = (Etot_pre - Evib_post) / (Etr_post + Erot_post);
    double Etr_target = Etr_post * scale;
    double Erot_target = Erot_post * scale;

    // Step 5: apply the conservation targets to the particles.
    double coef_tr = 1.0, coef_rot = 1.0;
    if (Etr_post > 0.0) coef_tr = sqrt(Etr_target / Etr_post);
    if (Erot_post > 0.0) coef_rot = Erot_target / Erot_post;

    for (int k = 0; k < np; ++k) {
        Particle::OnePart& part = particles[plist[k]];
        for (int id = 0; id < 3; ++id) {
            part.v[id] = cm.u_pre[id] + (part.v[id] - u_post[id]) * coef_tr;
        }
        part.erot *= coef_rot;
    }
}

/**
 * @brief Computes the post-collision velocities for a pair of particles.
 * @details Uses the Variable Hard Sphere (VHS) model for scattering.
 * Momentum and Energy are conserved during the process.
 */
void CollideMSMC::perform_binary_collision(Particle::OnePart* p1, Particle::OnePart* p2, int ic) {
    double Cr_vec[3]{};
    for (int id = 0; id < 3; ++id) {
        Cr_vec[id] = p1->v[id] - p2->v[id];
    }
    double Cr = sqrt(Cr_vec[0] * Cr_vec[0] + Cr_vec[1] * Cr_vec[1] + Cr_vec[2] * Cr_vec[2]);
    double exponent = 2.0 - 2.0 * params[p1->ispecies].omega;
    if (crmax[ic] <= 0.0) return;
    double prob = pow(Cr / crmax[ic], exponent);
    if (random->uniform() > prob) {
        return;
    }
    // --- MSMC polyatomic ---
    double Cr_post_mag = Cr;
    if (particle->species[p1->ispecies].internaldof) {
        Cr_post_mag = inelastic_collision_pair_selection(Cr, ic, p1, p2);
    }
    // --- MSMC polyatomic ---

    double rand_cosX = 2.0 * random->uniform() - 1.0;
    double sin_sq = 1.0 - rand_cosX * rand_cosX;
    double rand_sinX = (sin_sq > 0.0) ? sqrt(sin_sq) : 0.0;
    double rand_theta = 2.0 * MY_PI * random->uniform();
    double Cm_vec[3]{};
    for (int id = 0; id < 3; ++id) {
        Cm_vec[id] = 0.5 * (p1->v[id] + p2->v[id]);
    }
    double Cr_post[3]{};
    Cr_post[0] = Cr_post_mag * rand_sinX * cos(rand_theta);
    Cr_post[1] = Cr_post_mag * rand_sinX * sin(rand_theta);
    Cr_post[2] = Cr_post_mag * rand_cosX;
    for (int id = 0; id < 3; ++id) {
        p1->v[id] = Cm_vec[id] + 0.5 * Cr_post[id];
        p2->v[id] = Cm_vec[id] - 0.5 * Cr_post[id];
    }
}


/**
 * @brief Internal energy mode selection for a collision pair.
 * Uses the prohibiting-double-relaxation rule to select elastic, rotational,
 * or vibrational energy exchange for the colliding pair, and updates the
 * corresponding post-collision internal energy by the Borgnakke-Larsen model.
 * @param Cr   Pre-collision relative speed.
 * @param c    Cell providing the local thermodynamic state.
 * @param p1   First particle.
 * @param p2   Second particle.
 * @return     Post-collision relative speed.
 */
double CollideMSMC::inelastic_collision_pair_selection(double Cr, int ic,
                                                       Particle::OnePart* p1,
                                                       Particle::OnePart* p2)
{
    const Particle::Species& species = particle->species[p1->ispecies];
    const Params& ps = params[p1->ispecies];
    const NoCommMacro& nmacro = grid->cinfo[ic].macro;
    const double mass = species.mass;
    const double Etr_pre = 0.25 * mass * Cr * Cr;
    double Etr_post = Etr_pre;
    double Eint_post = 0.0;

    const double dof_tr_pair = 5.0 - 2.0 * ps.omega;
    const double vis = ps.mu_ref * pow(nmacro.Ttr / ps.T_ref, ps.omega);
    double tau_vis = 0.0;
    if (nmacro.ptr > 0.0 && vis > 0.0) tau_vis = vis / nmacro.ptr;
    const double tau_coll = 4.0 / MY_PI * tau_vis;
    const double Zrot_cont = RotRelNum(species, ps, nmacro.Ttr);
    const double Zvib_cont = VibRelNum(species, ps, nmacro.Ttr, nmacro.ptr, tau_coll);

    double pvib1 = 0.0;
    double pvib2 = 0.0;
    if (species.vibdof > 0 && vibstyle != NONE && Zvib_cont > 0.0 && Zvib_cont < 1.0e19 && nmacro.Ttr > 0.0) {
        if (vibstyle == DISCRETE) {
            const double dof_v = dof_vib(species, nmacro.Ttr);
            //const double dof_v = dof_vib(species, grid->cells[ic].macro.Temp);
            const double theta0_vib = species.vibtemp[0];
            if (theta0_vib > 0.0) {
                const double Gamma = 0.5 * dof_v * dof_v * exp(theta0_vib / nmacro.Ttr);
                //const double Gamma = 0.5 * dof_v * dof_v * exp(theta0_vib / grid->cells[ic].macro.Temp);
                pvib1 = (dof_tr_pair + Gamma) / (dof_tr_pair * Zvib_cont);
            }
        } else {
            const double dof_v = dof_vib(species, nmacro.Ttr);
            if (dof_v > 0.0) {
                pvib1 = (dof_tr_pair + dof_v) / (dof_tr_pair * Zvib_cont);
            }
        }
        pvib1 = MAX(0.0, pvib1);
        pvib2 = pvib1;
    }

    double prot1 = 0.0;
    double prot2 = 0.0;
    if (species.rotdof > 0 && Zrot_cont > 0.0 && Zrot_cont < 1.0e19) {
        prot1 = (dof_tr_pair + species.rotdof) / (dof_tr_pair * Zrot_cont);
        prot1 = MAX(0.0, prot1);
        prot2 = prot1;
    }

    const double A1 = pvib1;
    const double A2 = A1 + pvib2;
    const double A3 = A2 + prot1;
    const double A4 = A3 + prot2;
    const double Rf = random->uniform();

    if (Rf < A1) {
        if (vibstyle == DISCRETE) {
            inelastic_collision_BL_discrete(Etr_pre, dof_tr_pair, p1->evib,
                                            species.vibtemp[0], Etr_post, Eint_post);
        } else {
            inelastic_collision_BL_smooth(Etr_pre, dof_tr_pair, p1->evib,
                                          dof_vib(species, nmacro.Ttr), Etr_post, Eint_post);
        }
        p1->evib = Eint_post;
    } else if (Rf < A2) {
        if (vibstyle == DISCRETE) {
            inelastic_collision_BL_discrete(Etr_pre, dof_tr_pair, p2->evib,
                                            species.vibtemp[0], Etr_post, Eint_post);
        } else {
            inelastic_collision_BL_smooth(Etr_pre, dof_tr_pair, p2->evib,
                                          dof_vib(species, nmacro.Ttr), Etr_post, Eint_post);
        }
        p2->evib = Eint_post;
    } else if (Rf < A3) {
        inelastic_collision_BL_smooth(Etr_pre, dof_tr_pair, p1->erot,
                                      static_cast<double>(species.rotdof), Etr_post, Eint_post);
        p1->erot = Eint_post;
    } else if (Rf < A4) {
        inelastic_collision_BL_smooth(Etr_pre, dof_tr_pair, p2->erot,
                                      static_cast<double>(species.rotdof), Etr_post, Eint_post);
        p2->erot = Eint_post;
    }

    if (Etr_post <= 0.0) return 0.0;
    return sqrt(4.0 * Etr_post / mass);
}

/**
 * @brief Borgnakke-Larsen (BL) Model for Inelastic Collisions
 * Redistributes the total collision energy (translational + rotational) between the
 * translational and rotational modes based on the continuous Borgnakke-Larsen model.
 * @param e_tr      Pre-collision relative translational energy.
 * @param dof_tr    Equivalent translational degrees of freedom of the collision pair (e.g., 5 - 2*omega).
 * @param e_rot     Pre-collision rotational energy.
 * @param dof_rot   Rotational degrees of freedom (e.g., 2.0 for diatomic molecules).
 * @param Etr_post  [out] Post-collision relative translational energy.
 * @param Erot_post [out] Post-collision rotational energy.
 */
void CollideMSMC::inelastic_collision_BL_smooth(double e_tr, double dof_tr,
                                                double e_int, double dof_int,
                                                double& Etr_post, double& Eint_post)
{
    const double e_coll = e_tr + e_int;
    if (e_coll <= 0.0 || dof_int <= 0.0) {
        Etr_post = MAX(0.0, e_tr);
        Eint_post = MAX(0.0, e_int);
        return;
    }

    const double z1 = dof_tr / 2.0 - 1.0;
    const double z2 = dof_int / 2.0 - 1.0;
    double etr_ratio = 1.0;

    if (fabs(dof_int - 2.0) < 1.0e-12) {
        etr_ratio = pow(random->uniform(), 1.0 / (z1 + 1.0));
    } else {
        const double A1 = (z1 > 0.0) ? (z1 + z2) / z1 : 1.0;
        const double A2 = (z2 > 0.0) ? (z1 + z2) / z2 : 1.0;
        while (true) {
            etr_ratio = random->uniform();
            const double w = pow(A1 * etr_ratio, z1) * pow(A2 * (1.0 - etr_ratio), z2);
            if (random->uniform() < w) break;
        }
    }

    Etr_post = e_coll * etr_ratio;
    Eint_post = e_coll - Etr_post;
}

/**
 * @brief Discrete Borgnakke-Larsen Model for Vibrational Energy
 * Uses Acceptance-Rejection method optimized for quantized energy levels.
 * @param e_tr      Pre-collision relative translational energy.
 * @param dof_tr    Equivalent translational degrees of freedom of the collision pair (e.g., 5 - 2*omega).
 * @param e_vib     Pre-collision vibrational energy.
 * @param Etr_post  [out] Post-collision relative translational energy.
 * @param Evib_post [out] Post-collision vibrational energy.
 */
void CollideMSMC::inelastic_collision_BL_discrete(double e_tr, double dof_tr,
                                                  double e_vib, double theta_v,
                                                  double& Etr_post, double& Evib_post)
{
    const double e_coll = e_tr + e_vib;
    const double Evib0 = update->boltz * theta_v;
    if (e_coll <= 0.0 || Evib0 <= 0.0) {
        Etr_post = MAX(0.0, e_tr);
        Evib_post = MAX(0.0, e_vib);
        return;
    }

    const int imax = static_cast<int>(e_coll / Evib0);
    int i_post = 0;
    if (imax > 0) {
        const double z = dof_tr / 2.0 - 1.0;
        while (true) {
            i_post = static_cast<int>((imax + 1) * random->uniform());
            double etr_ratio = 1.0 - i_post * Evib0 / e_coll;
            if (etr_ratio < 0.0) etr_ratio = 0.0;
            const double w = pow(etr_ratio, z);
            if (random->uniform() < w) break;
        }
    }

    Evib_post = i_post * Evib0;
    Etr_post = e_coll - Evib_post;
}



/* ----------------------------------------------------------------------
   Relaxation sampler using a partial Grad form.
   The energy-mode correction is absorbed into modal temperatures, while
   stress and heat-flux corrections remain in the rejection weight.
------------------------------------------------------------------------- */
void CollideMSMC::perform_msmc_relaxation(Particle::OnePart* p, int ic,
                                          const Thermo& th)
{
    Grid::ChildCell* cells = grid->cells;
    Grid::ChildInfo* cinfo = grid->cinfo;
    const CommMacro& cmacro = cells[ic].macro;
    const NoCommMacro& nmacro = cinfo[ic].macro;
    const FadCoeff& fad = fadCoeff[ic];
    Particle::Species& species = particle->species[p->ispecies];
    const CommMacro* interMacro = &cmacro;
    if (interpolate_flag && grid->gridCommMacro) {
        interMacro = grid->gridCommMacro->interpolation(p);
        if ((!interMacro) || (!(interMacro->Temp > 0.0))) interMacro = &cmacro;
    }

    const double boltz = update->boltz;
    const double mass = species.mass;
    // Step 1: build the reference equilibrium state at the particle location.
    double Teq = interMacro->Temp;
    if (Teq <= 0.0) Teq = nmacro.Ttr;
    if (!(Teq > 0.0) || !std::isfinite(Teq)) {
        ++count_fail_relaxation;
        return;
    }

    for (int id = 0; id < 6; id++) {
        if (!std::isfinite(fad.yh_s[id])) {
            ++count_fail_relaxation;
            return;
        }
    }
    if (!std::isfinite(fad.yh_etr) || !std::isfinite(fad.yh_erot) || !std::isfinite(fad.yh_evib)) {
        ++count_fail_relaxation;
        return;
    }
    for (int id = 0; id < 3; id++) {
        if (!std::isfinite(fad.yh_qtr[id]) || !std::isfinite(fad.yh_qrot[id]) || !std::isfinite(fad.yh_qvib[id])) {
            ++count_fail_relaxation;
            return;
        }
    }

    const bool rot_active = (species.rotdof > 0);
    const bool vib_active = (species.vibdof > 0 && vibstyle != NONE);
    const double dof_vib_eq = dof_vib(species, Teq);
    const double evib_eq = 0.5 * dof_vib_eq * boltz * Teq;

    // Step 2: convert the energy coefficients into modal temperatures.
    const double Ttr_rel = Teq * (1.0 + fad.yh_etr);
    const double Trot_rel = rot_active ? Teq * (1.0 + fad.yh_erot) : Teq;
    if (!(Ttr_rel > 0.0) || !(Trot_rel > 0.0) ||
        !std::isfinite(Ttr_rel) || !std::isfinite(Trot_rel)) {
        ++count_fail_relaxation;
        return;
    }

    double Tvib_rel = Teq;
    if (vib_active) {
        if (!(th.Cvib > 0.0)) {
            ++count_fail_relaxation;
            return;
        }
        const double evib_rel = evib_eq + th.Cvib * boltz * Teq * fad.yh_evib;
        if (!(evib_rel > 0.0) || !std::isfinite(evib_rel)) {
            ++count_fail_relaxation;
            return;
        }
        Tvib_rel = compute_Tvib(species, evib_rel);
        if (!(Tvib_rel > 0.0) || !std::isfinite(Tvib_rel)) {
            ++count_fail_relaxation;
            return;
        }
    }

    double Cvib_rel = 0.0;
    if (vib_active && vibstyle == DISCRETE) {
        const double theta0_vib = species.vibtemp[0];
        if (theta0_vib > 0.0 && Tvib_rel > 1.0e-12) {
            const double dof_vib_rel = dof_vib(species, Tvib_rel);
            Cvib_rel = exp(theta0_vib / Tvib_rel) * dof_vib_rel * dof_vib_rel / 4.0;
        }
    } else if (vib_active) {
        Cvib_rel = 0.5 * species.vibdof;
    }

    const double nrho = nmacro.rho / mass;
    if (!(nrho > 0.0) || !std::isfinite(nrho)) {
        ++count_fail_relaxation;
        return;
    }

    const double peq = nrho * boltz * Teq;
    const double prel = nrho * boltz * Ttr_rel;
    const double RTeq = boltz * Teq / mass;
    const double RTtr = boltz * Ttr_rel / mass;
    const double RTrot = boltz * Trot_rel / mass;
    const double RTvib = boltz * Tvib_rel / mass;
    if (!(peq > 0.0) || !(prel > 0.0) || !(RTeq > 0.0) ||
        !(RTtr > 0.0) || !(RTrot > 0.0) || !(RTvib > 0.0)) {
        ++count_fail_relaxation;
        return;
    }

    // Step 3: recover the physical stress and heat-flux corrections.
    double sigma_rel[6]{};
    double qtr_rel[3]{}, qrot_rel[3]{}, qvib_rel[3]{};
    const double sqrtRTeq = sqrt(RTeq);
    for (int id = 0; id < 6; id++) sigma_rel[id] = peq * fad.yh_s[id];
    for (int id = 0; id < 3; id++) {
        qtr_rel[id] = 2.5 * peq * sqrtRTeq * fad.yh_qtr[id];
        qrot_rel[id] = th.Crot * peq * sqrtRTeq * fad.yh_qrot[id];
        // Use the relaxation-state vibrational heat-capacity scale.
        qvib_rel[id] = Cvib_rel * peq * sqrtRTeq * fad.yh_qvib[id];
    }

    const double sqrtRTtr = sqrt(RTtr);
    const double inv_2RTtr = 1.0 / (2.0 * RTtr);
    const double inv_prel = 1.0 / prel;
    double ci[3]{};
    double erot = 0.0, evib = 0.0;
    int count_loop = 0;
    bool accepted = false;

    // Step 4: sample from the partial-Grad base state and accept/reject
    // with only stress and heat-flux corrections in the weight.
    while (true) {
        ++count_try_relaxation;
        ++count_loop;

        for (int id = 0; id < 3; id++) ci[id] = random->gaussian() * sqrtRTtr;
        erot = particle->erot(p->ispecies, Trot_rel, random);
        evib = particle->evib(p->ispecies, Tvib_rel, random);

        const double ci_sqr = ci[0] * ci[0] + ci[1] * ci[1] + ci[2] * ci[2];
        const double phi_s0 = (ci[0] * ci[0] - ci_sqr / 3.0) * inv_2RTtr;
        const double phi_s1 = (ci[1] * ci[1] - ci_sqr / 3.0) * inv_2RTtr;
        const double phi_s2 = (ci[2] * ci[2] - ci_sqr / 3.0) * inv_2RTtr;
        const double phi_s3 = 2.0 * ci[0] * ci[1] * inv_2RTtr;
        const double phi_s4 = 2.0 * ci[0] * ci[2] * inv_2RTtr;
        const double phi_s5 = 2.0 * ci[1] * ci[2] * inv_2RTtr;

        double w = 1.0;
        w += inv_prel * (sigma_rel[0] * phi_s0 + sigma_rel[1] * phi_s1 +
                         sigma_rel[2] * phi_s2 + sigma_rel[3] * phi_s3 +
                         sigma_rel[4] * phi_s4 + sigma_rel[5] * phi_s5);

        const double phi_tr = ci_sqr / (2.0 * RTtr) - 2.5;
        double phi_rot = 0.0;
        if (rot_active) phi_rot = erot / (boltz * Trot_rel) - 0.5 * species.rotdof;
        double phi_vib = 0.0;
        if (vib_active) phi_vib = evib / (boltz * Tvib_rel) - 0.5 * dof_vib(species, Tvib_rel);

        for (int id = 0; id < 3; id++) {
            w += 2.0 * ci[id] * qtr_rel[id] * phi_tr / (5.0 * prel * RTtr);
            if (rot_active) {
                w += 2.0 * ci[id] * qrot_rel[id] * phi_rot /
                     (static_cast<double>(species.rotdof) * prel * RTrot);
            }
            if (vib_active && Cvib_rel > 0.0) {
                w += ci[id] * qvib_rel[id] * phi_vib / (Cvib_rel * prel * RTvib);
            }
        }

        if (!std::isfinite(w) || w < 0.0) {
            ++count_fail_relaxation;
            return;
        }
        if (!(cinfo[ic].macro.Wmax > 0.0) || !std::isfinite(cinfo[ic].macro.Wmax)) {
            cinfo[ic].macro.Wmax = 1.0;
            resetWmax_flag[ic] = 0;
        }
        if (w > cinfo[ic].macro.Wmax && w < 5.0) {
            cinfo[ic].macro.Wmax = w;
            resetWmax_flag[ic] = 0;
            accepted = true;
            break;
        }
        const double accept_prob = w / cinfo[ic].macro.Wmax;
        if (accept_prob >= 1.0 || (accept_prob > 0.0 && random->uniform() < accept_prob)) {
            accepted = true;
            break;
        }
        if (count_loop > 100) {
            accepted = true;
            break;
        }
    }

    if (!accepted) {
        ++count_fail_relaxation;
        return;
    }

    // Step 5: write the accepted sample back to the particle.
    for (int id = 0; id < 3; id++) p->v[id] = ci[id] + interMacro->v[id];
    p->erot = erot;
    p->evib = evib;
    ++count_done_relaxation;
}


/* ----------------------------------------------------------------------
   estimate a good value for vremax for a group pair in any grid cell
   called by Collide parent in init()

   NOTE: MSMC uses its own per-cell crmax cache in collisions(), so this
   hook is only a placeholder for parent-class initialization.
------------------------------------------------------------------------- */
double CollideMSMC::vremax_init(int igroup, int jgroup)
{
    return 300.0;
}

/* ----------------------------------------------------------------------
* placeholder for Collide parent interface; not used by CollideMSMC
------------------------------------------------------------------------- */
double CollideMSMC::attempt_collision(int icell, int, double tao)
{
    error->all(FLERR, "call attempt_collision function of CollideMSMC");
    return 0.0;
}

/* ----------------------------------------------------------------------
  NOTE: perform_collision is replaced by the MSMC-specific collision routines,
        should never be called
------------------------------------------------------------------------- */
int CollideMSMC::perform_collision(Particle::OnePart*&, Particle::OnePart*&, Particle::OnePart*&)
{
    error->all(FLERR, "call perform_collision function of CollideMSMC");
    return 0;
}

/* ----------------------------------------------------------------------
  compute macro quantities for all cells
  including velocity, temperatures, stress and heat flux
  NOTE: some quantities are only needed when a cell is eligible for MSMC
------------------------------------------------------------------------- */
void CollideMSMC::computeMacro()
{
    if (!(nmaxconserv >= 0)) error->one(FLERR,
        "CollideMSMC::computeMacro(): !(nmaxconserv >= 0)");
    if (nglocal > nmaxconserv) {
        while (nglocal > nmaxconserv) nmaxconserv += DELTAPART;
        memory->destroy(conservMacro);
        memory->create(conservMacro, nmaxconserv, "collideMSMC:conservmacro");
    }
    // Clear moment accumulators.
    for (int icell = 0; icell < nglocal; icell++)
    {
        grid->cells[icell].macro.v[0] = 0.0;
        grid->cells[icell].macro.v[1] = 0.0;
        grid->cells[icell].macro.v[2] = 0.0;
        grid->cells[icell].macro.Temp = 0.0;
        NoCommMacro& nmacro = grid->cinfo[icell].macro;
        ConservMacro& cm = conservMacro[icell];
        nmacro.sum_vi[0] = nmacro.sum_vi[1] = nmacro.sum_vi[2] = 0.0;
        nmacro.sum_vij[0] = nmacro.sum_vij[1] = nmacro.sum_vij[2] = 0.0;
        nmacro.sum_vij[3] = nmacro.sum_vij[4] = nmacro.sum_vij[5] = 0.0;
        nmacro.sum_C2vi[0] = nmacro.sum_C2vi[1] = nmacro.sum_C2vi[2] = 0.0;
        // --- MSMC polyatomic ---
        nmacro.sum_erot = 0.0;
        nmacro.sum_evib = 0.0;
        nmacro.sum_erot_vi[0] = nmacro.sum_erot_vi[1] = nmacro.sum_erot_vi[2] = 0.0;
        nmacro.sum_evib_vi[0] = nmacro.sum_evib_vi[1] = nmacro.sum_evib_vi[2] = 0.0;
        nmacro.rho = 0.0;
        // --- MSMC polyatomic ---
        cm.done_relaxation = 0;
        cm.u_pre[0] = cm.u_pre[1] = cm.u_pre[2] = 0.0;
        cm.Etr_pre = 0.0;
        cm.Erot_pre = 0.0;
        cm.Evib_pre = 0.0;
    }
    // Accumulate particle moments.
    for (int ipart = 0; ipart < particle->nlocal; ++ipart) {
        Particle::OnePart& part = particle->particles[ipart];
        NoCommMacro& nmacro = grid->cinfo[part.icell].macro;
        double* v = part.v;
        double c_sqr = 0.0;
        for (int id = 0; id < 3; ++id) {
            // 1st-order moment
            nmacro.sum_vi[id] += v[id];
            // 2nd-order self moment
            double vii = v[id] * v[id];
            nmacro.sum_vij[id] += vii;
            c_sqr += vii;
        }
        // 2nd-order cross moment
        nmacro.sum_vij[3] += v[0] * v[1];
        nmacro.sum_vij[4] += v[0] * v[2];
        nmacro.sum_vij[5] += v[1] * v[2];
        // 3rd-order moment
        nmacro.sum_C2vi[0] += c_sqr * v[0];
        nmacro.sum_C2vi[1] += c_sqr * v[1];
        nmacro.sum_C2vi[2] += c_sqr * v[2];
        // --- MSMC polyatomic ---
        nmacro.sum_erot += part.erot;
        nmacro.sum_evib += part.evib;
        for (int id = 0; id < 3; ++id) {
            nmacro.sum_erot_vi[id] += part.erot * v[id];
            nmacro.sum_evib_vi[id] += part.evib * v[id];
        }
        // --- MSMC polyatomic ---
    }
    // Compute macroscopic quantities.
    for (int icell = 0; icell < nglocal; icell++)
    {
        NoCommMacro& nmacro_EMA = grid->cinfo[icell].macro;
        CommMacro& cmacro = grid->cells[icell].macro;
        Grid::ChildCell& cell = grid->cells[icell];
        Grid::ChildInfo& cinfo = grid->cinfo[icell];
        Particle::OnePart* particles = particle->particles;
        int Np = cinfo.count;
        if (Np <= 0) {
            nmacro_EMA.do_relaxation = 0;
            conservMacro[icell].done_relaxation = 0;
            continue;
        }
        if (cinfo.first < 0) {
            error->one(FLERR, "CollideMSMC::computeMacro(): cell has particles but invalid first index");
        }
        // Currently assume all particles have same ispecies
        Particle::Species& species = particle->species[particles[cinfo.first].ispecies];
        double mass = species.mass;
        Params& ps = params[particles[cinfo.first].ispecies];
        // (1) Density
        double nrho = cinfo.count * update->fnum * cinfo.weight / cell.dt_weight / cinfo.volume;
        nmacro_EMA.rho = nrho * mass;
        // (2) Velocity
        double* ui = cmacro.v;
        for (int i = 0; i < 3; ++i) {
            ui[i] = nmacro_EMA.sum_vi[i] / Np;
        }
        double sum_c_sqr_fallback = nmacro_EMA.sum_vij[0] + nmacro_EMA.sum_vij[1] + nmacro_EMA.sum_vij[2];
        double u_sqr_fallback = ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2];
        double Ttr_fallback = mass / update->boltz * (sum_c_sqr_fallback / Np - u_sqr_fallback) / 3.0;
        if (Ttr_fallback < 0.0) Ttr_fallback = 0.0;
        double Erot_fallback = nmacro_EMA.sum_erot / Np;
        double Evib_fallback = nmacro_EMA.sum_evib / Np;
        nmacro_EMA.Ttr = Ttr_fallback;
        nmacro_EMA.Trot = compute_Trot(species, Erot_fallback);
        nmacro_EMA.Tvib = compute_Tvib(species, Evib_fallback);
        // Use DSMC fallback when there are too few particles for MSMC relaxation.
        if (Np < 20) {
            nmacro_EMA.do_relaxation = 0;
            conservMacro[icell].done_relaxation = 0;
            ++count_lowNp_cell;
            continue;
        }
        else
        {
            nmacro_EMA.do_relaxation = 1;
            conservMacro[icell].done_relaxation = 1;
        }
        // (3) Temperature
        double* sum_ci_cj = nmacro_EMA.sum_vij;
        double u_sqr = ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2];
        double sum_c_sqr = sum_ci_cj[0] + sum_ci_cj[1] + sum_ci_cj[2];
        // Conservation quantities - velocity & total energy
        ConservMacro& cm = conservMacro[icell];
        for (int id = 0; id < 3; ++id) cm.u_pre[id] = ui[id];
        cm.Etr_pre = 0.5 * mass * (sum_c_sqr - Np * u_sqr) / Np;
        cm.Erot_pre = nmacro_EMA.sum_erot / Np;
        cm.Evib_pre = nmacro_EMA.sum_evib / Np;
        // NOTE: temperature is Unbiased estimate
        double Ttr_inst = ((double)Np / (Np - 1)) * mass / update->boltz * (sum_c_sqr / Np - u_sqr) / 3.0;
        double Trot_inst = compute_Trot(species, nmacro_EMA.sum_erot / Np);
        double Tvib_inst = compute_Tvib(species, nmacro_EMA.sum_evib / Np);
        // time-averaging of Ttr, Trot, Tvib
        nmacro_EMA.Ttr = nmacro_EMA.Ttr * time_ave_coef + Ttr_inst * (1.0 - time_ave_coef);
        nmacro_EMA.Trot = nmacro_EMA.Trot * time_ave_coef + Trot_inst * (1.0 - time_ave_coef);
        nmacro_EMA.Tvib = nmacro_EMA.Tvib * time_ave_coef + Tvib_inst * (1.0 - time_ave_coef);
        // CommMacro.Temp stores the instantaneous equilibrium temperature Teq
        cmacro.Temp = solve_Teq(species, Ttr_inst * static_cast<double>(Np - 1) / Np, Trot_inst, Tvib_inst);
        if (!(nmacro_EMA.Ttr > ps.T_ref * 0.01)) {
            // If particles are weighted, identical velocities can appear and
            // yield an apparent zero temperature from floating-point truncation.
            ++count_lowtemp_cell;
            nmacro_EMA.do_relaxation = 0;
            conservMacro[icell].done_relaxation = 0;
            continue;
        }
        ++count_msmc_cell;
        double ptr_inst = nrho * update->boltz * Ttr_inst;
        double peq_inst = nrho * update->boltz * cmacro.Temp;
        nmacro_EMA.ptr = nmacro_EMA.ptr * time_ave_coef + ptr_inst * (1.0 - time_ave_coef);
        nmacro_EMA.peq = nmacro_EMA.peq * time_ave_coef + peq_inst * (1.0 - time_ave_coef);
        // (4) Stress
        double pij[6]{};
        double Sfactor = ((double)Np / (Np - 1)) * mass * update->fnum * cinfo.weight / cell.dt_weight / cinfo.volume;
        for (int id = 0; id < 3; ++id) {
            pij[id] = Sfactor * (sum_ci_cj[id] - Np * ui[id] * ui[id]);
        }
        pij[3] = Sfactor * (sum_ci_cj[3] - Np * ui[0] * ui[1]);
        pij[4] = Sfactor * (sum_ci_cj[4] - Np * ui[0] * ui[2]);
        pij[5] = Sfactor * (sum_ci_cj[5] - Np * ui[1] * ui[2]);
        // time-averaging of stress
        double ptrace = (pij[0] + pij[1] + pij[2]) / 3.0;
        for (int id = 0; id < 3; ++id) {
            nmacro_EMA.sigma_ij[id] = nmacro_EMA.sigma_ij[id] * time_ave_coef
                                     + (pij[id] - ptrace) * (1.0 - time_ave_coef);
        }
        for (int id = 3; id < 6; ++id) {
            nmacro_EMA.sigma_ij[id] = nmacro_EMA.sigma_ij[id] * time_ave_coef
                                     + pij[id] * (1.0 - time_ave_coef);
        }
        // (5) Heat Flux
        double qi[3]{};
        double Qfactor = 0.5 * ((double)Np / (Np - 2)) * Sfactor;
        qi[0] = Qfactor * (nmacro_EMA.sum_C2vi[0]
            - ui[0] * sum_c_sqr + 2 * Np * u_sqr * ui[0]
            - 2 * (ui[0] * sum_ci_cj[0] + ui[1] * sum_ci_cj[3] + ui[2] * sum_ci_cj[4]));
        qi[1] = Qfactor * (nmacro_EMA.sum_C2vi[1]
            - ui[1] * sum_c_sqr + 2 * Np * u_sqr * ui[1]
            - 2 * (ui[0] * sum_ci_cj[3] + ui[1] * sum_ci_cj[1] + ui[2] * sum_ci_cj[5]));
        qi[2] = Qfactor * (nmacro_EMA.sum_C2vi[2]
            - ui[2] * sum_c_sqr + 2 * Np * u_sqr * ui[2]
            - 2 * (ui[0] * sum_ci_cj[4] + ui[1] * sum_ci_cj[5] + ui[2] * sum_ci_cj[2]));
        // time-averaging of heat flux
        for (int id = 0; id < 3; ++id) {
            nmacro_EMA.qi[id] = nmacro_EMA.qi[id] * time_ave_coef
                               + qi[id] * (1.0 - time_ave_coef);
        }
        // --- MSMC polyatomic ---
        for (int id = 0; id < 3; ++id) {
            double qrot = nrho * (nmacro_EMA.sum_erot_vi[id] - nmacro_EMA.sum_erot * ui[id]) / Np;
            double qvib = nrho * (nmacro_EMA.sum_evib_vi[id] - nmacro_EMA.sum_evib * ui[id]) / Np;
            nmacro_EMA.qrot[id] = nmacro_EMA.qrot[id] * time_ave_coef
                                 + qrot * (1.0 - time_ave_coef);
            nmacro_EMA.qvib[id] = nmacro_EMA.qvib[id] * time_ave_coef
                                 + qvib * (1.0 - time_ave_coef);
        }
        // --- MSMC polyatomic ---
    }
    // run commMacro
    if (grid->gridCommMacro) grid->gridCommMacro->runComm();
}

// --- MSMC polyatomic ---
double CollideMSMC::compute_Trot(const Particle::Species& species, double erot_avg)
{
    if (species.rotdof <= 0 || erot_avg <= 0.0) return 0.0;
    return 2.0 * erot_avg / (species.rotdof * update->boltz);
}

double CollideMSMC::compute_Tvib(const Particle::Species& species, double evib_avg)
{
    if (vibstyle == NONE || species.vibdof <= 0 || evib_avg <= 0.0) return 0.0;
    // discrete vibrational mode
    if (vibstyle == DISCRETE) {
        const double theta0_vib = species.vibtemp[0];
        if (theta0_vib <= 0.0) return 0.0;
        const double ibar = evib_avg / (update->boltz * theta0_vib);
        if (ibar <= 0.0) return 0.0;
        return theta0_vib / log(1.0 + 1.0 / ibar);
    }
    // smooth vibrational mode
    return 2.0 * evib_avg / (species.vibdof * update->boltz);
}

double CollideMSMC::dof_vib(const Particle::Species& species, double T)
{
    if (vibstyle == NONE || species.vibdof <= 0) return 0.0;
    // smooth vibrational mode
    if (vibstyle != DISCRETE) return static_cast<double>(species.vibdof);
    if (T < 1.0e-3) return 0.0;
    // discrete vibrational mode
    const double theta0_vib = species.vibtemp[0];
    if (theta0_vib <= 0.0) return 0.0;
    const double x = theta0_vib / T;
    if (x > 50.0) return 0.0;
    return (2.0 * x) / (exp(x) - 1.0);
}

double CollideMSMC::solve_Teq(const Particle::Species& species,
                              double Ttr, double Trot, double Tvib)
{
    const double dof_rot = static_cast<double>(species.rotdof);
    const double dof_vib_now = dof_vib(species, Tvib);
    const double total = 3.0 * Ttr + dof_rot * Trot + dof_vib_now * Tvib;

    if (species.vibdof <= 0 || vibstyle == NONE) {
        return total / (3.0 + dof_rot);
    }
    // smooth vibrational mode
    if (vibstyle != DISCRETE) {
        return total / (3.0 + dof_rot + static_cast<double>(species.vibdof));
    }
    // discrete vibrational mode
    double Teq = (3.0 * Ttr + dof_rot * Trot) / (3.0 + dof_rot);
    for (int iter = 0; iter < 5; ++iter) {
        const double dof_eq = dof_vib(species, Teq);
        const double Teq_new = total / (3.0 + dof_rot + dof_eq);
        if (fabs(Teq_new - Teq) < 1.0e-3) return Teq_new;
        Teq = Teq_new;
    }
    return Teq;
}

CollideMSMC::ExpRKProb CollideMSMC::build_exprk_prob(int icell, double vc)
{
    ExpRKProb epr;
    epr.vc = vc;
    epr.dt = update->dt / grid->cells[icell].dt_weight;
    epr.vc_dt = epr.vc * epr.dt;
    epr.Pcoll1 = epr.vc_dt * RKw1 * exp(-(1.0 - RKc1) * epr.vc_dt);
    epr.Pcoll2 = epr.vc_dt * RKw2 * exp(-(1.0 - RKc2) * epr.vc_dt);
    epr.Prelax = 1.0 - exp(-epr.vc_dt) - epr.Pcoll1 - epr.Pcoll2;
    epr.P2c1 = epr.vc_dt * RKa21 * exp(-(RKc2 - RKc1) * epr.vc_dt);
    epr.P2r = 1.0 - exp(-RKc2 * epr.vc_dt) - epr.P2c1;
    grid->cinfo[icell].macro.tao = epr.vc_dt;
    return epr;
}

double CollideMSMC::RotRelNum(const Particle::Species& species,
                              const Params& ps,
                              double Ttr)
{
    if (species.rotdof <= 0) return 1.0e20;
    if (ps.RotRelNum_model == Z_CONSTANT) return ps.Zrot_cont;

    if (Ttr <= 1.0e-12) return 1.0e20;

    const double Zrot_inf = 18.1;
    const double Tstar = 91.5;
    const double ratio = Tstar / Ttr;
    const double denom = 1.0 + 0.5 * pow(MY_PI, 1.5) * sqrt(ratio) +
                         (0.25 * MY_PI * MY_PI + MY_PI) * ratio;
    if (denom <= 0.0) return 1.0e20;
    return Zrot_inf / denom;
}

double CollideMSMC::VibRelNum(const Particle::Species& species,
                              const Params& ps,
                              double Ttr,
                              double ptr,
                              double tau_coll)
{
    if (species.vibdof <= 0) return 1.0e20;
    if (ps.VibRelNum_model == Z_CONSTANT) return ps.Zvib_cont;

    if (Ttr <= 1.0e-12 || ptr <= 0.0 || tau_coll <= 0.0) return 1.0e20;

    // Use the original Millikan-White N2 parameterization adopted by the
    // MSPD circle2D case and the recent FPM cylinder paper.
    const double A = 219.138;
    const double B = 0.029;
    const double sigma_s = 5.81e-21;
    const double p_atm = ptr / 101325.0;
    const double nrho = ptr / (update->boltz * Ttr);
    if (p_atm <= 0.0 || nrho <= 0.0) return 1.0e20;

    // Millikan-White uses pressure in atm; Park's correction is evaluated in SI.
    const double tau_MW = exp(A * pow(Ttr, -1.0 / 3.0) - A * B - 18.42) / p_atm;
    const double tau_p = sqrt(MY_PI * species.mass / (8.0 * update->boltz * Ttr)) /
                         (sigma_s * nrho);
    const double Zvib = (tau_MW + tau_p) / tau_coll;
    return (Zvib > 0.0) ? Zvib : 1.0e20;
}

CollideMSMC::Thermo
CollideMSMC::updateThermo(const Particle::Species& species, const Params& ps,
                          const CommMacro& cmacro, const NoCommMacro& nmacro)
{
    Thermo th;
    // Eigenvalue for viscosity
    th.tau_vis = 0.0;
    th.Ls = 0.0;
    const double vis = ps.mu_ref * pow(nmacro.Ttr / ps.T_ref, ps.omega);
    if (nmacro.ptr > 0.0 && vis > 0.0) {
        th.tau_vis = vis / nmacro.ptr;
        th.Ls = 1.0 / th.tau_vis;
    }

    // Heat capacity
    th.Ctr = 1.5;
    th.Crot = 0.5 * species.rotdof;
    th.Ctr_rot = th.Ctr + th.Crot;

    th.Cvib = 0.0;
    if (species.vibdof > 0 && vibstyle == DISCRETE) {       // discrete vibrational mode
        const double theta0_vib = species.vibtemp[0];
        if (theta0_vib > 0.0 && cmacro.Temp > 1.0e-12) {
            const double dof_eq = dof_vib(species, cmacro.Temp);
            th.Cvib = exp(theta0_vib / cmacro.Temp) * dof_eq * dof_eq / 4.0;
        }
    } else if (species.vibdof > 0 && vibstyle != NONE) {    // smooth vibrational mode
        th.Cvib = 0.5 * species.vibdof;
    }
    th.Ctr_vib = th.Ctr + th.Cvib;
    th.Ctot = th.Ctr_rot + th.Cvib;

    // projection matrices for rotational & vibrational energy
    th.Prot = (1.0 / th.Ctr_rot) *
      Mat3x3(th.Crot, -th.Crot, 0.0,
             -th.Ctr, th.Ctr, 0.0,
             0.0, 0.0, 0.0);

    // th.Pvib = (1.0 / th.Ctot) *
    //   Mat3x3(th.Cvib * th.Ctr / th.Ctr_rot, th.Cvib * th.Crot / th.Ctr_rot, -th.Cvib,
    //          th.Cvib * th.Ctr / th.Ctr_rot, th.Cvib * th.Crot / th.Ctr_rot, -th.Cvib,
    //          -th.Ctr, -th.Crot, th.Ctr_rot);

    th.Pvib = (1.0 / th.Ctr_vib) *
      Mat3x3(th.Cvib, 0.0, -th.Cvib,
             0.0, 0.0, 0.0,
             -th.Ctr, 0.0, th.Ctr);

    if (th.tau_vis <= 0.0) return th;

    // Eigenvalues matrix for energy
    const double tau_coll = 4.0 / MY_PI * th.tau_vis;
    double Zrot = RotRelNum(species, ps, nmacro.Ttr);
    double Zvib = VibRelNum(species, ps, nmacro.Ttr, nmacro.ptr, tau_coll);
    double tau_rot = Zrot * th.Ctr / (th.Ctr + th.Crot) * tau_coll;
    double tau_vib = Zvib * th.Ctr / (th.Ctr + th.Cvib) * tau_coll;
    // double tau_vib = Zvib * th.Ctr_rot / th.Ctot * tau_coll;

    if (Zrot < 1.0e19) th.Le = th.Le + (1.0 / tau_rot) * th.Prot;
    if (Zvib < 1.0e19) th.Le = th.Le + (1.0 / tau_vib) * th.Pvib;

    // Eigenvalues matrix for heat flux
    const double alpha = 1.0;
    const double Sc = 5.0 * (2.0 + alpha) / (3.0 * (7.0 - 2.0 * ps.omega) * alpha);
    const double invZrot = (Zrot < 1.0e19) ? 1.0 / Zrot : 0.0;
    const double invZvib = (Zvib < 1.0e19) ? 1.0 / Zvib : 0.0;
    th.Lq.m[0][0] = 2.0 / 3.0 + 5.0 / (6.0 * th.Ctr) * (th.Crot * invZrot + th.Cvib * invZvib);
    th.Lq.m[0][1] = -th.Crot * invZrot / 3.0;
    th.Lq.m[0][2] = -th.Cvib * invZvib / 3.0;
    th.Lq.m[1][0] = -2.5 * invZrot / 3.0;
    th.Lq.m[1][1] = Sc + 0.5 * invZrot;
    th.Lq.m[2][0] = -2.5 * invZvib / 3.0;
    th.Lq.m[2][2] = Sc + 0.5 * invZvib;
    th.Lq = (1.0 / th.tau_vis) * th.Lq;

    return th;
}

void CollideMSMC::compute_stress_compensation(const NoCommMacro& nmacro,
                                              const ExpRKProb& epr,
                                              const Thermo& th,
                                              FadCoeff& fad)
{
    if (epr.vc <= 0.0 || th.tau_vis <= 0.0 || nmacro.ptr <= 0.0) return;

    const double I_Ls_vc = 1.0 - th.Ls / epr.vc;
    const double Cs = exp(-epr.vc_dt) *
      (1.0 + epr.vc_dt * I_Ls_vc + 0.5 * epr.vc_dt * epr.vc_dt * I_Ls_vc * I_Ls_vc);
    const double Bs = 1.0 - Cs -
      (RKw1 * exp(-(1.0 - RKc1) * epr.vc_dt) +
       RKw2 * exp(-(1.0 - RKc2) * epr.vc_dt) +
       RKw2 * RKa21 * epr.vc_dt * exp(-(1.0 - RKc1) * epr.vc_dt) * I_Ls_vc) *
      th.Ls * epr.dt;
    const double Fs = 1.0 / (1.0 + 0.5 * epr.dt / th.tau_vis);

    for (int id = 0; id < 6; id++) {
      const double sPos = nmacro.sigma_ij[id] / nmacro.ptr; // tr or eq?
      const double sMid = Fs * sPos;
      const double sNeg = (1.0 - 0.5 * epr.dt / th.tau_vis) * sMid;
      fad.yh_s[id] = solve_h(Cs, sPos, Bs, 1.0, sNeg);
    }
}

void CollideMSMC::compute_energy_compensation(const Particle::Species& species,
                                              const CommMacro& cmacro,
                                              const NoCommMacro& nmacro,
                                              const ExpRKProb& epr,
                                              const Thermo& th,
                                              FadCoeff& fad)
{
    if (epr.vc <= 0.0 || th.tau_vis <= 0.0 || cmacro.Temp <= 1.0e-12) return;

    const Mat3x3 I_Le_vc = Mat3x3::Identity() - (1.0 / epr.vc) * th.Le;
    const Mat3x3 Ce = exp(-epr.vc_dt) *
      (Mat3x3::Identity() + epr.vc_dt * I_Le_vc +
       0.5 * epr.vc_dt * epr.vc_dt * I_Le_vc * I_Le_vc);
    const Mat3x3 Be = Mat3x3::Identity() - Ce -
      (Mat3x3::Identity() *
       (RKw1 * exp(-(1.0 - RKc1) * epr.vc_dt) + RKw2 * exp(-(1.0 - RKc2) * epr.vc_dt)) +
       I_Le_vc * (RKw2 * RKa21 * epr.vc_dt * exp(-(1.0 - RKc1) * epr.vc_dt))) *
      th.Le * epr.dt;

    const double etr_ratio = (cmacro.Temp > 0.0) ? (nmacro.Ttr - cmacro.Temp) / cmacro.Temp : 0.0;
    const double erot_ratio = (cmacro.Temp > 0.0) ? (nmacro.Trot - cmacro.Temp) / cmacro.Temp : 0.0;
    double evib_ratio = 0.0;
    if (cmacro.Temp > 0.0 && th.Cvib > 0.0 && species.vibdof > 0) {
      double evib = 0.5 * dof_vib(species, nmacro.Tvib) * update->boltz * nmacro.Tvib;
      double evib_eq = 0.5 * dof_vib(species, cmacro.Temp) * update->boltz * cmacro.Temp;
      evib_ratio = (evib - evib_eq) / (th.Cvib * update->boltz * cmacro.Temp);
    }

    const Mat3x3 Fe = (Mat3x3::Identity() + 0.5 * epr.dt * th.Le).inverse();
    const Vec3 ePos(etr_ratio, erot_ratio, evib_ratio);
    const Vec3 eMid = Fe * ePos;
    const Vec3 eNeg = (Mat3x3::Identity() - 0.5 * epr.dt * th.Le) * eMid;
    const Vec3 he = solve_h(Ce, ePos, Be, Mat3x3::Identity(), eNeg);
    fad.yh_etr = he.v[0];
    fad.yh_erot = he.v[1];
    fad.yh_evib = he.v[2];
}

void CollideMSMC::compute_heatflux_compensation(const Particle::Species& species,
                                                const CommMacro& cmacro,
                                                const NoCommMacro& nmacro,
                                                const ExpRKProb& epr,
                                                const Thermo& th,
                                                FadCoeff& fad)
{
    if (epr.vc <= 0.0 || th.tau_vis <= 0.0 || nmacro.peq <= 0.0 || cmacro.Temp <= 0.0) return;

    const Mat3x3 I_Lq_vc = Mat3x3::Identity() - (1.0 / epr.vc) * th.Lq;
    const Mat3x3 Cq = exp(-epr.vc_dt) *
      (Mat3x3::Identity() + epr.vc_dt * I_Lq_vc +
       0.5 * epr.vc_dt * epr.vc_dt * I_Lq_vc * I_Lq_vc);
    const Mat3x3 Bq = Mat3x3::Identity() - Cq -
      (Mat3x3::Identity() *
       (RKw1 * exp(-(1.0 - RKc1) * epr.vc_dt) + RKw2 * exp(-(1.0 - RKc2) * epr.vc_dt)) +
       I_Lq_vc * (RKw2 * RKa21 * epr.vc_dt * exp(-(1.0 - RKc1) * epr.vc_dt))) *
      th.Lq * epr.dt;

    const double factor = 1.0 / (nmacro.peq * sqrt(update->boltz * cmacro.Temp / species.mass));
    const Mat3x3 Fq = (Mat3x3::Identity() + 0.5 * epr.dt * th.Lq).inverse();

    for (int id = 0; id < 3; id++) {
      const double qtr = 0.4 * factor * nmacro.qi[id];
      const double qrot = (th.Crot > 0.0) ? factor * nmacro.qrot[id] / th.Crot : 0.0;
      const double qvib = (th.Cvib > 0.0) ? factor * nmacro.qvib[id] / th.Cvib : 0.0;
      const Vec3 qPos(qtr, qrot, qvib);
      const Vec3 qMid = Fq * qPos;
      const Vec3 qNeg = (Mat3x3::Identity() - 0.5 * epr.dt * th.Lq) * qMid;
      const Vec3 hq = solve_h(Cq, qPos, Bq, Mat3x3::Identity(), qNeg);
      fad.yh_qtr[id] = hq.v[0];
      fad.yh_qrot[id] = hq.v[1];
      fad.yh_qvib[id] = hq.v[2];
    }
}
// --- MSMC polyatomic ---

/* ----------------------------------------------------------------------
   read list of species defined in species file
   store info in filespecies and nfilespecies
   only invoked by proc 0
------------------------------------------------------------------------- */
void CollideMSMC::read_param_file(char* fname)
{
    FILE* file = fopen(fname, "r");
    if (file == NULL) {
        char str[128];
        sprintf(str, "Cannot open MSMC parameter file %s", fname);
        error->one(FLERR, str);
    }
    // set all species diameters to -1, so can detect if not read
    // set all cross-species parameters to -1 to catch no-reads, as
    // well as user-selected average
    for (int i = 0; i < nparams; i++) params[i].mu_ref = -1.0;
    // read file line by line
    // skip blank lines or comment lines starting with '#'
    // all other lines must have at least REQWORDS, which depends on VARIABLE flag
    // [0]Species, [1]Diameter, [2]Viscosity, [3]Omega, [4]Tref
    // [5]Pr, [6]Zrot, [7]Zvib
    // Zrot and Zvib are parsed independently:
    // numeric values select the constant model, while text keywords enable
    // the temperature-dependent N2 relaxation model for that mode.
    int REQWORDS = 8;
    char** words = new char* [REQWORDS];
    char line[MAXLINE];
    int isp;

    while (fgets(line, MAXLINE, file)) {
        int pre = strspn(line, " \t\n\r");
        if (pre == strlen(line) || line[pre] == '#') continue;
        // Parse tokens and check field count.
        int nwords = wordparse(REQWORDS, line, words);
        if (nwords < REQWORDS) {
            error->one(FLERR, "Incorrect line format in MSMC parameter file");
        }
        // Find species ID.
        isp = particle->find_species(words[0]);
        if (isp < 0) {
            continue;
        }
        else {
            // Read species parameters.
            params[isp].d_ref = atof(words[1]);
            params[isp].mu_ref = atof(words[2]);
            params[isp].omega = atof(words[3]);
            params[isp].T_ref = atof(words[4]);
            params[isp].Pr = atof(words[5]);
            char* endrot = NULL;
            char* endvib = NULL;
            double Zrot_value = strtod(words[6], &endrot);
            double Zvib_value = strtod(words[7], &endvib);
            int Zrot_numeric = (endrot != words[6] && *endrot == '\0');
            int Zvib_numeric = (endvib != words[7] && *endvib == '\0');
            if (Zrot_numeric) {
                params[isp].RotRelNum_model = Z_CONSTANT;
                params[isp].Zrot_cont = Zrot_value;
            } else {
                params[isp].RotRelNum_model = Z_TEMPERATURE_DEPENDENT;
                params[isp].Zrot_cont = 1.0e20;
            }

            if (Zvib_numeric) {
                params[isp].VibRelNum_model = Z_CONSTANT;
                params[isp].Zvib_cont = Zvib_value;
            } else {
                params[isp].VibRelNum_model = Z_TEMPERATURE_DEPENDENT;
                params[isp].Zvib_cont = 1.0e20;
            }
        }
    }
    delete[] words;
    fclose(file);
    // check that params were read for all species
    for (int i = 0; i < nparams; i++) {
        if (params[i].mu_ref < 0.0) {
            char str[128];
            sprintf(str, "Species %s did not appear in MSMC parameter file",
                particle->species[i].id);
            error->one(FLERR, str);
        }
        if (params[i].Pr <= 0.0) {
            char str[128];
            sprintf(str, "Species %s has invalid Pr in MSMC parameter file",
                particle->species[i].id);
            error->one(FLERR, str);
        }
        if (params[i].RotRelNum_model == Z_CONSTANT &&
            particle->species[i].rotdof > 0 && params[i].Zrot_cont <= 0.0) {
            char str[128];
            sprintf(str, "Species %s has invalid Zrot_cont in MSMC parameter file",
                particle->species[i].id);
            error->one(FLERR, str);
        }
        if (params[i].VibRelNum_model == Z_CONSTANT &&
            particle->species[i].vibdof > 0 && params[i].Zvib_cont <= 0.0) {
            char str[128];
            sprintf(str, "Species %s has invalid Zvib_cont in MSMC parameter file",
                particle->species[i].id);
            error->one(FLERR, str);
        }
    }
}

/* ----------------------------------------------------------------------
   parse up to n=maxwords whitespace-delimited words in line
   store ptr to each word in words and count number of words
   same as CollideVSS::wordparse
------------------------------------------------------------------------- */
int CollideMSMC::wordparse(int maxwords, char* line, char** words)
{
    int nwords = 1;
    char* word;

    words[0] = strtok(line, " \t\n");
    while ((word = strtok(NULL, " \t\n")) != NULL && nwords < maxwords) {
        words[nwords++] = word;
    }
    return nwords;
}

/* ---------------------------------------------------------------------- */

void CollideMSMC::print_warning() {
    if (output->next_stats != update->ntimestep)return;
    bigint sum1, sum2, sum3, sum4, sum5, sum6;
    sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = 0;
    MPI_Allreduce(&count_fail_relaxation, &sum1, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    MPI_Allreduce(&count_done_relaxation, &sum2, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    MPI_Allreduce(&count_lowtemp_cell, &sum3, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    MPI_Allreduce(&count_msmc_cell, &sum4, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    MPI_Allreduce(&count_lowNp_cell, &sum5, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    MPI_Allreduce(&count_conserv_clip_cell, &sum6, 1, MPI_SPARTA_BIGINT, MPI_SUM, world);
    if (comm->me == 0) {
        if (sum1) {
            char str[128];
            sprintf(str, "%ld relaxation failed in total %ld relaxation, percentage = %.4f",
                sum1, sum2, 100.0 * sum1 / sum2);
            error->warning(FLERR, str);
        }
        if (sum6) {
            char str[160];
            double pct = (sum4 > 0) ? 100.0 * sum6 / sum4 : 0.0;
            sprintf(str, "%ld MSMC cells required non-physical conservation clipping/scaling, percentage = %.4f",
                sum6, pct);
            error->warning(FLERR, str);
        }
    }
    reset_count();
}

/* ---------------------------------------------------------------------- */

CollideMSMCModify::CollideMSMCModify(SPARTA* sparta) : Pointers(sparta) {}

/* ---------------------------------------------------------------------- */

CollideMSMCModify::~CollideMSMCModify() {}

/* ----------------------------------------------------------------------
* process collide_msmc_modify command, included in style_command.h
------------------------------------------------------------------------- */

void CollideMSMCModify::command(int narg, char** arg)
{
    if (strcmp(collide->style, "msmc") != 0) {
        error->all(FLERR,
            "Using collide_msmc_modify command when collide.style != msmc");
    }
    if (narg == 0) error->all(FLERR, "Illegal collide_modify command");
    CollideMSMC* collideMSMC = dynamic_cast<CollideMSMC*>(collide);
    if (!collideMSMC) {
        error->all(FLERR, "CollideMSMCModify: dynamic_cast fault");
    }
    int iarg = 0;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "reset_wmax") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal collide_msmc_modify command");
            double reset = atof(arg[iarg + 1]);
            if (reset <= 0) {
                collideMSMC->resetWmax = 0.0;
            }
            else if (reset >= 1.0) {
                error->all(FLERR,
                    "Illegal collide_msmc_modify command: resetWmax > 1");
            }
            else {
                collideMSMC->resetWmax = reset;
            }
            iarg += 2;
        }
        else if (strcmp(arg[iarg], "pr_num") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal collide_msmc_modify command");
            collideMSMC->Pr = atof(arg[iarg + 1]);
            if (collideMSMC->Pr <= 0)
                error->all(FLERR, "Illegal collide_msmc_modify Prantl number");
            iarg += 2;
        }
        else if (strcmp(arg[iarg], "time_ave") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal collide_msmc_modify command");
            collideMSMC->time_ave_coef = atof(arg[iarg + 1]);
            if (collideMSMC->time_ave_coef < 0 || collideMSMC->time_ave_coef >= 1)
                error->all(FLERR, "Illegal collide_msmc_modify time_ave_coef");
            iarg += 2;
        }
        else if (strcmp(arg[iarg], "interpolate") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal collide_msmc_modify command");
            if (strcmp(arg[iarg + 1], "yes") == 0) collideMSMC->interpolate_flag = 1;
            else if (strcmp(arg[iarg + 1], "no") == 0) collideMSMC->interpolate_flag = 0;
            else error->all(FLERR, "Illegal collide_msmc_modify command");
            iarg += 2;
        }
        else error->all(FLERR, "Illegal collide_msmc_modify command");

    }
}
