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

#ifdef COLLIDE_CLASS

CollideStyle(msmc, CollideMSMC)

#else
#ifdef COMMAND_CLASS

CommandStyle(collide_msmc_modify, CollideMSMCModify)

#else

#ifndef SPARTA_COLLIDE_MSMC_H
#define SPARTA_COLLIDE_MSMC_H

#include "collide.h"
#include "grid.h"
#include "particle.h"
#include "msmc_math.h"
#include "random_park.h"
#include <vector>

namespace SPARTA_NS {

    class CollideMSMC : public Collide {
    public:
        friend class CollideMSMCModify; // allows modify command to update parameters
        // Constructors and destructor
        CollideMSMC(class SPARTA*, int, char**);
        virtual ~CollideMSMC();
        // Overrides for the Collide base class
        virtual void init();
        virtual void collisions();
        double vremax_init(int, int);
        virtual double attempt_collision(int, int, double);
        double attempt_collision(int, int, int, double) { return 0.0; }; // overload required by Collide
        virtual int test_collision(int, int, int, Particle::OnePart*, Particle::OnePart*) { return 1; };
        virtual void setup_collision(Particle::OnePart*, Particle::OnePart*) { return; };
        virtual int perform_collision(Particle::OnePart*&, Particle::OnePart*&, Particle::OnePart*&);
        double extract(int, int, const char*) { return 0.0; };
        // Physical and model parameters
        enum RelNumModel {
            Z_CONSTANT,
            Z_TEMPERATURE_DEPENDENT
        };
        struct Params {             // MSMC model parameters
            double mu_ref;          // reference viscosity
            double omega;           // mu ~ T^omega
            double T_ref;           // reference temperature
            double d_ref;           // reference molecular diameter
            // --- MSMC polyatomic ---
            double Pr;              // continuum Prandtl number
            double Zrot_cont;       // continuum rotational collision number
            double Zvib_cont;       // continuum vibrational collision number
            int RotRelNum_model;    // rotational relaxation number model
            int VibRelNum_model;    // vibrational relaxation number model
            // --- MSMC polyatomic ---
        };
        // Cell macro cache for conservation correction
        struct ConservMacro
        {
            int done_relaxation;
            double u_pre[3];
            double Etr_pre;
            double Erot_pre;
            double Evib_pre;
        };
        // --- MSMC polyatomic ---
        struct ExpRKProb {
            double vc;
            double dt;
            double vc_dt;
            double Pcoll1;
            double Pcoll2;
            double Prelax;
            double P2c1;
            double P2r;
        };
        struct FadCoeff {
            double yh_s[6];
            double yh_etr, yh_erot, yh_evib;
            double yh_qtr[3];
            double yh_qrot[3];
            double yh_qvib[3];
        };
        struct Thermo {
            double tau_vis;
            double Ls;
            double Ctr;
            double Crot;
            double Cvib;
            double Ctr_rot;
            double Ctr_vib;
            double Ctot;
            msmc_math::Mat3x3 Prot;
            msmc_math::Mat3x3 Pvib;
            msmc_math::Mat3x3 Le;
            msmc_math::Mat3x3 Lq;
        };
        // Conservation correction
        void conserv_correction(int, int*, int);
        // --- MSMC polyatomic ---
        // Diagnostic counters
        bigint count_try_relaxation, count_done_relaxation, count_fail_relaxation;
        bigint count_msmc_cell, count_lowNp_cell, count_lowtemp_cell;
        bigint count_conserv_clip_cell;

    protected:
        enum SolveMode {
            MODE_MSMC,
            MODE_DSMC
        };

        int nmaxconserv;
        ConservMacro* conservMacro;
        int nmaxfad;
        FadCoeff* fadCoeff;
        int nmaxcrmax;
        double* crmax;
        SolveMode solve_mode;

        Params* params;             // MSMC params for each species
        int nparams;                // # of per-species params read in
        int maxglocal;
        int interpolate_flag;       // 1/0 = yes/no do interpolation, default = 1;
        int* resetWmax_flag;        // flag to decide whether resetWmax in current cell
        double resetWmax;           // coefficient to reduce Wmax, default = 0.9999,
        // if resetWmax <= 0, don't do reset
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        double RKc1, RKc2, RKa21;
        double RKw1, RKw2;
        double Pr;                  // Prandtl number
        double time_ave_coef;
        double alpha_Pc;
        // Stochastic rounding helpers
        enum RoundType { ROUND_NONE, ROUND_EVEN, ROUND_ODD };
        int Iround(double z, RoundType type);
        void Iround_group(double P1, double P2, double Pr, int N, int& Nc1, int& Nc2, int& Nr);
        // Binary collision helpers
        void perform_binary_collision(Particle::OnePart* p1, Particle::OnePart* p2, int ic);
        void choose_collision_pair_DSMC(int Ntot, int Nc, std::vector<int>& coll_pair, RanPark* rng);
        // --- MSMC polyatomic ---
        double inelastic_collision_pair_selection(double, int, Particle::OnePart*, Particle::OnePart*);
        void inelastic_collision_BL_smooth(double, double, double, double, double&, double&);
        void inelastic_collision_BL_discrete(double, double, double, double, double&, double&);
        // --- MSMC polyatomic ---
        // MSMC relaxation
        void perform_msmc_relaxation(Particle::OnePart* p, int ic, const Thermo& th);
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
        int nplocalmax;
        // Macroscopic variable evaluation
        void computeMacro();
        // --- MSMC polyatomic ---
        double compute_Trot(const Particle::Species&, double);
        double compute_Tvib(const Particle::Species&, double);
        double dof_vib(const Particle::Species&, double);
        double solve_Teq(const Particle::Species&, double, double, double);
        ExpRKProb build_exprk_prob(int, double);
        double RotRelNum(const Particle::Species&, const Params&, double);
        double VibRelNum(const Particle::Species&, const Params&, double, double, double);
        Thermo updateThermo(const Particle::Species&, const Params&, const CommMacro&, const NoCommMacro&);
        void compute_stress_compensation(const NoCommMacro&, const ExpRKProb&, const Thermo&, FadCoeff&);
        void compute_energy_compensation(const Particle::Species&, const CommMacro&, const NoCommMacro&, const ExpRKProb&, const Thermo&, FadCoeff&);
        void compute_heatflux_compensation(const Particle::Species&, const CommMacro&, const NoCommMacro&, const ExpRKProb&, const Thermo&, FadCoeff&);
        // --- MSMC polyatomic ---
        // Utility helpers
        void read_param_file(char*);
        int wordparse(int, char*, char**);
        void reset_count();
        void print_warning();
    };

    class CollideMSMCModify : protected Pointers {
    public:
        CollideMSMCModify(class SPARTA*);
        ~CollideMSMCModify();
        void command(int, char**);
    };

}

#endif
#endif
#endif
