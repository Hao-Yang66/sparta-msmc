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

#include "string.h"
#include "math.h"
#include "compute_polystate_grid.h"
#include "collide.h"
#include "particle.h"
#include "mixture.h"
#include "grid.h"
#include "update.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

// user keywords

enum{QTRX,QTRY,QTRZ,QROTX,QROTY,QROTZ,QVIBX,QVIBY,QVIBZ,TTR,TROT,TVIB};

// internal accumulators

enum{COUNT,MASSSUM,MVX,MVY,MVZ,MVSQ,
     MVXSQ,MVYSQ,MVZSQ,MVXVY,MVXVZ,MVYVZ,
     MVSQVX,MVSQVY,MVSQVZ,
     ENGROT,ENGVIB,DOFROT,DOFVIB,ENGROTVX,ENGROTVY,ENGROTVZ,
     ENGVIBVX,ENGVIBVY,ENGVIBVZ,LASTSIZE};

// max # of quantities to accumulate for any user value

#define MAXACCUMULATE 10

enum{NONE,DISCRETE,SMOOTH};

/* ---------------------------------------------------------------------- */

ComputePolyStateGrid::ComputePolyStateGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal compute polystate/grid command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,"Compute grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0)
    error->all(FLERR,"Compute polystate/grid mixture ID does not exist");
  ngroup = particle->mixture[imix]->ngroup;

  nvalue = narg - 4;
  value = new int[nvalue];
  tvibflag = 0;

  npergroup = 0;
  unique = new int[LASTSIZE];
  nmap = new int[nvalue];
  group2species = new int[ngroup];
  memory->create(map,ngroup*nvalue,MAXACCUMULATE,"polystate/grid:map");
  for (int i = 0; i < nvalue; i++) nmap[i] = 0;
  for (int i = 0; i < ngroup; i++) group2species[i] = -1;

  int ivalue = 0;
  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"qtrx") == 0) {
      value[ivalue] = QTRX;
      set_map(ivalue,COUNT);
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVX);
      set_map(ivalue,MVY);
      set_map(ivalue,MVZ);
      set_map(ivalue,MVSQ);
      set_map(ivalue,MVXSQ);
      set_map(ivalue,MVXVY);
      set_map(ivalue,MVXVZ);
      set_map(ivalue,MVSQVX);
    } else if (strcmp(arg[iarg],"qtry") == 0) {
      value[ivalue] = QTRY;
      set_map(ivalue,COUNT);
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVX);
      set_map(ivalue,MVY);
      set_map(ivalue,MVZ);
      set_map(ivalue,MVSQ);
      set_map(ivalue,MVYSQ);
      set_map(ivalue,MVXVY);
      set_map(ivalue,MVYVZ);
      set_map(ivalue,MVSQVY);
    } else if (strcmp(arg[iarg],"qtrz") == 0) {
      value[ivalue] = QTRZ;
      set_map(ivalue,COUNT);
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVX);
      set_map(ivalue,MVY);
      set_map(ivalue,MVZ);
      set_map(ivalue,MVSQ);
      set_map(ivalue,MVZSQ);
      set_map(ivalue,MVXVZ);
      set_map(ivalue,MVYVZ);
      set_map(ivalue,MVSQVZ);
    } else if (strcmp(arg[iarg],"qrotx") == 0) {
      value[ivalue] = QROTX;
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVX);
      set_map(ivalue,ENGROT);
      set_map(ivalue,ENGROTVX);
    } else if (strcmp(arg[iarg],"qroty") == 0) {
      value[ivalue] = QROTY;
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVY);
      set_map(ivalue,ENGROT);
      set_map(ivalue,ENGROTVY);
    } else if (strcmp(arg[iarg],"qrotz") == 0) {
      value[ivalue] = QROTZ;
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVZ);
      set_map(ivalue,ENGROT);
      set_map(ivalue,ENGROTVZ);
    } else if (strcmp(arg[iarg],"qvibx") == 0) {
      value[ivalue] = QVIBX;
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVX);
      set_map(ivalue,ENGVIB);
      set_map(ivalue,ENGVIBVX);
    } else if (strcmp(arg[iarg],"qviby") == 0) {
      value[ivalue] = QVIBY;
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVY);
      set_map(ivalue,ENGVIB);
      set_map(ivalue,ENGVIBVY);
    } else if (strcmp(arg[iarg],"qvibz") == 0) {
      value[ivalue] = QVIBZ;
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVZ);
      set_map(ivalue,ENGVIB);
      set_map(ivalue,ENGVIBVZ);
    } else if (strcmp(arg[iarg],"ttr") == 0) {
      value[ivalue] = TTR;
      set_map(ivalue,COUNT);
      set_map(ivalue,MASSSUM);
      set_map(ivalue,MVX);
      set_map(ivalue,MVY);
      set_map(ivalue,MVZ);
      set_map(ivalue,MVSQ);
    } else if (strcmp(arg[iarg],"trot") == 0) {
      value[ivalue] = TROT;
      set_map(ivalue,ENGROT);
      set_map(ivalue,DOFROT);
    } else if (strcmp(arg[iarg],"tvib") == 0) {
      value[ivalue] = TVIB;
      tvibflag = 1;
      set_map(ivalue,COUNT);
      set_map(ivalue,ENGVIB);
      set_map(ivalue,DOFVIB);
    } else error->all(FLERR,"Illegal compute polystate/grid command");

    ivalue++;
    iarg++;
  }

  ntotal = ngroup*npergroup;
  reset_map();

  per_grid_flag = 1;
  size_per_grid_cols = ngroup*nvalue;
  post_process_grid_flag = 1;

  nglocal = 0;
  vector_grid = NULL;
  tally = NULL;
}

/* ---------------------------------------------------------------------- */

ComputePolyStateGrid::~ComputePolyStateGrid()
{
  if (copymode) return;

  delete [] value;
  delete [] unique;
  delete [] group2species;

  delete [] nmap;
  memory->destroy(map);

  memory->destroy(vector_grid);
  memory->destroy(tally);
}

/* ---------------------------------------------------------------------- */

void ComputePolyStateGrid::init()
{
  if (ngroup != particle->mixture[imix]->ngroup)
    error->all(FLERR,"Number of groups in compute polystate/grid mixture "
               "has changed");

  particle->mixture[imix]->init();
  int *groupsize = particle->mixture[imix]->groupsize;
  int **groupspecies = particle->mixture[imix]->groupspecies;
  for (int igroup = 0; igroup < ngroup; igroup++) {
    if (groupsize[igroup] == 1) group2species[igroup] = groupspecies[igroup][0];
    else group2species[igroup] = -1;
  }

  if (tvibflag) {
    for (int igroup = 0; igroup < ngroup; igroup++)
      if (group2species[igroup] < 0)
        error->all(FLERR,"Compute polystate/grid tvib requires one species "
                   "per mixture group");
  }

  reallocate();
}

/* ---------------------------------------------------------------------- */

void ComputePolyStateGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;
  int *s2g = particle->mixture[imix]->species2group;
  int nlocal = particle->nlocal;

  int i,j,k,m,ispecies,igroup,icell;
  double mass,vsq;
  double *v,*vec;

  for (i = 0; i < nglocal; i++)
    for (j = 0; j < ntotal; j++)
      tally[i][j] = 0.0;

  for (i = 0; i < nlocal; i++) {
    ispecies = particles[i].ispecies;
    igroup = s2g[ispecies];
    if (igroup < 0) continue;
    icell = particles[i].icell;
    if (!(cinfo[icell].mask & groupbit)) continue;

    mass = species[ispecies].mass;
    v = particles[i].v;
    vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];

    vec = tally[icell];
    k = igroup*npergroup;

    for (m = 0; m < npergroup; m++) {
      switch (unique[m]) {
      case COUNT:
        vec[k++] += 1.0;
        break;
      case MASSSUM:
        vec[k++] += mass;
        break;
      case MVX:
        vec[k++] += mass*v[0];
        break;
      case MVY:
        vec[k++] += mass*v[1];
        break;
      case MVZ:
        vec[k++] += mass*v[2];
        break;
      case MVSQ:
        vec[k++] += mass*vsq;
        break;
      case MVXSQ:
        vec[k++] += mass*v[0]*v[0];
        break;
      case MVYSQ:
        vec[k++] += mass*v[1]*v[1];
        break;
      case MVZSQ:
        vec[k++] += mass*v[2]*v[2];
        break;
      case MVXVY:
        vec[k++] += mass*v[0]*v[1];
        break;
      case MVXVZ:
        vec[k++] += mass*v[0]*v[2];
        break;
      case MVYVZ:
        vec[k++] += mass*v[1]*v[2];
        break;
      case MVSQVX:
        vec[k++] += mass*vsq*v[0];
        break;
      case MVSQVY:
        vec[k++] += mass*vsq*v[1];
        break;
      case MVSQVZ:
        vec[k++] += mass*vsq*v[2];
        break;
      case ENGROT:
        vec[k++] += particles[i].erot;
        break;
      case ENGVIB:
        vec[k++] += particles[i].evib;
        break;
      case DOFROT:
        vec[k++] += species[ispecies].rotdof;
        break;
      case DOFVIB:
        vec[k++] += species[ispecies].vibdof;
        break;
      case ENGROTVX:
        vec[k++] += particles[i].erot*v[0];
        break;
      case ENGROTVY:
        vec[k++] += particles[i].erot*v[1];
        break;
      case ENGROTVZ:
        vec[k++] += particles[i].erot*v[2];
        break;
      case ENGVIBVX:
        vec[k++] += particles[i].evib*v[0];
        break;
      case ENGVIBVY:
        vec[k++] += particles[i].evib*v[1];
        break;
      case ENGVIBVZ:
        vec[k++] += particles[i].evib*v[2];
        break;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   query info about internal tally array for this compute
------------------------------------------------------------------------- */

int ComputePolyStateGrid::query_tally_grid(int index, double **&array, int *&cols)
{
  index--;
  int ivalue = index % nvalue;
  array = tally;
  cols = map[index];
  return nmap[ivalue];
}

/* ----------------------------------------------------------------------
   tally accumulated info to compute final normalized values
------------------------------------------------------------------------- */

void ComputePolyStateGrid::post_process_grid(int index, int nsample,
                                             double **etally, int *emap,
                                             double *vec, int nstride)
{
  index--;
  int ivalue = index % nvalue;

  int lo = 0;
  int hi = nglocal;
  int k = 0;

  if (!etally) {
    nsample = 1;
    etally = tally;
    emap = map[index];
    vec = vector_grid;
    nstride = 1;
  }

  double fnum = update->fnum;
  Grid::ChildInfo *cinfo = grid->cinfo;

  for (int icell = lo; icell < hi; icell++) {
    double volume = cinfo[icell].volume;
    if (volume == 0.0) {
      vec[k] = 0.0;
      k += nstride;
      continue;
    }

    double wt = fnum * cinfo[icell].weight /
      grid->cells[icell].dt_weight / volume;
    double *values = etally[icell];

    switch (value[ivalue]) {

    case QTRX:
    case QTRY:
    case QTRZ:
      {
        double count = values[emap[0]];
        double countavg = count / nsample;
        double summass = values[emap[1]];
        if (countavg <= 2.0 || summass == 0.0) vec[k] = 0.0;
        else {
          double ux = values[emap[2]] / summass;
          double uy = values[emap[3]] / summass;
          double uz = values[emap[4]] / summass;
          double u2 = ux*ux + uy*uy + uz*uz;
          double mvsq = values[emap[5]];
          double prefactor = 0.5 * (countavg / (countavg - 2.0)) * wt / nsample;
          double qinst;

          if (value[ivalue] == QTRX) {
            qinst = values[emap[9]]
              - ux*mvsq
              + 2.0*summass*u2*ux
              - 2.0*(ux*values[emap[6]] + uy*values[emap[7]] + uz*values[emap[8]]);
          } else if (value[ivalue] == QTRY) {
            qinst = values[emap[9]]
              - uy*mvsq
              + 2.0*summass*u2*uy
              - 2.0*(ux*values[emap[7]] + uy*values[emap[6]] + uz*values[emap[8]]);
          } else {
            qinst = values[emap[9]]
              - uz*mvsq
              + 2.0*summass*u2*uz
              - 2.0*(ux*values[emap[7]] + uy*values[emap[8]] + uz*values[emap[6]]);
          }

          vec[k] = prefactor * qinst;
        }
        break;
      }

    case QROTX:
    case QROTY:
    case QROTZ:
      {
        double summass = values[emap[0]];
        if (summass == 0.0) vec[k] = 0.0;
        else {
          double ucomp = values[emap[1]] / summass;
          vec[k] = wt/nsample * (values[emap[3]] - values[emap[2]]*ucomp);
        }
        break;
      }

    case QVIBX:
    case QVIBY:
    case QVIBZ:
      {
        double summass = values[emap[0]];
        if (summass == 0.0) vec[k] = 0.0;
        else {
          double ucomp = values[emap[1]] / summass;
          vec[k] = wt/nsample * (values[emap[3]] - values[emap[2]]*ucomp);
        }
        break;
      }

    case TTR:
      {
        double count = values[emap[0]];
        double countavg = count / nsample;
        double summass = values[emap[1]];
        if (countavg <= 1.0 || summass == 0.0) vec[k] = 0.0;
        else {
          double mom2 = values[emap[2]]*values[emap[2]] +
            values[emap[3]]*values[emap[3]] + values[emap[4]]*values[emap[4]];
          vec[k] = (countavg / (countavg - 1.0)) *
            (values[emap[5]] - mom2/summass) / (count * 3.0 * update->boltz);
        }
        break;
      }

    case TROT:
      {
        double dof = values[emap[1]];
        if (dof == 0.0) vec[k] = 0.0;
        else vec[k] = 2.0 * values[emap[0]] / (dof * update->boltz);
        break;
      }

    case TVIB:
      {
        int igroup = index / nvalue;
        int ispecies = group2species[igroup];
        if (ispecies < 0) vec[k] = 0.0;
        else {
          Particle::Species &sp = particle->species[ispecies];
          double evib_avg = 0.0;
          double count = values[emap[0]];
          if (count > 0.0) evib_avg = values[emap[1]] / count;

          if (collide->vibstyle == NONE || sp.vibdof <= 0 || evib_avg <= 0.0)
            vec[k] = 0.0;
          else if (collide->vibstyle == DISCRETE) {
            const double theta0_vib = sp.vibtemp[0];
            if (theta0_vib <= 0.0) vec[k] = 0.0;
            else {
              const double ibar = evib_avg / (update->boltz * theta0_vib);
              if (ibar <= 0.0) vec[k] = 0.0;
              else vec[k] = theta0_vib / log(1.0 + 1.0 / ibar);
            }
          } else {
            double dof = values[emap[2]];
            if (dof == 0.0) vec[k] = 0.0;
            else vec[k] = 2.0 * values[emap[1]] / (dof * update->boltz);
          }
        }
        break;
      }
    }

    k += nstride;
  }
}

/* ----------------------------------------------------------------------
   reallocate arrays if nglocal has changed
------------------------------------------------------------------------- */

void ComputePolyStateGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  memory->destroy(tally);
  nglocal = grid->nlocal;
  memory->create(vector_grid,nglocal,"polystate/grid:vector_grid");
  memory->create(tally,nglocal,ntotal,"polystate/grid:tally");
}

/* ----------------------------------------------------------------------
   memory usage of local grid-based data
------------------------------------------------------------------------- */

bigint ComputePolyStateGrid::memory_usage()
{
  bigint bytes = 0;
  bytes += nglocal * sizeof(double);
  bytes += ntotal*nglocal * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   setup map from user values to internal tallies
------------------------------------------------------------------------- */

void ComputePolyStateGrid::set_map(int ivalue, int name)
{
  int i;
  for (i = 0; i < npergroup; i++)
    if (unique[i] == name) break;

  if (i == npergroup) unique[npergroup++] = name;
  for (int igroup = 0; igroup < ngroup; igroup++)
    map[igroup*nvalue+ivalue][nmap[ivalue]] = i;
  nmap[ivalue]++;
}

/* ----------------------------------------------------------------------
   reset indices in map once npergroup is finalized
------------------------------------------------------------------------- */

void ComputePolyStateGrid::reset_map()
{
  for (int i = 0; i < ngroup*nvalue; i++)
    for (int j = 0; j < nmap[i % nvalue]; j++)
      map[i][j] += (i / nvalue) * npergroup;
}
