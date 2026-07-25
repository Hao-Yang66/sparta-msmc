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

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "read_particles.h"
#include "particle.h"
#include "grid.h"
#include "domain.h"
#include "comm.h"
#include "input.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

enum{UNKNOWN,OUTSIDE,INSIDE,OVERLAP};   // same as Grid

#define MAXLINE 1024        // max line length in dump file
#define CHUNK 1024

/* ---------------------------------------------------------------------- */

ReadParticles::ReadParticles(SPARTA *sparta) : Pointers(sparta) {}

/* ---------------------------------------------------------------------- */

void ReadParticles::command(int narg, char **arg)
{
  if (!grid->exist)
    error->all(FLERR,"Cannot read particles before grid is defined");

  particle->exist = 1;

  if (narg != 2) error->all(FLERR,"Illegal read_particles command");

  // process args

  char *file = arg[0];
  bigint nrequest = input->bnumeric(FLERR,arg[1]);

  me = comm->me;
  nspecies = particle->nspecies;
  line = new char[MAXLINE];
  nfield_read = 8;
  ifield_id = 0;
  ifield_ispecies = 1;
  ifield_x = 2;
  ifield_y = 3;
  ifield_z = 4;
  ifield_vx = 5;
  ifield_vy = 6;
  ifield_vz = 7;
  ifield_erot = -1;
  ifield_evib = -1;

  MPI_Barrier(world);
  double time1 = MPI_Wtime();

  // open file on proc 0

  if (me == 0) {
    fp = fopen(file,"r");
    if (fp == NULL) error->one(FLERR,"Read_particles could not open file");
  }

  // scan file for dump snapshot with correct timestamp
  // exit loop when dump timestep >= nrequest
  // np = # of particles in snapshot

  int eofflag;
  bigint ntimestep = -1;

  if (me == 0) {
    while (1) {
      eofflag = read_time(ntimestep);
      if (eofflag) break;
      if (ntimestep >= nrequest) break;
      skip();
      if (ntimestep >= nrequest) break;
    }
  }

  MPI_Bcast(&ntimestep,1,MPI_SPARTA_BIGINT,0,world);
  if (ntimestep != nrequest)
    error->all(FLERR,"Read_particles could not find timestep in file");

  int np;
  if (me == 0) np = read_header();
  MPI_Bcast(&np,1,MPI_INT,0,world);
  MPI_Bcast(&nfield_read,1,MPI_INT,0,world);

  int ifields[10];
  if (me == 0) {
    ifields[0] = ifield_id;
    ifields[1] = ifield_ispecies;
    ifields[2] = ifield_x;
    ifields[3] = ifield_y;
    ifields[4] = ifield_z;
    ifields[5] = ifield_vx;
    ifields[6] = ifield_vy;
    ifields[7] = ifield_vz;
    ifields[8] = ifield_erot;
    ifields[9] = ifield_evib;
  }
  MPI_Bcast(ifields,10,MPI_INT,0,world);
  ifield_id = ifields[0];
  ifield_ispecies = ifields[1];
  ifield_x = ifields[2];
  ifield_y = ifields[3];
  ifield_z = ifields[4];
  ifield_vx = ifields[5];
  ifield_vy = ifields[6];
  ifield_vz = ifields[7];
  ifield_erot = ifields[8];
  ifield_evib = ifields[9];

  // read, broadcast, and process particles from snapshot in chunks
  // required fields are ID,type/ispecies,x,y,z,vx,vy,vz
  // optional fields are erot and evib

  double **fields;
  memory->create(fields,CHUNK,nfield_read,"read_particles:fields");

  int nlocal_previous = particle->nlocal;
  bigint nglobal_previous = particle->nglobal;

  int nchunk;
  bigint nread = 0;
  while (nread < np) {
    nchunk = MIN(np-nread,CHUNK);
    if (me == 0) read_particles(nchunk,nfield_read,fields);
    MPI_Bcast(&fields[0][0],nchunk*nfield_read,MPI_DOUBLE,0,world);
    process_particles(nchunk,nfield_read,fields);
    nread += nchunk;
  }

  memory->destroy(fields);

  MPI_Barrier(world);
  double time2 = MPI_Wtime();

  // check that no read-in particle species is invalid

  Particle::OnePart *particles = particle->particles;
  int nlocal = particle->nlocal;

  int flag = 0;

  for (int i = nlocal_previous; i < nlocal; i++)
    if (particles[i].ispecies > nspecies) flag++;

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flag) {
    char str[128];
    sprintf(str,"%d read-in particles have invalid species",flag);
    error->all(FLERR,str);
  }

  // check that no read-in particle is in an INSIDE cell

  Grid::ChildInfo *cinfo = grid->cinfo;

  flag = 0;
  for (int i = nlocal_previous; i < nlocal; i++)
    if (cinfo[particles[i].icell].type == INSIDE) flag++;

  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flag) {
    char str[128];
    sprintf(str,"%d read-in particles are inside surface",flag);
    error->all(FLERR,str);
  }

  // close file

  if (me == 0) fclose(fp);
  delete [] line;

  // print stats

  bigint nme = particle->nlocal;
  MPI_Allreduce(&nme,&particle->nglobal,1,MPI_SPARTA_BIGINT,MPI_SUM,world);
  bigint nactual = particle->nglobal - nglobal_previous;

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"Read " BIGINT_FORMAT " particles out of "
              "%d\n",nactual,np);
      fprintf(screen,"  CPU time = %g secs\n",time2-time1);
    }
    if (logfile) {
      fprintf(logfile,"Read " BIGINT_FORMAT " particles out of "
              "%d\n",nactual,np);
      fprintf(logfile,"  CPU time = %g secs\n",time2-time1);
    }
  }
}

/* ----------------------------------------------------------------------
   process N particles and their fields read from dump file
   store the ones in grid cells I own
   required fields are id,type/ispecies,x,y,z,vx,vy,vz
   optional fields are erot,evib
------------------------------------------------------------------------- */

void ReadParticles::process_particles(int n, int, double **fields)
{
  int id,ispecies,icell;
  double x[3],v[3],erot,evib;

  Grid::ChildCell *cells = grid->cells;
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;

  for (int i = 0; i < n; i++) {
    x[0] = fields[i][ifield_x];
    x[1] = fields[i][ifield_y];
    x[2] = fields[i][ifield_z];

    // discard particles outside simulation box

    if (x[0] < boxlo[0] || x[0] > boxhi[0] ||
        x[1] < boxlo[1] || x[1] > boxhi[1] ||
        x[2] < boxlo[2] || x[2] > boxhi[2]) continue;

    // id_find_child() recurses from root cell to find owning child cell
    // assumes x is inside or on surface of parent cell
    // returned icell can be owned or ghost cell

    icell = grid->id_find_child(0,0,boxlo,boxhi,x);
    if (icell < 0 || cells[icell].proc != me) continue;

    id = static_cast<int> (fields[i][ifield_id]);
    ispecies = static_cast<int> (fields[i][ifield_ispecies]) - 1;
    v[0] = fields[i][ifield_vx];
    v[1] = fields[i][ifield_vy];
    v[2] = fields[i][ifield_vz];
    erot = 0.0;
    evib = 0.0;
    if (ifield_erot >= 0) erot = fields[i][ifield_erot];
    if (ifield_evib >= 0) evib = fields[i][ifield_evib];

    particle->add_particle(id,ispecies,icell,x,v,erot,evib);
  }
}

/* ----------------------------------------------------------------------
   read and return time stamp from dump file
   if first read reaches end-of-file, return 1
   only called by proc 0
------------------------------------------------------------------------- */

int ReadParticles::read_time(bigint &ntimestep)
{
  char *eof = fgets(line,MAXLINE,fp);
  if (eof == NULL) return 1;

  if (strstr(line,"ITEM: TIMESTEP") != line)
    error->one(FLERR,"Read_particles file is incorrectly formatted");
  read_lines(1);
  sscanf(line,BIGINT_FORMAT,&ntimestep);

  return 0;
}

/* ----------------------------------------------------------------------
   skip snapshot from timestamp onward
   only called by proc 0
------------------------------------------------------------------------- */

void ReadParticles::skip()
{
  read_lines(2);
  bigint natoms;
  sscanf(line,BIGINT_FORMAT,&natoms);

  read_lines(5);

  // invoke read_lines() in chunks no larger than MAXSMALLINT

  int nchunk;
  bigint nremain = natoms;
  while (nremain) {
    nchunk = MIN(nremain,MAXSMALLINT);
    read_lines(nchunk);
    nremain -= nchunk;
  }
}

/* ----------------------------------------------------------------------
   read remaining dump snapshot header info
   skip box bounds, parse optional column info
   return natoms
   only called by proc 0
------------------------------------------------------------------------- */

bigint ReadParticles::read_header()
{
  bigint natoms;
  read_lines(2);
  sscanf(line,BIGINT_FORMAT,&natoms);

  read_lines(4);
  read_lines(1);

  int nwords = input->count_words(line);
  if (nwords > 2) {
    char *copy = new char[strlen(line)+1];
    strcpy(copy,line);

    strtok(copy," \t\n\r\f");
    strtok(NULL," \t\n\r\f");

    nfield_read = nwords - 2;
    ifield_id = ifield_ispecies = -1;
    ifield_x = ifield_y = ifield_z = -1;
    ifield_vx = ifield_vy = ifield_vz = -1;
    ifield_erot = ifield_evib = -1;

    int m = 0;
    char *word = strtok(NULL," \t\n\r\f");
    while (word) {
      if (strcmp(word,"id") == 0) ifield_id = m;
      else if (strcmp(word,"type") == 0 || strcmp(word,"ispecies") == 0)
        ifield_ispecies = m;
      else if (strcmp(word,"x") == 0) ifield_x = m;
      else if (strcmp(word,"y") == 0) ifield_y = m;
      else if (strcmp(word,"z") == 0) ifield_z = m;
      else if (strcmp(word,"vx") == 0) ifield_vx = m;
      else if (strcmp(word,"vy") == 0) ifield_vy = m;
      else if (strcmp(word,"vz") == 0) ifield_vz = m;
      else if (strcmp(word,"erot") == 0) ifield_erot = m;
      else if (strcmp(word,"evib") == 0) ifield_evib = m;
      m++;
      word = strtok(NULL," \t\n\r\f");
    }

    delete [] copy;

    if (ifield_id < 0 || ifield_ispecies < 0 ||
        ifield_x < 0 || ifield_y < 0 || ifield_z < 0 ||
        ifield_vx < 0 || ifield_vy < 0 || ifield_vz < 0)
      error->one(FLERR,"Read_particles file is missing required atom columns");
  }

  return natoms;
}

/* ----------------------------------------------------------------------
   read N particle lines from dump file
   stores appropriate values in fields array
   return 0 if success, 1 if error
   only called by proc 0
------------------------------------------------------------------------- */

void ReadParticles::read_particles(int n, int nfield, double **fields)
{
  int i,m,nwords;
  char *eof,*word;

  for (i = 0; i < n; i++) {
    eof = fgets(line,MAXLINE,fp);
    if (eof == NULL) error->one(FLERR,"Unexpected end of read_particles file");

    if (i == 0) {
      nwords = input->count_words(line);
      if (nwords != nfield)
        error->one(FLERR,"Bad particle line in read_particles file");
    }

    // tokenize the line and convert words to fields

    for (m = 0; m < nfield; m++) {
      if (m == 0) word = strtok(line," \t\n\r\f");
      else word = strtok(NULL," \t\n\r\f");
      if (word == NULL)
        error->one(FLERR,"Bad particle line in read_particles file");
      fields[i][m] = atof(word);
    }
  }
}

/* ----------------------------------------------------------------------
   read N lines from dump file
   only last one is saved in line
   only called by proc 0
------------------------------------------------------------------------- */

void ReadParticles::read_lines(int n)
{
  char *eof = NULL;
  if (n <= 0) return;
  for (int i = 0; i < n; i++) eof = fgets(line,MAXLINE,fp);
  if (eof == NULL) error->one(FLERR,"Unexpected end of read_particles file");
}
