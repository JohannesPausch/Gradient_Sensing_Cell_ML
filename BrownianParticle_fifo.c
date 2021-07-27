#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sys/resource.h>
#include "Gradient_Sensing_Cell_ML_git_stamps.h"

/* This code is based on BrownianParticle.c
 *  + signalling particles are released from the origin.
 *  + there might be some initial warm-up
 *  + reads from fifo for initial pos of cell
 *  + spits out signalling particles' positions on cell on another fifo
 *  + reads from stdin updated pos
 *
 *  mkfifo SignalArrivals
 *  mkfifo CellPosition
 *  ./BrownianParticle_fifo -o SignalArrivals -i CellPosition
 *
 *  echo "1.0 1.0 1.0" >> CellPosition
 */


/* This code generates N trajectories of Brownian particles.
 * They eminate from a source at the origin and proceeed until
 * their distances is greater than cutoff or until they hit the
 * surface of a sphere (center on x-axis at distance d radius r).
 *
 * The output is 
 * time coordinate
 *
 * Plus other output. 
 *
 * Potential optimisation: Make every trajectory count by assuming that 
 * a sphere was in the way of a trajectory that reaches the cutoff.
 *
 * At the moment, I don't bother much about optimising -- I draw normally
 * distributed rvs.
 *
 *
 * make BrownianParticle
 * ./BrownianParticle > BrownianParticle_ref.dat &
 * grep EVENT BrownianParticle_ref.dat | sed 's/.*EVENT //' > BrownianParticle_ref.txt
 * gnuplot> sp 'BrownianParticle_ref.txt' u 5:6:7
 *
 * ./BrownianParticle -p 1.001 > BrownianParticle_ref2.dat
 * ... has a large number of particles arriving at the point closest to the origin.
 *
 * Validation:
 * Look at the histogram of column 5, that's the x-coordinate at arrival.
 * How does that compare to theory?
 */


/*
 * double gsl_ran_gaussian_ziggurat(const gsl_rng *r, double sigma)
 */


/* Distance the cue particle has to diffusive before considered "lost". */
#ifndef CUTOFF
#define CUTOFF (20.)
#endif
double param_cutoff=CUTOFF;
double param_cutoff_squared=CUTOFF*CUTOFF;

/* Radius of the cell sphere. */ 
#ifndef SPHERE_RADIUS
#define SPHERE_RADIUS (1.0)
#endif
double param_sphere_radius=SPHERE_RADIUS;
double param_sphere_radius_squared=SPHERE_RADIUS*SPHERE_RADIUS;

/* Time step length */
#ifndef DELTA_T
#define DELTA_T (0.001)
#endif 
double param_delta_t=DELTA_T;

/* Diffusion constant */
#ifndef DIFFUSION
#define DIFFUSION (1.0)
#endif
double param_diffusion=DIFFUSION;
double param_sigma=0.;

/* Release rate. */
#ifndef RELEASE_RATE
#define RELEASE_RATE (1.0)
#endif 
double param_release_rate=RELEASE_RATE;

/* Max particles */
#ifndef MAX_PARTICLES
#define MAX_PARTICLES (1000000)
#endif
int param_max_particles=MAX_PARTICLES;


/* Seed */
#ifndef SEED
#define SEED (5UL)
#endif
unsigned long int param_seed=SEED;

#ifndef PATH_MAX
#define PATH_MAX (1024)
#endif

char param_output[PATH_MAX]={0};
char param_input[PATH_MAX]={0};

double param_warmup_time=0.;


gsl_rng *rng;

#define MALLOC(a,n) if ((a=malloc(sizeof(*a)*(n)))==NULL) { fprintf(stderr, "Not enough memory for %s, requested %i bytes, %i items of size %i. %i::%s\n", #a, (int)(sizeof(*a)*n), n, (int)sizeof(*a), errno, strerror(errno)); exit(EXIT_FAILURE); } else { printf("# Info malloc(3)ed %i bytes (%i items of %i bytes) for %s.\n", (int)(sizeof(*a)*(n)), n, (int)sizeof(*a), #a); }


void postamble(void);

/* Each signalling particle has a position relative to the origin.
 * There are at most N signalling particles.
 */

typedef struct {
double x, y, z;
double release_time;
} particle_strct;


int main(int argc, char *argv[])
{
int ch;
double tm=0.;
particle_strct *particle;
particle_strct cell;
FILE *fin, *fout;
int active_particles, total_particles;
double origin_distance2, sphere_distance2;


#define STRCPY(dst,src) strncpy(dst,src,sizeof(dst)-1); dst[sizeof(dst)-1]=(char)0


setlinebuf(stdout);
while ((ch = getopt(argc, argv, "c:d:i:N:o:r:S:t:w:")) != -1) {
  switch (ch) {
    case'c':
      param_cutoff=strtod(optarg, NULL);
      break;
    case'd':
      param_diffusion=strtod(optarg, NULL);
      break;
    case 'i':
      STRCPY(param_input, optarg);
      break;
    case 'N':
      param_max_particles=strtoll(optarg, NULL, 10);
      break;
    case 'o':
      STRCPY(param_output, optarg);
      break;
    case 'r':
      param_release_rate=strtod(optarg, NULL);
      break;
    case 'S':
      param_seed=strtoul(optarg, NULL, 10);
      break;
    case 't':
      param_delta_t=strtod(optarg, NULL);
      break;
    case 'w':
      param_warmup_time=strtod(optarg, NULL);
      break;
    default:
      fprintf(stderr, "Unknown flag %c.\n", ch);
      exit(EXIT_FAILURE);
      break;
    }
  }



{ 
int i;

printf("# Info: Command: %s", argv[0]);
for (i=1; i<argc; i++) printf(" \"%s\"", argv[i]);
printf("\n");
}


/* Some infos. */
{
time_t tim;
tim=time(NULL);

printf("# Info Starting at %s", ctime(&tim));
}

/* For version control if present. */
printf("# Info: Version of git_version_string to follow.\n");
printf("%s", git_version_string);
printf("# $Header$\n");


/* Hostname */
{ char hostname[128];
  gethostname(hostname, sizeof(hostname)-1);
  hostname[sizeof(hostname)-1]=(char)0;
  printf("# Info: Hostname: %s\n", hostname);
}


/* Dirname */
{ char cwd[1024];
  cwd[0]=0;
  if(getcwd(cwd, sizeof(cwd)-1)!=NULL){
  cwd[sizeof(cwd)-1]=(char)0;
  printf("# Info: Directory: %s\n", cwd);}
}

/* Process ID. */
printf("# Info: PID: %i\n", (int)getpid());

#define PRINT_PARAM(a,o, f) printf("#Info: %s: %s " f "\n", #a, o, a)

param_sigma=sqrt(2.*param_delta_t*param_diffusion);
param_cutoff_squared=param_cutoff*param_cutoff;
param_sphere_radius_squared=param_sphere_radius*param_sphere_radius;
PRINT_PARAM(param_delta_t, "-t", "%g");
PRINT_PARAM(param_diffusion, "-d", "%g");
PRINT_PARAM(param_sigma, "", "%g");
PRINT_PARAM(param_cutoff, "-c", "%g");
PRINT_PARAM(param_cutoff_squared, "", "%g");
PRINT_PARAM(param_sphere_radius, "", "%g");
PRINT_PARAM(param_sphere_radius_squared, "", "%g");
PRINT_PARAM(param_release_rate, "-r", "%g");
PRINT_PARAM(param_warmup_time, "-w", "%g");
PRINT_PARAM(param_max_particles, "-N", "%i");
PRINT_PARAM(param_seed, "-S", "%lu");
PRINT_PARAM(param_input, "-i", "%s");
PRINT_PARAM(param_output, "-o", "%s");

if (param_output[0]) {
  if ((fout=fopen(param_output, "wt"))==NULL) {
    fprintf(stderr, "Cannot open file %s for writing. %i::%s\n", param_output, errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
  setlinebuf(fout);
} else fout=stdout;
printf("# Info: Output open.\n");

if (param_input[0]) {
  if ((fin=fopen(param_input, "rt"))==NULL) {
    fprintf(stderr, "Cannot open file %s for reading. %i::%s\n", param_input, errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
} else fin=stdin;
printf("# Info: Input open.\n");

rng=gsl_rng_alloc(gsl_rng_taus2);
gsl_rng_set(rng, SEED);


MALLOC(particle, param_max_particles);
active_particles=0;
total_particles=0;

#define CREATE_NEW_PARTICLE particle[active_particles].x=particle[active_particles].y=particle[active_particles].z=0.; particle[active_particles].release_time=tm; active_particles++; total_particles++;


CREATE_NEW_PARTICLE;

if (fscanf(fin, "%lg %lg %lg", &(cell.x), &(cell.y), &(cell.z))!=3) {
  fprintf(stderr, "Error: fscanf returned without all three conversions. %i::%s\n", errno, strerror(errno));
  exit(EXIT_FAILURE);
}
printf("# Info: Initial cell position %g %g %g\n", (cell.x), (cell.y), (cell.z));

for (tm=0.; ;tm+=param_delta_t) {
  int i;


  if (param_release_rate*param_delta_t>gsl_ran_flat(rng, 0., 1.)) {
    if (active_particles>=param_max_particles) {
      fprintf(stderr, "Warning: particle creation suppressed because active_particles=%i >= param_max_particles=%i.\n", active_particles, param_max_particles);
    } else {
      CREATE_NEW_PARTICLE;
      printf("# Info: New particle created at time %g. Active: %i, Max: %i, Total: %i\n", tm, active_particles, param_max_particles, total_particles);
    }
  }

  for (i=0; i<active_particles; i++) {
    particle[i].x+=gsl_ran_gaussian_ziggurat(rng, param_sigma);
    particle[i].y+=gsl_ran_gaussian_ziggurat(rng, param_sigma);
    particle[i].z+=gsl_ran_gaussian_ziggurat(rng, param_sigma);
    origin_distance2 = particle[i].x*particle[i].x + particle[i].y*particle[i].y + particle[i].z*particle[i].z;


    if (origin_distance2>param_cutoff_squared) {
      printf("# Info: Loss. Particle %i of %i actives (max %i total generated %i) got lost to position %g %g %g at time %g at distance %g>%g having started at time %g.\n", 
	i, active_particles, param_max_particles, total_particles,
        particle[i].x, particle[i].y, particle[i].z, tm, sqrt(origin_distance2), param_cutoff, particle[i].release_time);
      active_particles--;
      particle[i]=particle[active_particles];
      /* This is a brutal way of dealing with active_particles-1, which has just been copied into i, to remove i: */
      i--;
      continue;
    }

    sphere_distance2=(particle[i].x-cell.x)*(particle[i].x-cell.x) + (particle[i].y-cell.y)*(particle[i].y-cell.y) + (particle[i].z-cell.z)*(particle[i].z-cell.z);
    if (sphere_distance2<param_sphere_radius_squared) {
      printf("# Info: Arrival. Particle %i of %i actives (max %i total generated %i) arrived at the cell at position %g %g %g at time %g at distance %g<%g having started at time %g.\n", 
        i, active_particles, param_max_particles, total_particles,
        particle[i].x, particle[i].y, particle[i].z, tm, sqrt(sphere_distance2), param_sphere_radius, particle[i].release_time);
      if (tm>param_warmup_time) {
        fprintf(fout, "%g %g %g %g\n", particle[i].x, particle[i].y, particle[i].z, tm);
	/* It looks like when I terminate the fscanf string by a \n then it tries to gobble as much whitespace as possible, so it waits until no-whitespace? */
        if (fscanf(fin, "%lg %lg %lg", &(cell.x), &(cell.y), &(cell.z))!=3) {
	  fprintf(stderr, "Error: fscanf returned without all three conversions. %i::%s\n", errno, strerror(errno));
	  exit(EXIT_FAILURE);
	}
        printf("# Info: New cell position %g %g %g\n", (cell.x), (cell.y), (cell.z));
      }
      active_particles--;
      particle[i]=particle[active_particles];
      i--;
      continue;

    }
  }
}


postamble();
return(0);
}






void postamble(void)
{
time_t tm;
struct rusage rus;

tm=time(NULL);

printf("# Info Terminating at %s", ctime(&tm));
if (getrusage(RUSAGE_SELF, &rus)) {
  printf("# Info getrusage(2) failed.\n");
} else {
printf("# Info getrusage.ru_utime: %li.%06li\n", (long int)rus.ru_utime.tv_sec, (long int)rus.ru_utime.tv_usec);
printf("# Info getrusage.ru_stime: %li.%06li\n", (long int)rus.ru_stime.tv_sec, (long int)rus.ru_stime.tv_usec);

#define GETRUSAGE_long(f) printf("# Info getrusage.%s: %li\n", #f, rus.f);
GETRUSAGE_long(ru_maxrss);
GETRUSAGE_long(ru_ixrss);
GETRUSAGE_long(ru_idrss);
GETRUSAGE_long(ru_isrss);
GETRUSAGE_long(ru_minflt);
GETRUSAGE_long(ru_majflt);
GETRUSAGE_long(ru_nswap);
GETRUSAGE_long(ru_inblock);
GETRUSAGE_long(ru_oublock);
GETRUSAGE_long(ru_msgsnd);
GETRUSAGE_long(ru_msgrcv);
GETRUSAGE_long(ru_nsignals);
GETRUSAGE_long(ru_nvcsw);
GETRUSAGE_long(ru_nivcsw);
}
printf("#Info: Good bye and thanks for all the fish.\n");
}

