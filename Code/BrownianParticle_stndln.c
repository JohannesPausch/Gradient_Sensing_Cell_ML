#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sys/resource.h>
#include "../Gradient_Sensing_Cell_ML_git_stamps.h"


/*
 * ./BrownianParticle_stndln -v -w 500 -T tmp.txt -s 0,0,4
 * ./BrownianParticle_stndln -h 1 -w 500 -T BrownianParticle_stndln_traj.txt -s 0,0,4
 * After a warm-up of 500 time units, the cell is allowed to move (originally parked at the origin, with the source at 0,0,4).
 * The cue particle might be inside the cell at that time. 
 * XXX Improvement needed: Blast all particles away that are inside the cell after warmup.
 * The cell then runs according to the cues.
 * Its trajectory is written into BrownianParticle_stndln_traj.txt and a snapshot of all coos is written into
 * BrownianParticle_stndln_snapshot*.txt
 *
 * The first line in that file is the cell's position followed by the number of active particles and then the time stamp.
 * The second line is the position of the source.
 * gnuplot> sp 'BrownianParticle_stndln_traj.txt' u 2:3:4 w l
 * gnuplot> p 'BrownianParticle_stndln_traj.txt' u 1:(sqrt($2*$2+$3*$3+(($4-4.)*($4-4.)))) w l
 * ./BrownianParticle_stndln -w 500 -T BrownianParticle_stndln_traj4.txt -s 0,0,4
 * ./BrownianParticle_stndln -w 500 -T BrownianParticle_stndln_traj6.txt -s 0,0,6
 */

#define VERSION_ACCORDING_TO_GUNNAR "MAGIC_VATG 20220306_123227"


/* This code is based on BrownianParticle.c
 *  + signalling particles are released from the origin.
 *  + there might be some initial warm-up
 *  + reads from fifo for initial pos of cell
 *  + spits out signalling particles' positions on cell on another fifo
 *  + reads from stdin updated pos
 *
 *  How to use this code
 *
 *  1) Compile 
 *  make BrownianParticle_fifo
 *  2) Create fifos 
 *  mkfifo SignalArrivals
 *  mkfifo CellPosition
 *  3) Kick off code
 *  ./BrownianParticle_fifo -o SignalArrivals -i CellPosition
 *  It expects to read the cell coordinate from the fifo CellPosition, then will write signal arrivals 
 *  into SignalArrivals. 
 *  CellPosition has format x y z
 *  SignalArrivals has format x y z time
 *
 *  To test, kick off the code in one terminal window and in another terminal window do
 *  cat SignalArrivals &
 *  and
 *  cat -u >> CellPosition
 *
 *  The former command will read everything that can be read on SignalArrivals and write
 *  it into the terminal. The latter will read from the terminal and write into CellPosition.
 *  
 *  I tested things by writing 
 *  1 1 1
 *  over and over. The newline "commits the text" to the fifo.
 *
 *
 *
 * 29 July 2021.
 * To be changed:
 *  + Read initial x, y, z and orientation angles from command line.
 *    No: Cell always at origin. Source position specified in command line.
 *     Tell source spherical angles.
 *  + Expect displacement x, y, z relative to cell coo sys on stdin.
 *  + Write spherical coordinates (relative to cell coo system) to stdout.
 *  + Write out "Source found" if source is found.
 *
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

typedef struct {
double x, y, z;
double release_time;
} particle_strct;



/* Distance the cue particle has to diffusive before considered "lost". */
#ifndef CUTOFF
#define CUTOFF (100.)
#endif
double param_cutoff=CUTOFF;
double param_cutoff_squared=CUTOFF*CUTOFF;

#ifndef INITIAL_SOURCEPOS
/* Terminating 0. is a time stamp without meaning. */
#define INITIAL_SOURCEPOS {0., 0., 2., 0.}
#endif
particle_strct source=INITIAL_SOURCEPOS;

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

/* Selfpropulsion velocity */
#ifndef W0
#define W0 (0.1)
#endif
double param_w0=W0;

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

/* Time that needs to pass since the last traj output. */
#define DELTA_TRAJ (0.05)
double next_traj=0.;
double param_delta_snapshot=-1.0;
double next_snapshot=0.;;


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

int param_protocol=0;

gsl_rng *rng;

#define MALLOC(a,n) if ((a=malloc(sizeof(*a)*(n)))==NULL) { fprintf(stderr, "Not enough memory for %s, requested %i bytes, %i items of size %i. %i::%s\n", #a, (int)(sizeof(*a)*n), n, (int)sizeof(*a), errno, strerror(errno)); exit(EXIT_FAILURE); } else { VERBOSE("# Info malloc(3)ed %i bytes (%i items of %i bytes) for %s.\n", (int)(sizeof(*a)*(n)), n, (int)sizeof(*a), #a); }

int verbose=0;
#define VERBOSE if (verbose) printf

int prepare_to_terminate(void);
void postamble(FILE *out);
void update_particles_and_cell(particle_strct d, double scale);
int fprintf_traj(char* format, ...);

/* Each signalling particle has a position relative to the origin.
 * There are at most N signalling particles.
 */

int active_particles, total_particles;
particle_strct *particle;
particle_strct delta, velocity;
double source_distance2, sphere_distance2;
FILE *fin=NULL, *fout=NULL, *traj=NULL;
double tm=0.;
particle_strct cell={0., 0., 0., 0.};

int full_snapshot(void);

int main(int argc, char *argv[])
{
  int ch;



#define STRCPY(dst,src) strncpy(dst,src,sizeof(dst)-1); dst[sizeof(dst)-1]=(char)0


  setlinebuf(stdout);
  while ((ch = getopt(argc, argv, "c:d:h:i:N:o:pR:r:s:S:t:T:vw:")) != -1) {
    switch (ch) {
      case 'c':
	param_cutoff=strtod(optarg, NULL);
	break;
      case 'd':
	param_diffusion=strtod(optarg, NULL);
	break;
      case 'h':
      	param_delta_snapshot=strtod(optarg, NULL);
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
      case 'p':
	param_protocol=1;
	break;
      case 'R':
	param_release_rate=strtod(optarg, NULL);
	break;
      case 'r':
	param_sphere_radius=strtod(optarg,NULL);
	break;
      case 's':
	{
	  char buffer[2048]; // BAD STYLE, buffer overflow should be caught.
	  char *p;

	  strncpy(buffer, optarg, sizeof(buffer)-1);
	  buffer[sizeof(buffer)-1]=(char)0;

	  for (p=buffer; *p; p++) if ((*p==',') || (*p==';')) *p=' ';

	  if (sscanf(buffer, "%lg %lg %lg", &(source.x), &(source.y), &(source.z))!=3) {
	    fprintf(stderr, "# Error: sscanf returned without all three conversions. %i::%s\n", errno, strerror(errno));
	    exit(EXIT_FAILURE);
	  }
	}
	break;
      case 'S':
	param_seed=strtoul(optarg, NULL, 10);
	break;
      case 'T':
	if ((traj=fopen(optarg, "at"))==NULL) {
	  fprintf(stderr, "# Error: Cannot open file %s to store the trajectory.\n", optarg);
	  exit(EXIT_FAILURE);
	}
	break;
      case 't':
	param_delta_t=strtod(optarg, NULL);
	if (param_delta_t<=0.) {
	  fprintf(stderr, "Error: param_delta_t<=0.\n");
	  exit(EXIT_FAILURE);
	}
	break;
      case 'v':
	verbose=1;
	break;
      case 'w':
	param_warmup_time=strtod(optarg, NULL);
	break;
      default:
	printf("# Unknown flag %c.\n", ch);
	exit(EXIT_FAILURE);
	break;
    }
  }



  { 
    int i;

    VERBOSE("# Info: Command: %s", argv[0]);
    fprintf_traj("# Info: Command: %s", argv[0]);
    for (i=1; i<argc; i++) {
      VERBOSE(" \"%s\"", argv[i]);
      fprintf_traj(" \"%s\"", argv[i]);
    }
    VERBOSE("\n");
    fprintf_traj("\n");
  }

  VERBOSE("# Info: Version according to Gunnar: %s\n", VERSION_ACCORDING_TO_GUNNAR);
  fprintf_traj("# Info: Version according to Gunnar: %s\n", VERSION_ACCORDING_TO_GUNNAR);

  /* Some infos. */
  {
    time_t tim;
    tim=time(NULL);

    VERBOSE("# Info Starting at %s", ctime(&tim));
    fprintf_traj("# Info Starting at %s", ctime(&tim));
  }

  /* For version control if present. */
  VERBOSE("# Info: Version of git_version_string to follow.\n");
  VERBOSE("%s", git_version_string);
  VERBOSE("# $Header$\n");
  fprintf_traj("# Info: Version of git_version_string to follow.\n");
  fprintf_traj("%s", git_version_string);
  fprintf_traj("# $Header$\n");


  /* Hostname */
  { char hostname[128];
    gethostname(hostname, sizeof(hostname)-1);
    hostname[sizeof(hostname)-1]=(char)0;
    VERBOSE("# Info: Hostname: %s\n", hostname);
    fprintf_traj("# Info: Hostname: %s\n", hostname);
  }


  /* Dirname */
  { char cwd[1024];
    cwd[0]=0;
    if(getcwd(cwd, sizeof(cwd)-1)!=NULL){
      cwd[sizeof(cwd)-1]=(char)0;
      VERBOSE("# Info: Directory: %s\n", cwd);
      fprintf_traj("# Info: Directory: %s\n", cwd);
    }
  }

  /* Process ID. */
  VERBOSE("# Info: PID: %i\n", (int)getpid());
  fprintf_traj("# Info: PID: %i\n", (int)getpid());

#define PRINT_PARAM(a,o, f) VERBOSE("# Info: %s: %s " f "\n", #a, o, a); fprintf_traj("# Info: %s: %s " f "\n", #a, o, a)

  param_sigma=sqrt(2.*param_delta_t*param_diffusion);
  param_cutoff_squared=param_cutoff*param_cutoff;
  param_sphere_radius_squared=param_sphere_radius*param_sphere_radius;
  PRINT_PARAM(param_protocol, "-p", "%i");
  PRINT_PARAM(param_delta_t, "-t", "%g");
  PRINT_PARAM(param_diffusion, "-d", "%g");
  PRINT_PARAM(param_sigma, "", "%g");
  PRINT_PARAM(param_cutoff, "-c", "%g");
  PRINT_PARAM(param_cutoff_squared, "", "%g");
  PRINT_PARAM(param_sphere_radius, "-r", "%g");
  PRINT_PARAM(param_sphere_radius_squared, "", "%g");
  PRINT_PARAM(param_release_rate, "-R", "%g");
  PRINT_PARAM(param_warmup_time, "-w", "%g");
  PRINT_PARAM(param_max_particles, "-N", "%i");
  PRINT_PARAM(param_seed, "-S", "%lu");
  PRINT_PARAM(param_input, "-i", "%s");
  PRINT_PARAM(param_output, "-o", "%s");

  VERBOSE("# Info: source: -s: %g %g %g\n", source.x, source.y, source.z);
  fprintf_traj("# Info: source: -s: %g %g %g\n", source.x, source.y, source.z);

  next_snapshot=param_warmup_time;



  if (param_output[0]) {
    if ((fout=fopen(param_output, "wt"))==NULL) {
      fprintf(stderr, "Cannot open file %s for writing. %i::%s\n", param_output, errno, strerror(errno));
      exit(EXIT_FAILURE);
    }
    setlinebuf(fout);
  } else fout=stdout;
  VERBOSE("# Info: Output open.\n");

  if (param_input[0]) {
    if ((fin=fopen(param_input, "rt"))==NULL) {
      fprintf(stderr, "Cannot open file %s for reading. %i::%s\n", param_input, errno, strerror(errno));
      exit(EXIT_FAILURE);
    }
  } else fin=stdin;
  VERBOSE("# Info: Input open.\n");

  rng=gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set(rng, param_seed); 
  //printf("%lu",param_seed);


  MALLOC(particle, param_max_particles);
  active_particles=0;
  total_particles=0;

#define CREATE_NEW_PARTICLE { particle[active_particles].x=source.x; particle[active_particles].y=source.y; particle[active_particles].z=source.z; \
  particle[active_particles].release_time=tm; active_particles++; total_particles++;\
  VERBOSE("# Info: New particle created at time %g. Active: %i, Max: %i, Total: %i\n", tm, active_particles, param_max_particles, total_particles);}


  CREATE_NEW_PARTICLE;

  /*
     if (fscanf(fin, "%lg %lg %lg", &(cell.x), &(cell.y), &(cell.z))!=3) {
     fprintf(stderr, "Error: fscanf returned without all three conversions. %i::%s\n", errno, strerror(errno));
     exit(EXIT_FAILURE);
     }
     VERBOSE("# Info: Initial cell position %g %g %g\n", (cell.x), (cell.y), (cell.z));
     */
  fprintf_traj("# Info: START OF TRACK.\n");
  fprintf_traj("0. %g %g %g\n", cell.x, cell.y, cell.z);

  for (tm=0.; ;tm+=param_delta_t) {
    int i;

    if ((tm>=next_snapshot) && (param_delta_snapshot>=0.)) {
      full_snapshot();
      next_snapshot+=param_delta_snapshot;
    }


    if ((velocity.x!=0.) || (velocity.y!=0.) || (velocity.z!=0.))
      update_particles_and_cell(velocity, DELTA_T);

    if (param_release_rate*param_delta_t>gsl_ran_flat(rng, 0., 1.)) {
      if (active_particles>=param_max_particles) {
	fprintf(stderr, "Warning: particle creation suppressed because active_particles=%i >= param_max_particles=%i.\n", active_particles, param_max_particles);
      } else {
	CREATE_NEW_PARTICLE;
      }
    }

    //warning "Using i here as some sort of global object is poor style. The variable i is really one that is too frequently used..."
    for (i=0; i<active_particles; i++) {
      particle[i].x+=gsl_ran_gaussian_ziggurat(rng, param_sigma);
      particle[i].y+=gsl_ran_gaussian_ziggurat(rng, param_sigma);
      particle[i].z+=gsl_ran_gaussian_ziggurat(rng, param_sigma);
      source_distance2 = 
	(particle[i].x-source.x)*(particle[i].x-source.x) 
	+ (particle[i].y-source.y)*(particle[i].y-source.y) 
	+ (particle[i].z-source.z)*(particle[i].z-source.z);


      if (source_distance2>param_cutoff_squared) {
	VERBOSE("# Info: Loss. Particle %i of %i actives (max %i total generated %i) got lost to position %g %g %g at time %g at distance %g>%g having started at time %g.\n", 
	    i, active_particles, param_max_particles, total_particles,
	    particle[i].x, particle[i].y, particle[i].z, tm, sqrt(source_distance2), param_cutoff, particle[i].release_time);
	active_particles--;
	particle[i]=particle[active_particles];
	/* This is a brutal way of dealing with active_particles-1, which has just been copied into i, to remove i: */
	i--;
	continue;
      }

      sphere_distance2=particle[i].x*particle[i].x + particle[i].y*particle[i].y + particle[i].z*particle[i].z;
      if (sphere_distance2<param_sphere_radius_squared) {
	VERBOSE("# Info: Arrival. Particle %i of %i actives (max %i total generated %i) arrived at the cell at position %g %g %g at time %g at distance %g<%g having started at time %g.\n", 
	    i, active_particles, param_max_particles, total_particles,
	    particle[i].x, particle[i].y, particle[i].z, tm, sqrt(sphere_distance2), param_sphere_radius, particle[i].release_time);
	if (tm>param_warmup_time) {
	  double theta, phi;

	  //fprintf(fout, "%g %g %g %g\n", particle[i].x, particle[i].y, particle[i].z, tm);
	  // Coordinates are relative to cell, as the cell is at the origin
	  //range theta from 0 to pi
	  // range phi from 0 to 2pi
	  // -- > spherical coord conventions

	  /* Super simple algorithm: */
	  //velocity=particle[i];
	  { double plen=sqrt(particle[i].x*particle[i].x+particle[i].y*particle[i].y+particle[i].z*particle[i].z);
	  velocity.x=param_w0*particle[i].x/plen;
	  velocity.y=param_w0*particle[i].y/plen;
	  velocity.z=param_w0*particle[i].z/plen;
	  }
	  fprintf_traj("# New velocity %g %g %g\n", velocity.x, velocity.y, velocity.z);
	  fprintf(stderr, "# New velocity %g %g %g %g\n", velocity.x, velocity.y, velocity.z, sqrt(source.x*source.x + source.y*source.y + source.z*source.z));

#if (0)
	  if (particle[i].z!=0.) {
	    theta=atan(sqrt(particle[i].x*particle[i].x + particle[i].y*particle[i].y)/particle[i].z); 
	    if(theta <0.) theta+=M_PI;
	  } else theta=M_PI/2.; // 0 to pi
	  phi=atan2(particle[i].y,particle[i].x); /* phi=0 for y=0 */ // this makes phi from -pi to pi, transform to 0 to 2pi
	  if (particle[i].y<0.) phi= 2*M_PI + phi;


	  // The cell is at the origin
	  // The cue particles have certain coordinates

	  /* This is how we communicated with the Python code. */
	  //fprintf(fout, "%g %g %g\n", theta, phi, tm);
	  fprintf_traj("# CUE %g %g %g\n", theta, phi, tm);

	  /* It looks like when I terminate the fscanf string by a \n then it tries to gobble as much whitespace as possible, so it waits until no-whitespace? */
	  if (param_protocol==0) {
	    if (fscanf(fin, "%lg %lg %lg", &(delta.x), &(delta.y), &(delta.z))!=3) {
	      fprintf(stderr, "# Error: fscanf returned without all three conversions. %i::%s\n", errno, strerror(errno));
	      fprintf_traj("# Error: Garbled message at basic protocol.\n");
	      exit(EXIT_FAILURE);
	      //printf("# Warning: fscanf returned without all three conversions. %i::%s\n", errno, strerror(errno));
	      //printf("# Warning: Exiting quietly.\n");
	      //exit(EXIT_SUCCESS);
	    } else {
	      velocity.x=0.;
	      velocity.y=0.;
	      velocity.z=0.;
	      fprintf_traj("# DELTA Displacement via basic protocol line %i: %g %g %g\n", __LINE__, delta.x , delta.y, delta.z);
	    }
	  } else {
	    char buffer[2048]; // BAD STYLE, buffer overflow should be caught.
	    char *p=buffer;

	    read(STDIN_FILENO, p, 1);
	    while (*p!='\n') {
	      p++;
	      read(STDIN_FILENO, p, 1);
	    }
	    *p=(char)0;
	    fprintf_traj("# DELTA message: [%s]\n", buffer);
	    if (sscanf(buffer, "%lg %lg %lg", &(delta.x), &(delta.y), &(delta.z))!=3) {
	      if (buffer[0]=='V') {
		if (sscanf(buffer+1, "%lg %lg %lg", &(velocity.x), &(velocity.y), &(velocity.z))!=3) {
	          fprintf_traj("# Error: Garbled message at proper protocol.\n");
		  fprintf(stderr, "# Error: sscanf of [%s] returned without all three conversions for velocity. %i::%s\n", buffer, errno, strerror(errno));
		  exit(EXIT_FAILURE);
		} else {
		  delta.x=0.;
		  delta.y=0.;
		  delta.z=0.;
	          fprintf_traj("# DELTA velocity via proper protocol line %i: %g %g %g\n", __LINE__, velocity.x , velocity.y, velocity.z);
		}
	      } else {
		if (strcmp(buffer, "STOP")==0) {
		  VERBOSE("# Info: STOP keyword received. Good bye!\n");
		  fprintf_traj("# STOP received.\n");
		  exit(EXIT_SUCCESS);
		}
		fprintf(stderr, "# Error: sscanf of [%s] returned without all three conversions. %i::%s\n", buffer, errno, strerror(errno));
		exit(EXIT_FAILURE);
	      }
	    } else {
	      velocity.x=0.;
	      velocity.y=0.;
	      velocity.z=0.;
	      fprintf_traj("# DELTA Displacement via proper protocol line %i: %g %g %g\n", __LINE__, delta.x , delta.y, delta.z);
	    }
	  }

	  if ((delta.x!=0.) || (delta.y!=0.) || (delta.z!=0.))
	    update_particles_and_cell(delta, 1.);
#endif

	}
	active_particles--;
	particle[i]=particle[active_particles];
	i--;
	continue;

      }
    }
  }


  if (verbose) postamble(stdout);
  if (traj) postamble(traj);
  return(0);
}






void postamble(FILE *out)
{
  time_t tm;
  struct rusage rus;

  tm=time(NULL);

  fprintf(out, "# Info Terminating at %s", ctime(&tm));
  if (getrusage(RUSAGE_SELF, &rus)) {
    fprintf(out, "# Info getrusage(2) failed.\n");
  } else {
    fprintf(out, "# Info getrusage.ru_utime: %li.%06li\n", (long int)rus.ru_utime.tv_sec, (long int)rus.ru_utime.tv_usec);
    fprintf(out, "# Info getrusage.ru_stime: %li.%06li\n", (long int)rus.ru_stime.tv_sec, (long int)rus.ru_stime.tv_usec);

#define GETRUSAGE_long(f) fprintf(out, "# Info getrusage.%s: %li\n", #f, rus.f);
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
  fprintf(out, "# Info: Good bye and thanks for all the fish.\n");
}


void update_particles_and_cell(particle_strct d, double scale)
{

  d.x*=scale;
  d.y*=scale;
  d.z*=scale;

  /* Update the position of all particles and the source.
   * One update is superfluous, as particle[i] will be purged
   * anyway. */

  /* This looks like a mess, but when you calculate the distance between 
   * cell and particle, this type of subtraction has to happen anyway. */
  { int j;
    for (j=0; j<active_particles; j++) {
      particle[j].x-=d.x;
      particle[j].y-=d.y;
      particle[j].z-=d.z;
    }
  }
  source.x-=d.x;
  source.y-=d.y;
  source.z-=d.z;

cell.x+=d.x;
cell.y+=d.y;
cell.z+=d.z;
if (traj) 
  if (tm>next_traj) {
    fprintf(traj, "%g %g %g %g\n", tm, cell.x, cell.y, cell.z);
    next_traj+=DELTA_TRAJ;
  }

  VERBOSE("# Info: New source position %g %g %g new velocity %g %g %g\n", (source.x), (source.y), (source.z), velocity.x, velocity.y, velocity.z);
  /* The source is found if it resides within the cell. */
  source_distance2=source.x*source.x + source.y*source.y + source.z*source.z;
  if (source_distance2<param_sphere_radius_squared) {
    fprintf(fout, "HEUREKA!\n");
    fprintf_traj("# HEUREKA source_distance2=%g<param_sphere_radius_squared=%g\n", source_distance2, param_sphere_radius_squared);
    VERBOSE("# Info: HEUREKA!\n");
    VERBOSE("# Info: source_distance2=%g<param_sphere_radius_squared=%g\n", source_distance2, param_sphere_radius_squared);
    VERBOSE("# Info: Expecting SIGHUP.\n");
    if (fout) fflush(fout);
    else if (fout!=stdout) fflush(stdout);
    if (traj) fflush(traj); 
    prepare_to_terminate();
  }
  if (source_distance2>param_cutoff_squared) {
//#warning "Unmitigated disaster."
	  fprintf(fout, "Left range\n");
          fprintf_traj("# LEFT source_distance2=%g>param_cutoff_squared=%g\n", source_distance2, param_cutoff_squared);
    	  if (fout) fflush(fout);
    	  else if (fout!=stdout) fflush(stdout);
    	  if (traj) fflush(traj); 
	  prepare_to_terminate();
	}

}


int prepare_to_terminate(void) 
{
char buffer[2048]; // BAD STYLE, buffer overflow should be caught.

if (param_protocol) {
  char *p=buffer;

  read(STDIN_FILENO, p, 1);
  while (*p!='\n') {
    p++;
    read(STDIN_FILENO, p, 1);
  }
  *p=(char)0;
  if (strcmp(buffer, "STOP")==0) {
    VERBOSE("# Info: STOP keyword received. Good bye!\n");
    fprintf_traj("# STOP received.\n");
    exit(EXIT_SUCCESS);
  } else {
    VERBOSE("# Info: Unrecognised instruction received. Good bye!\n");
    fprintf_traj("# Unrecognised instruction received.\n");
    exit(EXIT_FAILURE);
  }
}
scanf("%s", buffer);
exit(EXIT_SUCCESS);
}

int fprintf_traj(char* format, ...)
{
va_list args;
va_start(args, format);
if (traj) return(vfprintf(traj, format, args));
return(0);
}



/* Superficially, 
 * source coo of the source
 * cell coo of the cell
 * particle[i] coo of the particle
 *
 * However, initially the cell is placed at the origin, the source somewhere like 0,0,4 and the particles diffuse from there.
 * This is useful because distances between anything and the cell is just the distance from the origin.
 * As the cell moves, it actually stays at the origin and all the other coos get shifted.
 * The total shift is accumulated in the variable cell. 
 * So, if you want to picture things with a stationary source, then all coos need to be shifted by cell.
 * I also print the source coo, as a matter of debugging.
 */

int full_snapshot(void)
{
int i;
static int count;
FILE *f;
char filename[PATH_MAX+1];

if (count>1000) return(-2);
printf("# Info: Writing file %i at time %g\n", count, tm);
snprintf(filename, PATH_MAX, "BrownianParticle_stndln_snapshot%05i.txt", count);
filename[PATH_MAX]=(char)0;

if ((f=fopen(filename, "wt"))==NULL) return(-1);
fprintf(f, "%g %g %g %i %g\n", cell.x, cell.y, cell.z, active_particles, tm);
fprintf(f, "%g %g %g\n", source.x+cell.x, source.y+cell.y, source.z+cell.z);
for (i=0; i<active_particles; i++) {
  fprintf(f, "%g %g %g\n", particle[i].x+cell.x, particle[i].y+cell.y, particle[i].z+cell.z);
}
fclose(f);
return(count++);
}
