#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <sys/resource.h>
//#include "../Gradient_Sensing_Cell_ML_git_stamps.h"
//#define VERSION_ACCORDING_TO_GUNNAR "MAGIC_VATG 20220306_123227"
#include "BrownianParticle_magic.h"


/*
 * 3 Oct 2022
 * Implemented tags for Johannes. They are now featuring in the output
 * fprintf(f, "%i %g %g %g %g\n", particle[ai].tag, particle[ai].release_tm, particle[ai].x+cell.x, particle[ai].y+cell.y, particle[ai].z+cell.z);
 * tag release_tm x y z
 * 
 * The current material still needs validating.
 * I have also implemented HACK_2D. That turns off any diffusion of cue
 * molecukes in the z-direction. If the source is displaced only wrt x-y,
 * this should result in a 2D simulation. It's actually only a single line.
 *
 *
 *
 * 27 Sep 2022
 * Implementing adaptive boost. It looks like that means that the cell
 * misses the source more often and there are more particles in the system.
 * The same happens when I crank up DELTA_T as a whole.
 * I suppose the number of particles in the system is simply a matter of
 * equilibration. When I have coarser DELTA_T or the boost, the simulation
 * runs for longer times.
 *  p 'tmp_noboost.txt_times', 'tmp_boost.txt_times', 'tmp_noboost_fast.txt_times'
 *
 * One way to validate things is to measure the average radial distance 
 * of the cue particles from the source in both schemes (adaptive and non-adative).
 * They should show a similar time dependence and at least for a small cell, 
 * there should be a good theoretical expectation for that.
 *
 * As a validation, I calculate the first two moments of the distance of the cue 
 * particles every param_delta_moments_tm.
 *
 * I will also record the mean time between cue particle arrivals. I can
 * set the drift to 0.0001 or so and see whether that changes significantly 
 * with boost.
 *
 * I wonder now whether the boost was too high. I am assuming that the time
 * for a cue particle to reach the cell is 
 * (cue_particle_distance^2 - cell_radius^2)/diffusion_constant
 *
 * It might be something more like
 * (cue_particle_distance^2 - cell_radius^2)/(6.*diffusion_constant)
 * and the distribution has a long tail, pushing the reasonable min
 * time to very high values. 
 *
 * Hm, even the naive Gaussian gives 2D\Delta_t in every spatial dimension, 
 * so I should be using a factor 6. Also, I should use 
 * (1) (\sqrt{distance_to_cell^2} - cell_radius)^2
 * rather than
 * (2) distance_to_cell^2 - cell_radius^2
 *
 * Is it clear that (1) < (2)? So, show that (a-b)^2 < a^2 - b^2 for a>b>0.
 *
 * (a-b)^2 < a^2 <=> -2ab + b^2 < 0 <=> b^2 < 2ab <=> b<2a
 * using b>0 in the last step.
 *
 * So, yes, (1) is smaller than (2), so I should be using that, as it gives a 
 * smaller boost. Hm. Doesn't make a hell of a difference. The factor 6 does!
 * p [90000:] 'distance_noboost.txt_interarrival' u 1:4 w l, 'distance_boost.txt_interarrival' u 1:4 w l, 'distance_boost6.txt_interarrival' u 1:4 w l, 'distance_boost6BUGGY.txt_interarrival' u 1:4 w l, 'distance_boost6BUGGIER.txt_interarrival' u 1:4 w l
 *
 *
 * Hang on. There is still a significant difference between the number of actives
 * in the boosted and the unboosted. Also, there are very few cue particles compared
 * to earlier simulations. Finally, the moments suggest 0th moments of thousands.
 * What's wrong here?
 *
 * Oh, gosh, I have been using index i multiple times. Sigh. I need to run the 
 * noboost and boost comparising again.
 * 
 * RUN04
 * Now the data compares very well. I think there is a bit of a hicup early
 * in noboost, otherwise things seem to be working very well. 
 * In earlier tests W0 was set to near 0, so that obviously that will have
 * had an effect. 
 * On the whole: I think my boosting earlier was affected by having 1/D rather
 * than 1/(6D) and by having a^2-b^2, when it should have been (a-b)^2. I think
 * the subsequent test which look ok (noboost looking similar to boost6) were 
 * strictly broken, because my loops over moments interfered with the loop over
 * particles.
 *
 *
 * 26 Sep 2022
 * Gunnar to implement repeated runs.
 * ./BrownianParticle_stndln -I 100 -w 100 -s 0,0,3 -m 1000. > tmp.txt
 *
 *
 * Older messages.
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
double release_tm;
int tag;
} particle_strct;


/* max time that the cell can wander arround. */
double param_max_tm=1000.;
/* Number of iterations */
long long int param_iterations=1000;
/* State of the particle */
#define CELL_NO_STATE (0)
#define CELL_PLACED (1)
int cell_placed=CELL_PLACED;
#define CELL_MAX_T_EXCEEDED (2)
#define CELL_READY_TO_RECEIVE (3)
#define CELL_ARRIVED_AT_SOURCE (4)
#define CELL_LEFT (5)
#define CELL_MOVING (6)

#define CELL_PLACED_BUT_TRANSPARENT (7)
#define CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD (8)

/*
 * initial state CELL_NO_STATE
 * 
 * Once a cell is placed, at the beginning of the warm-up, it's CELL_PLACED
 * At end of warm-up, it goes from CELL_PLACED to CELL_READY_TO_RECEIVE;
 * When the first cue arrives, it switches to CELL_MOVING.
 * 
 * When it arrives at the source, the state changes to CELL_ARRIVED_AT_SOURCE
 * When it leaves the cutoff, the state changes to CELL_LEFT.
 *
 * As of 30 Jan 2023 I will have a new state, CELL_PLACED_BUT_TRANSPARENT,
 * which is an alternative to CELL_PLACED, with the difference that particles
 * arriving at the surface are not discarded.
 *
 * I could also CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD to not discard any 
 * cue particles upon making the cell ready to receive.
 *
 * We think that by the cell being constantly killing cue particles while
 * waiting for the warm-up to complete or (as I though after our meeting)
 * by discarding particles inside the cell, we bias the cell to have higher
 * contrast.
 *
 *
 *
 * 6 Feb 2023
 * We still get a mismatch between what we expect to see as first
 * velocities and the numerics.
 *
 * We now have probes all over the place, at which we measure occasionally,
 * at every param_delta_probe_tm.
 * particle_strct
 *
typedef struct {
double x, y, z;
double release_tm;
int tag;
} particle_strct;
 */



particle_strct probe[]={
{1.,0.,0.,0.,0},
{2.,0.,0.,0.,0},
{3.,0.,0.,0.,0},
{4.,0.,0.,0.,0},
{5.,0.,0.,0.,0},
{6.,0.,0.,0.,0},
{7.,0.,0.,0.,0},
{8.,0.,0.,0.,0},
{9.,0.,0.,0.,0},
{10.,0.,0.,0.,0},
{15.,0.,0.,0.,0},
{20.,0.,0.,0.,0},
{25.,0.,0.,0.,0},
{30.,0.,0.,0.,0},
{35.,0.,0.,0.,0},
{40.,0.,0.,0.,0},
{45.,0.,0.,0.,0},
{50.,0.,0.,0.,0},
{0.,1.,0.,0.,0},
{0.,2.,0.,0.,0},
{0.,3.,0.,0.,0},
{0.,4.,0.,0.,0},
{0.,5.,0.,0.,0},
{0.,6.,0.,0.,0},
{0.,7.,0.,0.,0},
{0.,8.,0.,0.,0},
{0.,9.,0.,0.,0},
{0.,10.,0.,0.,0},
{0.,15.,0.,0.,0},
{0.,20.,0.,0.,0},
{0.,25.,0.,0.,0},
{0.,30.,0.,0.,0},
{0.,35.,0.,0.,0},
{0.,40.,0.,0.,0},
{0.,45.,0.,0.,0},
{0.,50.,0.,0.,0},
{0.,0.,1.,0.,0},
{0.,0.,2.,0.,0},
{0.,0.,3.,0.,0},
{0.,0.,4.,0.,0},
{0.,0.,5.,0.,0},
{0.,0.,6.,0.,0},
{0.,0.,7.,0.,0},
{0.,0.,8.,0.,0},
{0.,0.,9.,0.,0},
{0.,0.,10.,0.,0},
{0.,0.,15.,0.,0},
{0.,0.,20.,0.,0},
{0.,0.,25.,0.,0},
{0.,0.,30.,0.,0},
{0.,0.,35.,0.,0},
{0.,0.,40.,0.,0},
{0.,0.,45.,0.,0},
{0.,0.,50.,0.,0}};

#define PROBE_BOX_LENGTH (0.5)
double param_delta_probe_tm=5.;
double next_probe=0.;
int num_probes;
double *probe_mom0, *probe_mom1, *probe_mom2;




int state=CELL_NO_STATE;

#define ADAPTIVE_DT (1e-10) /* Prefactor to calculate the boost from the allowed DT */
double param_adaptive_dt=ADAPTIVE_DT;
#define INITIAL_BOOST (1.)

long long int it;

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
double param_delta_snapshot_tm=-1.0;
double next_snapshot=0.;;

#define MAX_MOM_DISTANCE (4)
int mom;
double next_moments_tm;
double param_delta_moments_tm=100.;
double param_moment_window=0.1;
double start_moment_tm;
double mom_distance[MAX_MOM_DISTANCE+1];

double mom_interarrival[2]={0.,0.};;
double last_arrival_tm;


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

double param_warmup_tm=0.;

int param_protocol=0;

gsl_rng *rng;

#define MALLOC(a,n) if ((a=malloc(sizeof(*a)*(n)))==NULL) { fprintf(stderr, "Not enough memory for %s, requested %i bytes, %i items of size %i. %i::%s\n", #a, (int)(sizeof(*a)*n), n, (int)sizeof(*a), errno, strerror(errno)); exit(EXIT_FAILURE); } else { VERBOSE("# Info malloc(3)ed %i bytes (%i items of %i bytes) for %s.\n", (int)(sizeof(*a)*(n)), n, (int)sizeof(*a), #a); }

int verbose=0;
#define VERBOSE if (verbose) printf

int prepare_to_terminate(void);
void postamble(FILE *out);
int update_particles_and_cell(particle_strct d, double scale);
int fprintf_traj(char* format, ...);

/* Each signalling particle has a position relative to the origin.
 * There are at most N signalling particles.
 */

int ai, active_particles, total_particles, left_particles, absorbed_particles, discarded_particles;
particle_strct *particle;
particle_strct delta, velocity;
double source_distance2, sphere_distance2;
FILE *fin=NULL, *fout=NULL, *traj=NULL;
double tm=0., start_tm;
particle_strct cell={0., 0., 0., 0.};

#define MAX_FILECOUNT (10000)
int full_snapshot(void);
int full_probe(void);


int main(int argc, char *argv[])
{
  int ch;
  double boost=INITIAL_BOOST;
  double min_sphere_distance_squared;
  int nudges;
  int total_probe_count=0;

  double mom_boost[2]={0.,0.};
  double next_release=0.; /* First release always right at the beginning. Hm. Na. Corrected below. */





#define STRCPY(dst,src) strncpy(dst,src,sizeof(dst)-1); dst[sizeof(dst)-1]=(char)0


  setlinebuf(stdout);
  while ((ch = getopt(argc, argv, "b:c:d:h:i:I:m:N:o:pP:R:r:s:S:t:T:vw:")) != -1) {
    switch (ch) {
      case 'b':
      	param_adaptive_dt=strtod(optarg, NULL);
	break;
      case 'c':
	param_cutoff=strtod(optarg, NULL);
	break;
      case 'd':
	param_diffusion=strtod(optarg, NULL);
	break;
      case 'h':
      	param_delta_snapshot_tm=strtod(optarg, NULL);
	break;
      case 'i':
	STRCPY(param_input, optarg);
	break;
      case 'I':
      	param_iterations=strtoll(optarg, NULL, 10);
	break;
      case 'm':
        param_max_tm=strtod(optarg, NULL);
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
      case 'P':
        cell_placed=-1;
	if (optarg[0]=='0'+CELL_PLACED) cell_placed=CELL_PLACED;
	else if (optarg[0]=='0'+CELL_PLACED_BUT_TRANSPARENT) cell_placed=CELL_PLACED_BUT_TRANSPARENT;
	else if (optarg[0]=='0'+CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD) cell_placed=CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD;
	if (cell_placed==-1) {
	  fprintf(stderr, "# Error: cell_placed %s not recognised, allowed values %i, %i and %i.\n", optarg, CELL_PLACED, CELL_PLACED_BUT_TRANSPARENT, CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD);
	  exit(EXIT_FAILURE);
	}
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
	param_warmup_tm=strtod(optarg, NULL);
	break;
      default:
	printf("# Unknown flag %c.\n", ch);
	exit(EXIT_FAILURE);
	break;
    }
  }



  { 
    int i;

    printf("# Info: Command: %s", argv[0]);
    fprintf_traj("# Info: Command: %s", argv[0]);
    for (i=1; i<argc; i++) {
      printf(" \"%s\"", argv[i]);
      fprintf_traj(" \"%s\"", argv[i]);
    }
    printf("\n");
    fprintf_traj("\n");
  }

  printf("# Info: Version according to Gunnar: %s\n", VERSION_ACCORDING_TO_GUNNAR);
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
  //VERBOSE("%s", git_version_string);
  VERBOSE("# $Header$\n");
  fprintf_traj("# Info: Version of git_version_string to follow.\n");
  //fprintf_traj("%s", git_version_string);
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

#define PRINT_PARAM(a,o, f) printf("# Info: %s: %s " f "\n", #a, o, a); fprintf_traj("# Info: %s: %s " f "\n", #a, o, a)

  param_sigma=sqrt(2.*param_delta_t*param_diffusion);
  param_cutoff_squared=param_cutoff*param_cutoff;
  param_sphere_radius_squared=param_sphere_radius*param_sphere_radius;
  #ifdef HACK_2D
  printf("# Info: ***** HACK_2D is defined. ***** \n");
  #else
  printf("# Info: HACK_2D is not defined.\n");
  #endif
  PRINT_PARAM(cell_placed, "", "%i");
  printf("# Info: cell_placed=%i compared to CELL_PLACED=%i CELL_PLACED_BUT_TRANSPARENT=%i CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD=%i\n",
  	cell_placed, CELL_PLACED, CELL_PLACED_BUT_TRANSPARENT, CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD);
  PRINT_PARAM(param_protocol, "-p", "%i");
  PRINT_PARAM(param_delta_t, "-t", "%g");
  PRINT_PARAM(param_diffusion, "-d", "%g");
  PRINT_PARAM(param_sigma, "", "%g");
  PRINT_PARAM(param_cutoff, "-c", "%g");
  PRINT_PARAM(param_cutoff_squared, "", "%g");
  PRINT_PARAM(param_sphere_radius, "-r", "%g");
  PRINT_PARAM(param_sphere_radius_squared, "", "%g");
  PRINT_PARAM(param_release_rate, "-R", "%g");
  PRINT_PARAM(param_warmup_tm, "-w", "%g");
  PRINT_PARAM(param_max_tm, "-m", "%g");
  PRINT_PARAM(param_max_particles, "-N", "%i");
  PRINT_PARAM(param_seed, "-S", "%lu");
  PRINT_PARAM(param_input, "-i", "%s");
  PRINT_PARAM(param_output, "-o", "%s");
  PRINT_PARAM(param_protocol, "-p", "%i");
  PRINT_PARAM(param_adaptive_dt, "-b", "%g");
  PRINT_PARAM(INITIAL_BOOST, "", "%g");
  PRINT_PARAM(param_sphere_radius, "", "%g");
  PRINT_PARAM(param_w0, "", "%g");
  PRINT_PARAM(param_delta_snapshot_tm, "-h", "%g");
  PRINT_PARAM(param_delta_probe_tm, "", "%g");
  num_probes=sizeof(probe)/sizeof(*probe);
  PRINT_PARAM(num_probes, "", "%i");
  PRINT_PARAM(PROBE_BOX_LENGTH, "", "%g");
  PRINT_PARAM(param_delta_moments_tm , "", "%g");
  PRINT_PARAM(param_iterations, "-I", "%lli");
  PRINT_PARAM(verbose, "-v", "%i");
  VERBOSE("# Info: source: -s: %g %g %g\n", source.x, source.y, source.z);

  MALLOC(probe_mom0, num_probes);
  MALLOC(probe_mom1, num_probes);
  MALLOC(probe_mom2, num_probes);
  {
  int i;
  for (i=0; i<num_probes; i++) {
    probe_mom0[i]=0.;
    probe_mom1[i]=0.;
    probe_mom2[i]=0.;
  }
  }

  next_moments_tm=0.;
  start_moment_tm=-1.;



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

for (mom=0; mom<=MAX_MOM_DISTANCE; mom++) mom_distance[mom]=0.;
tm=0.;
total_particles=0;
active_particles=0;
left_particles=0;
absorbed_particles=0;
discarded_particles=0;
last_arrival_tm=0.;
next_release=(-log(1.-gsl_ran_flat(rng, 0., 1.))/param_release_rate);
for (it=1LL; it<=param_iterations; it++) {
  
/* This is copied from update_particles_and_cell.
 * The idea is to move the whole coo sys so the cell
 * can start again at the origin. This involves effectively
 * moving all the cue particles and then deleting those
 * that now happen to be inside the cell.
 *
 * It is this sort of thing will need a redesign at some point.
 * */
printf("# Info: Adjusting coordinates of %i particles by %g,%g,%g.\n", active_particles, cell.x, cell.y, cell.z);
{
particle_strct d;

d.x=-cell.x;
d.y=-cell.y;
d.z=-cell.z;
  /* Update the position of all particles and the source.
   * One update is superfluous, as particle[ai] will be purged
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
printf("# Info: At the beginning of %lli there are %i particles in the system. Source at %g %g %g, cell at %g %g %g\n", it, active_particles, source.x, source.y, source.z, cell.x, cell.y, cell.z);
}



  cell.release_tm=tm; /* Just for the time being */
  cell.tag=0;
  state=cell_placed;
  nudges=0;
  // Not allowed: velocity={0.,0.,0.,0.};
  velocity.x=0.;
  velocity.y=0.;
  velocity.z=0.;
  velocity.release_tm=tm;
  velocity.tag=0;
  start_tm=tm; 
  //  active_particles=0;
printf("# Info: Not starting from scratch, but allowing for warmup.\n");
next_snapshot=tm+param_warmup_tm;
next_probe=tm+param_warmup_tm;

/* The tags start at 1. */
#define CREATE_NEW_PARTICLE { particle[active_particles].x=source.x; particle[active_particles].y=source.y; particle[active_particles].z=source.z; \
  total_particles++;\
  particle[active_particles].release_tm=tm; particle[active_particles].tag=total_particles; active_particles++;\
  VERBOSE("# Info: New particle created at time %g. Active: %i, Max: %i, Total: %i\n", tm, active_particles, param_max_particles, total_particles);}


  //CREATE_NEW_PARTICLE;

  /*
     if (fscanf(fin, "%lg %lg %lg", &(cell.x), &(cell.y), &(cell.z))!=3) {
     fprintf(stderr, "Error: fscanf returned without all three conversions. %i::%s\n", errno, strerror(errno));
     exit(EXIT_FAILURE);
     }
     VERBOSE("# Info: Initial cell position %g %g %g\n", (cell.x), (cell.y), (cell.z));
     */
  fprintf_traj("# Info: START OF TRACK.\n");
  fprintf_traj("0. %g %g %g\n", cell.x, cell.y, cell.z);
  boost=INITIAL_BOOST;

  for (; ;tm+=(boost*param_delta_t)) {
    mom_boost[0]++;
    mom_boost[1]+=boost;
    if ((tm>=next_snapshot) && (param_delta_snapshot_tm>=0.)) {
      full_snapshot();
      next_snapshot+=param_delta_snapshot_tm;
    }
    if ((tm>=next_probe) && (param_delta_probe_tm>=0.)) {
      int probe_count;
      probe_count=full_probe();
      total_probe_count+=probe_count;
      printf("# Info: full_probe() at tm=%g found %i, total %i. Active are %i.\n", tm, probe_count, total_probe_count, active_particles);
      next_probe+=param_delta_probe_tm;
    }

    if ((tm-start_tm>param_max_tm) && (param_max_tm>0.)) {
      state=CELL_MAX_T_EXCEEDED;
      printf("# FINISHED %lli %i %g %g %g %g %i %i %i %i %i %i %i\n", it, state, start_tm, tm, cell.release_tm, tm-cell.release_tm, nudges, active_particles, left_particles, absorbed_particles, total_particles, discarded_particles, active_particles+left_particles+absorbed_particles+discarded_particles);
      break;
    }
    if ((tm-start_tm>param_warmup_tm) && (state==cell_placed)) {
      cell.release_tm=tm;
      cell.tag=0;
      if (state==CELL_PLACED_BUT_TRANSPARENT) {
        /* Remove cue particles inside the cell. 
	 * This is not to be done when the state is
	 * CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD */
        for (ai=active_particles-1; ai>=0; ai--) {
          sphere_distance2=particle[ai].x*particle[ai].x + particle[ai].y*particle[ai].y + particle[ai].z*particle[ai].z;
          if (sphere_distance2<param_sphere_radius_squared) {
	    /* Overwrite current particle ai with the one at the end.
	     * Maybe ai==--active_particles, but then 
	     * active_particles is reduced by one.
	     * For example active_particles=15, then ai=14
	     * and I do ai[14]=ai[14] with active_particles now
	     * reduced to 14. */
	    particle[--active_particles]=particle[ai];
	    discarded_particles++;
	  }
	}
        printf("# Info: discarded_particles=%i\n", discarded_particles);

      }
      state=CELL_READY_TO_RECEIVE;
      printf("# Info: Cell released and ready to receive at %g\n", tm);
    }


    if ((velocity.x!=0.) || (velocity.y!=0.) || (velocity.z!=0.)) {
      state=update_particles_and_cell(velocity, boost*param_delta_t);

      if ((state==CELL_ARRIVED_AT_SOURCE) || (state==CELL_LEFT)) {
        printf("# FINISHED %lli %i %g %g %g %g %i %i %i %i %i %i\n", it, state, start_tm, tm, cell.release_tm, tm-cell.release_tm, nudges, active_particles, left_particles, absorbed_particles, total_particles, active_particles+left_particles+absorbed_particles);
      	break; /* Breaks out of time-loop. */
      }
    }

/* Old version of Poissonian realease
    if (param_release_rate*boost*param_delta_t>gsl_ran_flat(rng, 0., 1.)) {
      if (active_particles>=param_max_particles) {
	fprintf(stderr, "Warning: particle creation suppressed because active_particles=%i >= param_max_particles=%i.\n", active_particles, param_max_particles);
      } else {
	CREATE_NEW_PARTICLE;
      }
    }
*/

  while (tm>=next_release) {
      if (active_particles>=param_max_particles) {
	fprintf(stderr, "Warning: particle creation suppressed because active_particles=%i >= param_max_particles=%i.\n", active_particles, param_max_particles);
      } else {
	CREATE_NEW_PARTICLE;
      }
      /* gsl_ran_flat gives 0...0.99999
       * t=-log(1-u)/gamma has distribution gamma exp(-t gamma)
       * if u is uniform.
       * exp(-gamma t)=1-u 
       * so du/dt=gamma exp(-gamma t) and 
       * P(u) du = P(t) dt,
       * so with P(u)=1 
       * P(t) = gamma exp(-gamma t) 
       * as expected.
       */
      next_release+=(-log(1.-gsl_ran_flat(rng, 0., 1.))/param_release_rate);

  }
    //warning "Using i here as some sort of global object is poor style. The variable i is really one that is too frequently used..."



    min_sphere_distance_squared=source.x*source.x + source.y*source.y + source.z*source.z;
    for (ai=0; ai<active_particles; ai++) {
      particle[ai].x+=gsl_ran_gaussian_ziggurat(rng, sqrt(boost)*param_sigma);
      particle[ai].y+=gsl_ran_gaussian_ziggurat(rng, sqrt(boost)*param_sigma);
#ifndef HACK_2D
      particle[ai].z+=gsl_ran_gaussian_ziggurat(rng, sqrt(boost)*param_sigma);
#endif
      source_distance2 = 
	  (particle[ai].x-source.x)*(particle[ai].x-source.x) 
	+ (particle[ai].y-source.y)*(particle[ai].y-source.y) 
	+ (particle[ai].z-source.z)*(particle[ai].z-source.z);
      
      /* Moments are assumed to be initialised. 
       * The 1e-6 is a bit of slag for window size 0 to capture the current time slice. */
      if ((tm>next_moments_tm) && ((tm<=start_moment_tm+param_moment_window+1e-6) || (start_moment_tm<0.))) {
      double p, sd;
      /* take moments */
        if (start_moment_tm<0.) start_moment_tm=tm;
        sd=sqrt(source_distance2);
        for (p=1., mom=0; mom<=MAX_MOM_DISTANCE; mom++) {
	  mom_distance[mom]+=p;
	  p*=sd;
	}
      }


      if (source_distance2>param_cutoff_squared) {
	VERBOSE("# Info: Loss. Particle %i tag %i of %i actives (max %i total generated %i) got lost to position %g %g %g (source at %g %g %g) at time %g at distance %g>%g having started at time %g. (%g-%g)^2+(%g-%g)^2+(%g-%g)^2=%g\n", 
	    ai, particle[ai].tag, active_particles, param_max_particles, total_particles,
	    particle[ai].x, particle[ai].y, particle[ai].z, 
	    source.x, source.y, source.z, 
	    tm, sqrt(source_distance2), param_cutoff, particle[ai].release_tm,
	    particle[ai].x, source.x,
	    particle[ai].y, source.y,
	    particle[ai].x, source.x,
(particle[ai].x-source.x)*(particle[ai].x-source.x)
        + (particle[ai].y-source.y)*(particle[ai].y-source.y)
        + (particle[ai].z-source.z)*(particle[ai].z-source.z)
	    );
	active_particles--;
	left_particles++;
	particle[ai]=particle[active_particles];
	/* This is a brutal way of dealing with active_particles-1, which has just been copied into i, to remove i: */
	ai--;
	continue;
      }

      sphere_distance2=particle[ai].x*particle[ai].x + particle[ai].y*particle[ai].y + particle[ai].z*particle[ai].z;
      if (sphere_distance2<min_sphere_distance_squared) min_sphere_distance_squared=sphere_distance2;
      if ((sphere_distance2<param_sphere_radius_squared) && (state!=CELL_PLACED_BUT_TRANSPARENT) && (state!=CELL_PLACED_BUT_TRANSPARENT_DONT_DISCARD)) {
	VERBOSE("# Info: Arrival. Particle %i of %i actives (max %i total generated %i) arrived at the cell at position %g %g %g at time %g at distance %g<%g having started at time %g.\n", 
	    ai, active_particles, param_max_particles, total_particles,
	    particle[ai].x, particle[ai].y, particle[ai].z, tm, sqrt(sphere_distance2), param_sphere_radius, particle[ai].release_tm);
	mom_interarrival[0]++;
	mom_interarrival[1]+=(tm-last_arrival_tm);
	last_arrival_tm=tm;
	if ((state==CELL_READY_TO_RECEIVE) || (state==CELL_MOVING)) {
	  //double theta, phi;

	  state=CELL_MOVING;
	  nudges++;
	  //fprintf(fout, "%g %g %g %g\n", particle[ai].x, particle[ai].y, particle[ai].z, tm);
	  // Coordinates are relative to cell, as the cell is at the origin
	  //range theta from 0 to pi
	  // range phi from 0 to 2pi
	  // -- > spherical coord conventions

	  /* Super simple algorithm: */
	  //velocity=particle[ai];
	  { double plen=sqrt(particle[ai].x*particle[ai].x+particle[ai].y*particle[ai].y+particle[ai].z*particle[ai].z);
	  velocity.x=param_w0*particle[ai].x/plen;
	  velocity.y=param_w0*particle[ai].y/plen;
	  velocity.z=param_w0*particle[ai].z/plen;
#ifdef HACK_2D
if (velocity.z!=0.) printf("# Error: velocity.z=%g despite HACK_2D\n", velocity.z);
#endif

	  velocity.release_tm=tm;
	  }
	  fprintf_traj("# New velocity %g %g %g\n", velocity.x, velocity.y, velocity.z);
	  if (0)
	    fprintf(stdout, "# Info: New velocity at tm=%g is %g %g %g distance %g particles %i %i boost %g\n", tm, velocity.x, velocity.y, velocity.z, sqrt(source.x*source.x + source.y*source.y + source.z*source.z), active_particles, total_particles, boost);


	}
	active_particles--;
	absorbed_particles++;
	particle[ai]=particle[active_particles];
	ai--;
	continue;

      }
    } /* particles */
    
#warning "The below amounts to writing a long of stuff to stdout. Maybe having an estimated end-time here etc, is CPU time more wisely spent."
    if ((start_moment_tm>0.) && (tm>start_moment_tm+param_moment_window)) {
      printf("# MOM_DISTANCE %g %g %g %i %i %g", tm, start_moment_tm, param_moment_window, MAX_MOM_DISTANCE, active_particles, mom_distance[0]);
      if (mom_distance[0]==0.0) mom_distance[0]=-1.;
      for (mom=1; mom<=MAX_MOM_DISTANCE; mom++) {
        printf(" %g", mom_distance[mom]/mom_distance[0]);
	mom_distance[mom]=0.;
      }
      mom_distance[0]=0.;
      printf("\n");

      printf("# MOM_INTERARRIVAL %g %g %g %i %i %g", tm, start_moment_tm, param_moment_window, 2, active_particles, mom_interarrival[0]);
      if (mom_interarrival[0]>0.) printf(" %g\n", mom_interarrival[1]/mom_interarrival[0]);
      else printf(" %g\n", -mom_interarrival[1]);

      next_moments_tm+=param_delta_moments_tm;
      start_moment_tm=-1.;
    }

#ifdef ADAPTIVE_DT
    {
    double diffusive_dt, ballistic_dt, realease_dt, best_dt;
    double min_sphere_distance;

    #define SQUARE(a) ((a)*(a))

    if (min_sphere_distance_squared>param_sphere_radius_squared) {
      min_sphere_distance=sqrt(min_sphere_distance_squared);
      //diffusive_dt=(min_sphere_distance_squared-param_sphere_radius_squared)/param_diffusion;
      diffusive_dt=SQUARE(min_sphere_distance-param_sphere_radius_squared)/(6.*param_diffusion);
      //diffusive_dt=(min_sphere_distance_squared-param_sphere_radius_squared)/(6.*param_diffusion);
      //diffusive_dt=(min_sphere_distance_squared-param_sphere_radius_squared)/(param_diffusion);
      //ballistic_dt=sqrt(min_sphere_distance_squared-param_sphere_radius_squared)/param_w0;
      ballistic_dt=(min_sphere_distance-param_sphere_radius)/param_w0;
    } else {
      diffusive_dt=ballistic_dt=0.;
    }
    realease_dt=1./param_release_rate;

    
    #define MIN(a,b) (((a)<(b)) ? (a) : (b))
    best_dt=MIN(diffusive_dt, ballistic_dt);
    best_dt=MIN(best_dt, realease_dt);
    best_dt*=param_adaptive_dt;

    if (best_dt>param_delta_t) boost=best_dt/param_delta_t;
    else boost=INITIAL_BOOST;
    //printf("# DTs: %g %g %g, best %g but is %g, so boost %g\n", diffusive_dt, ballistic_dt, realease_dt, best_dt, param_delta_t, boost);
    }
#endif


  } /* time loop */
} /* iterations */


  
  printf("# MOM_BOOST %g %i %g", tm, 2, mom_boost[0]);
  if (mom_boost[0]>0.) printf(" %g\n", mom_boost[1]/mom_boost[0]);
  else printf(" %g\n", -mom_boost[1]);

  { int i;
  printf("# Info: total_probe_count=%i for %i probes box size %g.\n", total_probe_count, num_probes, PROBE_BOX_LENGTH);
  for (i=0; i<num_probes; i++) {
    printf("#PROBE %i %g %g %g %g %g %g\n", i, probe[i].x, probe[i].y, probe[i].z, 
      probe_mom0[i], 
      probe_mom1[i]/( (probe_mom0[i]<=0.) ? (-1.) : probe_mom0[i]),
      probe_mom2[i]/( (probe_mom0[i]<=0.) ? (-1.) : probe_mom0[i]) );
  }
  }
  postamble(stdout);
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


int update_particles_and_cell(particle_strct d, double scale)
{

  d.x*=scale;
  d.y*=scale;
  d.z*=scale;

  /* Update the position of all particles and the source.
   * One update is superfluous, as particle[ai] will be purged
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

/* This is just for keeping track of the whole displacement,
 * so that it can later be added to all particles. */
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
    fprintf(fout, "# Info: HEUREKA!\n");
    fprintf_traj("# HEUREKA source_distance2=%g<param_sphere_radius_squared=%g\n", source_distance2, param_sphere_radius_squared);
    VERBOSE("# Info: HEUREKA!\n");
    VERBOSE("# Info: source_distance2=%g<param_sphere_radius_squared=%g\n", source_distance2, param_sphere_radius_squared);
    VERBOSE("# Info: Expecting SIGHUP.\n");
    if (fout) fflush(fout);
    else if (fout!=stdout) fflush(stdout);
    if (traj) fflush(traj); 
    return(CELL_ARRIVED_AT_SOURCE);
    prepare_to_terminate();
  }
  if (source_distance2>param_cutoff_squared) {
//#warning "Unmitigated disaster."
	  fprintf(fout, "Left range\n");
          fprintf_traj("# LEFT source_distance2=%g>param_cutoff_squared=%g\n", source_distance2, param_cutoff_squared);
    	  if (fout) fflush(fout);
    	  else if (fout!=stdout) fflush(stdout);
    	  if (traj) fflush(traj); 
	  return(CELL_LEFT);
	  //prepare_to_terminate();
	}

return(CELL_MOVING);
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
 * particle[ai] coo of the particle
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
int ai;
static int count;
FILE *f;
char filename[PATH_MAX+1];

if (count>MAX_FILECOUNT) return(-2);
printf("# Info: Writing file %i at time %g\n", count, tm);
snprintf(filename, PATH_MAX, "BrownianParticle_stndln_snapshot%05i.txt", count);
filename[PATH_MAX]=(char)0;

if ((f=fopen(filename, "wt"))==NULL) return(-1);
fprintf(f, "%g %g %g %i %g\n", cell.x, cell.y, cell.z, active_particles, tm);
fprintf(f, "%g %g %g\n", velocity.x, velocity.y, velocity.z);
fprintf(f, "%g %g %g\n", source.x+cell.x, source.y+cell.y, source.z+cell.z);
for (ai=0; ai<active_particles; ai++) {
  fprintf(f, "%i %g %g %g %g\n", particle[ai].tag, particle[ai].release_tm, particle[ai].x+cell.x, particle[ai].y+cell.y, particle[ai].z+cell.z);
}
fclose(f);
return(count++);
}

#define IN_PROBE(a,b) ( (IN_PROBE_COMPO(a,b,x)) && (IN_PROBE_COMPO(a,b,y)) && (IN_PROBE_COMPO(a,b,z)) )
#define IN_PROBE_COMPO(a,b,c) IN_PROBE_RANGE((a.c+source.c)-(b.c+cell.c) )
#define IN_PROBE_RANGE(d) (fabs(d)<PROBE_BOX_LENGTH/2.)

/* The coordinates of everything are relative to the cell.
 * The cell has in principle position (0,0,0), but cell.x,y,z
 * are a record of the total thus far. 
 * The probes are meant to be relative to the source. The source
 * is located at source.x,y,z.
 * In RUN12 I have used the macro
 * #define IN_PROBE_COMPO(a,b,c) IN_PROBE_RANGE(a.c-(b.c+cell.c) )
 * but I really think of the probe coordinate as to be read relative
 * to the source,
 * #define IN_PROBE_COMPO(a,b,c) IN_PROBE_RANGE((a.c+source.c)-(b.c+cell.c) )
 *
 *
 */
int full_probe(void)
{
int i, j;
int count;
int total_count=0;


for (i=0; i<num_probes; i++) {
  for (count=j=0; j<active_particles; j++) {
    /* particle[j]'s position is at
     * particle[j]+cell,
     * as the coordinates that are being maintained are relative 
     * to the cell. The cell is at the origin (but its total discplavcement
     * is kept track of in cell */
    
    if ( IN_PROBE(probe[i],particle[j]) ) count++;
    }
  probe_mom0[i]++;
  probe_mom1[i]+=count;
  probe_mom2[i]+=(count*count);
  total_count+=count;
}


return(total_count);
}
