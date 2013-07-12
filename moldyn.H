/* Copyright (c) 1995 by Mark Hill, James Larus, Joel Saltz, andfor 
 * the Wisconsin Wind Tunnel Project and Joel Saltz for the High Performance 
 * Software Laboratory, University of Maryland, College Park.
 *
 * ALL RIGHTS RESERVED.
 *
 * This software is furnished under a license and may be used and
 * copied only in accordance with the terms of such license and the
 * inclusion of the above copyright notice.  This software or any other
 * copies thereof or any derivative works may not be provided or
 * otherwise made available to any other persons.  Title to and ownership
 * of the software is retained by Mark Hill, James Larus, Joel Saltz, and
 * David Wood.  Any use of this software must include the above copyright notice.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS".  THE LICENSOR MAKES NO
 * WARRANTIES ABOUT ITS CORRECTNESS OR PERFORMANCE.
 *
 * Shamik Sharma generated the sequential version in C. 
 * Shubhendu Mukherjee (Shubu) parallelized moldyn on Tempest. 
 */

/*
|| Joint project between the Wisconsin Wind Tunnel at the University of Wisconsin, 
|| Madison and Joel Saltz's research group at the University of Maryland, College 
|| Park.  
||
|| Participants: Shubu Mukherjee (*) and Shamik Sharma (!).
||               Mark Hill (*), Jim Larus(*), Anne Rogers (%), Joel Saltz (!).
||		 
||		 (*) University of Wisconsin-Madison
||		 (!) University of Maryland, College Park
||		 (%) Princeton University
*/

#ifndef _moldyn_h
#define _moldyn_h

ENV

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "malloc.h"
#include "lock.h"
#include "int_macros.h"
#include "tppi_thread.h"
#include "tppi_msg.h"
#include "util_inlines.h"
#include "parmacs.h"
#include "barrier.h"
#include "fatal_error.h"
#include "assert.h"

#ifdef FIRST_COORD_BISEC
#define COORD_BISEC
#endif

#define LOCAL       /* Such a function that changes no global vars */
#define INPARAMS    /* All parameters following this are 'in' params  */

#define SQRT(a)  sqrt(a)
#define POW(a,b) pow(a,b)
#define SQR(a)   ((a)*(a))
#define DRAND(a)  drand_x(a)

#define PAGE_SIZE	TPPI_PAGE_SIZE
#define MAX_PROC	32

#ifndef DIV_CEIL
#define DIV_CEIL(n, d)	((n + d - 1) / d)
#endif

#ifdef DEFAULT_PARAMETERS
/*
!======================  DATA-SETS  ======================================
*/

# ifdef  SMALL
#      define BOXSIZE                 4    /* creates 256 molecules */
#      define NUMBER_TIMESTEPS       30     
#      define MAXINTERACT         32000    /* size of interaction array */
# elif defined(MEDIUM)
#      define BOXSIZE                 8 
#      define NUMBER_TIMESTEPS       30 
#      define MAXINTERACT        320000 
# else 
#      define BOXSIZE                13 
#      define NUMBER_TIMESTEPS       30 
#      define MAXINTERACT       1600000 
# endif     
#endif DEFAULT_PARAMETERS

EXTERN int boxsize, number_timesteps, maxinteract, dimsize, num_particles;

#define DENSITY            0.83134 
#define TEMPERATURE        0.722
#define CUTOFF             3.5000
#define DEFAULT_TIMESTEP   0.064
#define SCALE_TIMESTEP     4
#define TOLERANCE          1.2  

#define DSIZE   	2

#define IND(aa,bb)   	((aa)*dimsize + (bb))    /* used to index x,f,vh */ 


/*
!======================  GLOBAL ARRAYS ======================================
!
! Note : inter is usually the biggest array. If BOXSIZE = 13, it will
!        cause 1 million interactions to be generated. This will need
!        a minimum of 80 MBytes to hold 'inter'. The other
!        arrays will need about a sum of around 800 KBytes. Note
!        that MAXINTERACT may be defined to a more safe value causing
!        extra memory to be allocated. (~ 130 MBytes !)
!============================================================================
*/

EXTERN struct val_t
{
    double x;
    double y;
    double z;
    
#ifndef PADDING64
#ifdef PADDING32
    double pad32;
#endif PADDING32
#endif 
    
    /* for even data distribution across processors for the medium input set */
#ifdef PADDING64
    double pad64[5];
#endif PADDING64
} *coord, *vel;

#define SIZE_OF_COORD		(sizeof(struct val_t))

EXTERN int *coord_indirection_array;

EXTERN struct force_val_t
{
    double x;
    double y;
    double z;
    
#ifdef PADDING32
    double pad32;
#endif 
    /* for even data distribution across processors for the medium input set */
} *force, *force_local; 

#define MAX_PAIRS	MAX_PROC

EXTERN struct remote_pair_t
{
    int num;
    int pairs[MAX_PAIRS];
} *remote_pairs;

void init_remote_pairs();

EXTERN struct inter_t
{
    int left;
    int right;
#ifdef PADDING32_INTER
    int pad[6];
#endif PADDING32_INTER
} *inter, *inter_local; /* pairs of interacting molecules     */ 

int local_num_interact;
int *tot_interact;

int low_node[MAX_PROC];
int high_node[MAX_PROC];

int low_interaction;
int high_local_interaction;
int high_interaction;

/*
   !======================  GLOBAL VARIABLES ===================================
   */

EXTERN double   side;             /*  length of side of box                 */ 
EXTERN double   sideHalf;         /*  1/2 of side                           */
EXTERN double   cutoffRadius;     /*  cuttoff distance for interactions     */
EXTERN int      neighUpdate ;      /*  timesteps between interaction updates */
EXTERN double   perturb;          /*  perturbs initial coordinates          */ 

EXTERN double   timeStep;         /*  length of each timestep   */
EXTERN double   timeStepSq;       /*  square of timestep        */
EXTERN double   timeStepSqHalf;   /*  1/2 of square of timestep */

EXTERN int      numMoles;         /*  number of molecules                   */
EXTERN double   vaver;            /*                                        */

EXTERN double   *epot;            /*  The potential energy      */
EXTERN double   *vir;             /*  The virial  energy        */

#define MAX_PARTICLES 10000

EXTERN struct global_t
{
    int num_interact;
    LOCKDEC(num_interact_lock)
      
#ifndef NO_LOCKING
      ALOCKDEC(force_lock, MAX_PARTICLES)
#endif  NO_LOCKING
	
} *global;

EXTERN int low_mol;
EXTERN int high_mol;
EXTERN int *inter_index;

#define abs(a)  (((a) > 0) ? (a) : -(a))

/* Function prototypes */
char *global_malloc_sbrk(int);
char *global_malloc(int);
char *local_malloc(int);
char *local_malloc_z(int);
int low_index(int, int);
int high_index(int, int);
int get_inter_index();
void iterate();
void UpdateCoordinates(struct val_t *, struct val_t *, struct force_val_t *, 
		       double, int);

void BuildNeigh(struct inter_t *, struct inter_t *, int, struct val_t *,
		double, double);

void BuildNeighOne(struct inter_t *, int, struct val_t *, double, double);
void UpdateVelocities(struct force_val_t *, struct val_t *, int, double);

#ifdef BULK_REDUCTION
void SetupChannels(int);
#endif

void ReduceForces(struct force_val_t *, struct force_val_t *, int);
void ComputeForces(struct force_val_t *, double *, double *, int, struct val_t *,
		   double, double, int, struct inter_t *, int);
void ComputeForcesOne(struct force_val_t *, double *, double *, int, struct val_t *,
		      double, double, int, struct inter_t *);

void align_pages();
void InitSettings();
void InitCoordinates(struct val_t *, int, int, double);
void InitVelocities(struct val_t *, int, double);
void InitForces(struct force_val_t *, int);

void ComputeKE(double *, int, struct val_t *, double);
void ComputeAvgVel(double *, double *, int, struct val_t *, double, double);
double drand_x(int);
double Foo(double, double, double, double, double, double, double, double);
int equal(double, double, double);
void verification_failure(char *);
void verify(int, double, double, double, double, double, double, double);
void PrintResults(int, double, double *, double *, double, double, int, int);
void PrintConnectivity(char *, int *);
void dump_values(char *);
void print_initial_message();
void local_init();

#ifdef USE_NOACK_UPDATES
void compute_num_updates(char *);
#endif

#ifdef DUMP_MESH_STATS
#define MAX_NODES       10000

EXTERN struct node_list_t
{
    int num;
    int nodes[MAX_NODES];
} *remote_nodes;

void dump_mesh_stats(int, struct inter_t *, int *);
#endif

#ifdef COORD_BISEC
void coordinate_bisection(int, struct val_t *, struct val_t *, 
			  struct force_val_t *);
#endif

#ifdef DEBUG
#define print_debug(s) \
{ \
    printf("%2d: Cleared %s\n", TPPI_self_address, s); \
    fflush(stdout); \
}
#else
#define print_debug(s)
#endif DEBUG

#endif _moldyn_h
