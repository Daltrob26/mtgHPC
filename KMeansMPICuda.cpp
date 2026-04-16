#include <stdio.h>
#include <mpi.h>     /* For MPI functions, etc */ 


void launch_cuda();
  
int main(void) {
   int        comm_sz;               /* Number of processes    */
   int        my_rank;               /* My process rank        */

   /* Start up MPI */
   MPI_Init(NULL, NULL); 

   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

   /* Print my message */
   printf("Greetings from process %d of %d!\n", my_rank, comm_sz);

   launch_cuda(); 

   /* Shut down MPI */
   MPI_Finalize(); 

   return 0;
}  /* main */
