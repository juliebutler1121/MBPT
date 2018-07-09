/************************************************************************************************
MBPT-Butler-Parallel.cpp
June 21st, 2018

A code which applies second order many-body pertubation theory to the paring problem.  This
paring problem has eight states with four particles, giving a total number of six possible
configurations.  This version of the program utilizes a parallel implementation of the program
whenever feasible.

June 21st, 2018 -- Constructs the 6x6 Hamiltonian and has a method for printing the 
	Hamiltonian
************************************************************************************************/

/************************************************************************************************
Methods:
printHamiltonian (H, n) - takes in a two dimensional array of type Sumbolic and the number of 
	columns/rows and prints the matrix in a nicely formatted array
calculateHamiltonian (&H0): Calcualtes the Hamiltonian for the pairing problem for a system 
	of N=8 and four particles (fermions).
************************************************************************************************/

/************************************************
*                                               *
*                  INCLUDES                     *
*                                               *
************************************************/

// Allows for printing
#include <iostream>
// Allows for the use of MPI methods of parallelization
#include <mpi.h>
// Using the namespace std, prevents having to type the namespace (ex: std::cout --> cout)
using namespace std;



/************************************************
*                                               *
*    Prototypes (definitions given later)       *
*                                               *
************************************************/
void printHamiltonian (double ** & H0, int n);

double** buildHamiltonian (double ** & H0, int size); 


/************************************************
*                                               *
*                   main                        *
*                                               *
************************************************/
int main() {
	/*
	Inputs:
		None.
	Outputs:
		None.
	Runs the program.  Currently only builds the Hamiltonian and the
	prints the formatted Hamiltonian.
	*/

	// Creates an empty array of double objects.  This is just a 
	// placeholder to hold the results of the buildHamiltonian method
	double ** H0 = new double * [6];
	for (int i = 0; i < 6; i ++){
		H0[i] = new double [6];
	}

	// Runs the buildHamiltonian method and stores the returned value
	// in H0
	H0 = buildHamiltonian (H0, 6);
	// Prints the Hamiltonian in a nicely formatted way
	//printHamiltonian (H0, 6);

	// Required returning of an int
	return 0;
} //main


/************************************************
*                                               *
*               printHamiltonian                *
*                                               *
************************************************/
void printHamiltonian (double ** & H0, int n){
	/*
	Input:
		H (a reference to a two dimensional array of type double): 
			A Hamiltonian matrix (MUST be square)
		n (an integer): The dimension of the Hamiltonian matrix,
		    can either be the number of rows of columns since it is 
                    a square matrix.
	Output:
		None.

	Takes the two dimensional array given in the parameters and prints it 
	in a nicely formatted way.
	*/


	// The final, formated string will have n lines.  Each line will have n 
	// elements, each separated by a tab.  For example, for the matrix 
	// [[a, b, c], [d, e, f], [g, h, i]], the formatted, printed output will look like:
	// a	b	c
	// d	e	f
	// g	h	i

	// Cycles through the n rows and n columns of the matrix and creates a formatted string
	// of all the matrix elements.  i cycles through the rows and j cycles through the columns.
	for (int i = 0; i < n; i ++){
		for (int j = 0; j < n; j ++){
			// Prints the current matrix element with a tab afterwards for
			// formatting purposes
			cout << H0[i][j] << "\t";
		}
		// Prints a new line character after each j-cycle so each row of the 
		// matrix prints on its own line
		cout << "\n";
	}
}

/************************************************
*                                               *
*             buildHamiltonian                  *
*                                               *
************************************************/
double ** buildHamiltonian (double ** & H0, int size){
	/*
	Input:
		H0 (a reference to a two dimensional double array): Currently 
			is a square two dimensional array filled with the symbol
			for "NONE".
		size (an integer): The number of rows or columns in the square array, H0.
	Output:
		H0 (a two dimensional double array): the Hamiltonian of the 
		pairing problem.  It will be a 6x6 matrix.

	Calcualtes the Hamiltonian for the pairing problem for a system of N=8 and four particles (fermions).
	Currently d (the level spacing) and g (the interaction strength) are double and do not have specific
	values.
	*/

	// The constant g, which represents the interactions strength
	// between two pairs of particles
	//double g("g");
	// The level spacing
	//double d("d");

	double g = -0.5;
	double d = 1;

	// Energy calculations for the different energy levels.  For example,
	// for the level p = 1, the energy is p*d, or 1*d, which is stored in p1.
	double p0  = 0*d;
	double p1 = d;
	double p2 = 2*d;
	double p3 = 3*d;
	
	// For all states, the two dimensional array gives the p value and the spin
	// of a particular orbital.  The p values refer to which energy level is being
	// occupied:
	// _ _  p = 3
	// _ _  p = 2
	// _ _  p = 1
	// _ _  p = 0
	// Spin (ms) has one of two values: either 1 representing a spin up particle of -1
	// representing a spin down particle

	// Due to the Pauli Exclusion Principle, two particles with the same p value 
	// must have different spins
	
	// Initializes an array of arrays
	double ** states = new double * [8];
	// Initializes each inner array
	for (int i = 0; i < 8; i ++){
		states [i] = new double [2];
	}
	
	// Fills in the values of p and ms for each orbital.  The numbering scheme
	// is as follows:
	// _ _  p = 3  6 7
	// _ _  p = 2  4 5
	// _ _  p = 1  2 3 
	// _ _  p = 0  0 1
	// In the array, the first number (index 0) is the p value and the second 
	// number (index 1) is the ms value
	states [0][0] = 0; states [0][1] = 1; 
	states [1][0] = 0; states [1][1] = -1; 
	states [2][0] = 1; states [2][1] = 1; 
	states [3][0] = 1; states [3][1] = -1; 
	states [4][0] = 2; states [4][1] = 1; 
	states [5][0] = 2; states [5][1] = -1; 
	states [6][0] = 3; states [6][1] = 1; 
	states [7][0] = 3; states [7][1] = -1;

	// Assigns the energies of each orbital to the appropriate index in the array
	// using the numbering scheme shown above.  It has to be an array of double
	// objects because d is a symbol.
	double * energies = new double [8];
	energies [0] = p0; energies [1] = p0; 
	energies [2] = p1; energies [3] = p1;
	energies [4] = p2; energies [5] = p2;
	energies [6] = p3; energies [7] = p3;

	// A list of all possible arrangements of particles
	// Phi0
	// _ _
	// _ _
	// * *
	// * *
	// Phi1
	// _ _
	// * *
	// _ _
	// * *
	// Phi2
	// * *
	// _ _
	// _ _
	// * *
	// Phi3
	// _ _
	// * *
	// * *
	// _ _
	// Phi4
	// * *
	// _ _
	// * *
	// _ _
	// Phi5
	// * *
	// * *
	// _ _
	// _ _
	// The numbers in the inner arrays represent the index of the occupied states from
	// the states array
	double ** basis = new double * [6];
	for (int i = 0; i < 6; i ++){
		basis [i] = new double [4];
	}
	basis [0][0] = 0; basis [0][1] = 1; basis [0][2] = 2; basis [0][3] = 3;
	basis [1][0] = 0; basis [1][1] = 1; basis [1][2] = 4; basis [1][3] = 5;
	basis [2][0] = 0; basis [2][1] = 1; basis [2][2] = 6; basis [2][3] = 7;
	basis [3][0] = 2; basis [3][1] = 3; basis [3][2] = 4; basis [3][3] = 5;
	basis [4][0] = 2; basis [4][1] = 3; basis [4][2] = 6; basis [4][3] = 7;
	basis [5][0] = 4; basis [5][1] = 5; basis [5][2] = 6; basis [5][3] = 7;

	// This will begin the parallel version of the code.

	// Initilizes MPI
	MPI::Init ();

	// The total number of processes that are being used to run the program.
	int num_procs = MPI::COMM_WORLD.Get_size();
	// The number of the process being currently being used.
	int rank = MPI::COMM_WORLD.Get_rank ();

	// Fills the Hamiltonian with the appropriate values. The values are
	// found by evaulating the expression <PhiI|HI|PhiJ>.   However, the
	// program does not directly analyze the expression.  The values for
	// all possible combinations of PhiI and PhiJ are known, so these
	// values are directly assigned to the appropriate matrix elements.

	// i and j cycle through all of the basis states and compares 
	// them.  There are a total of 36 comparisons made.

	// Each process fills one row of the matrix, with some ranks doing multiple rows
	// as needed.  For example, given a 6x6 H0 matrix and four processes, rank 0 will do
	// rows 0 and 4, rank 1 will do rows 1 and 5, rank 2 will do row 2 and rank 3 will do 
	// row 3.
	for (int i = rank; i < size; i += num_procs){
		double * A = new double [6];
		for (int j = 0; j < size; j ++){

			// Gets the occupied states from the basis cases which are
			// being considered.
			int a, b, c, d, w, x, y, z;

			a = basis [i][0];	w = basis [j][0];
			b = basis [i][1];	x = basis [j][1];
			c = basis [i][2];	y = basis [j][2];
			d = basis [i][3];	z = basis [j][3];

			// This test to see if all eight values are the same
			// i.e. the states being described have all four particles
			// in the exact same locations
			if (a == w && b == x && c == y && d == z){
			    // Extracts the information for all four particles
				double ea, eb, ec, ed;
				ea = energies [a];
				eb = energies [b];
				ec = energies [c];
				ed = energies [d];

			    // Calculates the matrix element by adding the energies
			    // of all four particles and then subtracting the
			    // constant g from the total energy
				double total = ea + eb + ec + ed - g;

			    // Sets the matrix element to the value calcualted above
				A[j] = total;
			}

			// The next four conditions find all conditions where two of the
			// particles are in the same locations, but two of the particles
			// are in different locations.  For all of these conditions, the matrix
			// element is set to g/2.

			// The value is g/2 becuase only one pair of particles is participating in
			// the interaction and g is defined as the interaction strength between two
			// pairs of particles.
			else if (a == w && b == x && c != y && d != z){
				A[j] = g / 2;
			}
			
			else if (a == y && b == z && c != w && d != x){

				A[j] = g / 2;
			}

			else if (c == w && d == x && a != y && b != z){


				A[j] = g / 2;
			}

			else if (c == y && d == z && a != w && b != x){

				A[j] = g / 2;
			}
			
			// The else case is the case where none of the particles are in the same 
			// position across the two basis states being compared.  In this case the
			// matrix element is 0 due to the Condon-Slater rule.  
	
			// Condon-Slater Rule:  with two-body interactions, an integral of the type
			// <PhiI|HI|PhiJ> is non-zero if PhiI and PhiJ differ at most by two single 
			// particle states

			// It is because of the Condon-Slater rule that one-body interactions for this
			// case are not considered because they are zero.  For one-body interactions, the
			// Condon-Slater Rule requires that PhiI and PhiJ differ by no more than one single
			// particle state, which would mean that no excitations could happen            
			else{

				A[j] = 0;
			}// else
		} // for j
		if (rank != 0){
			std :: cout << rank << i << "\n";
			MPI::COMM_WORLD.Send (&A, 6, MPI::DOUBLE, 0, i);
		}
		else{
			H0[i] = A;
		}

	} // for i
	
	MPI::COMM_WORLD.Barrier ();

	if (rank == 0){
		MPI::COMM_WORLD.Recv (H0[1], 6, MPI::DOUBLE, 1, 1);
		MPI::COMM_WORLD.Recv (H0[2], 6, MPI::DOUBLE, 2, 2);
		MPI::COMM_WORLD.Recv (H0[3], 6, MPI::DOUBLE, 3, 3);
		MPI::COMM_WORLD.Recv (&H0[5], 6, MPI::DOUBLE, 1, 5);
	

	}	

	// Terminates the MPI Process.
	MPI::Finalize ();

	// Returns the complete Hamiltonian
	return H0;
	
}
