/************************************************************************************************
MBPT-Butler.cpp
June 21st, 2018

A code which applies second order many-body pertubation theory to the paring problem.  This
paring problem has eight states with four particles, giving a total number of six possible
configurations.

June 21st, 2018 -- Constructs the 6x6 Hamiltonian and has a method for printing the 
	Hamiltonian
July 2nd, 2018 -- Code is complete.  Calculates the correlation energy of the pairing 
	model using second and third order many-body pertubation theory.

To-Do:
1. Document the code
2. Time test
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
// Using the namespace std, prevents having to type the namespace (ex: std::cout --> cout)
using namespace std;

/************************************************
*                                                                                             *
*                                   Global Variables                                   *
*                                                                                             *                                                                                  
************************************************/
// The level spacing
double d = 1.0;

// Energy calculations for the different energy levels.  For example,
// for the level p = 1, the energy is p*d, or 1*d, which is stored in p1.
double  energy_p0  = 0*d;
double energy_p1 = d;
double energy_p2 = 2*d;
double energy_p3 = 3*d;

// The values in the array are indices in the states array.  The indices refer to the
// states that are below the Fermi level, i.e. the states that are occupied in the ground
// state of the pairing model.  The ground state of the pairing model is defined using
// the following diagram:
// p = 3  _ _
// p = 2  _ _ 
// p = 1  * *
// p = 0  * *
// Therefore the Fermi level is between p = 1 and p = 2.
int states_below_Fermi  [4] = {0, 1, 2, 3};

// The values in the array are indices in the states array that refer to the states that
// are above the Fermi level.
int states_above_Fermi [4] = {4, 5, 6, 7};


// For all states, the two dimensional array gives the p value, the spin, and the energy
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

// The energy values for a particular p value have the value E = p*d.

// Fills in the values of p, ms, and E for each orbital.  The numbering scheme
// is as follows:
// _ _  p = 3  6 7
// _ _  p = 2  4 5
// _ _  p = 1  2 3 
// _ _  p = 0  0 1

// In the array, the first number (index 0) is the p value and the second 
// number (index 1) is the ms value and the third number (index 2) is the energy value


double states [8][3] = {
	{0, 1, energy_p0}, {0, -1, energy_p0},
	{1, 1, energy_p1}, {1, -1, energy_p1},
	{2, 1, energy_p2}, {2, -1, energy_p2},
	{3, 1, energy_p3}, {3, -1, energy_p3}
};

/************************************************
*                                                                                             *
*    Prototypes (definitions given later)                                          *
*                                                                                             * 
************************************************/

double assymmetrized ( int w, int x, int y, int z, double g);

double twoParticleTwoHole (double g);

double denominator (int w, int x, int y, int z);

double mbpt_2 (double g);

double mbpt_3 (double g);


/************************************************
*                                               	                                       *
*                   main                                                                   * 
*                                                                                             *
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

	double g_values [21] = {-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, 
		-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

	double * mbpt_2_values = new double [21];
	double * mbpt_3_values = new double [21];

	for (int i = 0; i < 21; i ++){
		mbpt_2_values [i] = mbpt_2 (g_values [i]);
		mbpt_3_values [i] = mbpt_3 (g_values [i]);
	}


/*	for (int i = 0; i < 21; i ++){
		cout << mbpt_2_values [i] << "\t" << mbpt_3_values [i] << "\n";
	} 
*/	

	// Required return of an int.
	return 0;
} //main


double assymmetrized (int w, int x, int y, int z, double g){
	int pw = states [w][0]; int sw = states [w][1];
	int px = states [x][0]; int sx = states [x][1];
	int py = states [y][0]; int sy = states [y][1];
	int pz = states [z][0]; int sz = states [z][1];

	double matrix_element = -1000;

	if (pw != px || py != pz){
		matrix_element = 0;
	}
	else if (sw == sx || sy == sz){
		matrix_element = 0;
	}
	else if (sw == sy && sx == sz){
		matrix_element = (-g) / 2;
	}
	else if (sw == sz && sx == sy){
		matrix_element = g / 2;
	}

	return matrix_element;
}

double denominator (int w, int x, int y, int z){
	return states [w][2] + states [x][2] - states[y][2] - states[z][2];	
}

double twoParticleTwoHole (double g){
	double energy;

	for (int ai = 0; ai < 4; ai ++){
		for (int bi = 0; bi < 4; bi ++){
			for (int ii = 0; ii < 4; ii ++){
				for (int ji = 0; ji < 4; ji ++){
					int a = states_above_Fermi [ai]; 
					int b = states_above_Fermi [bi];
					int i = states_below_Fermi [ii];
					int j = states_below_Fermi [ji];

					energy += 0.25 * assymmetrized (a, b, i, j, g) * assymmetrized (i, j, a, b, g) / denominator (i, j, a, b);
				}
			}
		}
	}

	return energy;
}

double mbpt_2 (double g){
	return twoParticleTwoHole (g);
}

double mbpt_3 (double g){
	double energy = twoParticleTwoHole (g);

	for (int ai = 0; ai < 4; ai ++){
		for (int bi = 0; bi < 4; bi ++){
			for (int ci = 0; ci < 4; ci ++){
				for (int ii = 0; ii < 4; ii ++){
					for (int ji = 0; ji < 4; ji ++){
						for (int ki = 0; ki < 4; ki ++){
							int a = states_above_Fermi [ai];
							int b = states_above_Fermi [bi];
							int c = states_above_Fermi [ci];

							int i = states_below_Fermi [ii];
							int j = states_below_Fermi [ji];
							int k = states_below_Fermi [ki];

							energy += assymmetrized (i, j, a, b, g) * assymmetrized (a, c, j, k, g) * assymmetrized (b, k, c, i, g) / denominator (i, j, a, b) / denominator (k, j, a, c);
						}
					}
				}
			}
		}
	}

	for (int ai = 0; ai < 4; ai ++){
		for (int bi = 0; bi < 4; bi ++){
			for (int ci = 0; ci < 4; ci ++){
				for (int di = 0; di < 4; di ++){
					for (int ii = 0; ii < 4; ii ++){
						for (int ji = 0; ji < 4; ji ++){
							int a = states_above_Fermi [ai];
							int b = states_above_Fermi [bi];
							int c = states_above_Fermi [ci];
							int d = states_above_Fermi [di];

							int i = states_below_Fermi [ii];
							int j = states_below_Fermi [ji];

							energy += 0.125 * assymmetrized (i, j, a, b, g) * assymmetrized (a, b, c, d, g) * assymmetrized (c, d, i, j, g) / denominator (i, j, a, b) / denominator (i, j, c, d);
						}
					}
				}
			}
		}
	}

	for (int ai = 0; ai < 4; ai ++){
		for (int bi = 0; bi < 4; bi ++){
			for (int ii = 0; ii < 4; ii ++){
				for (int ji = 0; ji < 4; ji ++){
					for (int ki = 0; ki < 4; ki ++){
						for (int li = 0; li < 4; li ++){
							int a = states_above_Fermi [ai];
							int b = states_above_Fermi [bi];
	
							int i = states_below_Fermi [ii];
							int j = states_below_Fermi [ji];
							int k = states_below_Fermi [ki];
							int l = states_below_Fermi [li];

							energy += 0.125 * assymmetrized (i, j, a, b, g) * assymmetrized (k, l, i, j, g) * assymmetrized (a, b, k, l, g) / denominator (i, j, a, b) / denominator (k, l, a, b);
						}
					}
				}
			}
		}
	}

	return energy;
}

/************************************************
*
* Old Code (No longer needed)
*
************************************************/


/************************************************
*                                               *
*               printHamiltonian                *
*                                               *
************************************************/
//void printHamiltonian (double ** & H0, int n){
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
//	for (int i = 0; i < n; i ++){
//		for (int j = 0; j < n; j ++){
			// Prints the current matrix element with a tab afterwards for
			// formatting purposes
//			cout << H0[i][j] << "\t";
//		}
		// Prints a new line character after each j-cycle so each row of the 
		// matrix prints on its own line
//		cout << "\n";
//	}
//}

/************************************************
*                                               *
*             buildHamiltonian                  *
*                                               *
************************************************/
//double ** buildHamiltonian (double ** & H0, int size){
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
//	double g("g");
	// The level spacing
//	double d("d");

	// Energy calculations for the different energy levels.  For example,
	// for the level p = 1, the energy is p*d, or 1*d, which is stored in p1.
//	double p0  = 0*d;
//	double p1 = d;
//	double p2 = 2*d;
//	double p3 = 3*d;
	
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
//	double ** states = new double * [8];
	// Initializes each inner array
//	for (int i = 0; i < 8; i ++){
//		states [i] = new double [2];
//	}
	
	// Fills in the values of p and ms for each orbital.  The numbering scheme
	// is as follows:
	// _ _  p = 3  6 7
	// _ _  p = 2  4 5
	// _ _  p = 1  2 3 
	// _ _  p = 0  0 1
	// In the array, the first number (index 0) is the p value and the second 
	// number (index 1) is the ms value
//	states [0][0] = 0; states [0][1] = 1; 
//	states [1][0] = 0; states [1][1] = -1; 
//	states [2][0] = 1; states [2][1] = 1; 
//	states [3][0] = 1; states [3][1] = -1; 
//	states [4][0] = 2; states [4][1] = 1; 
//	states [5][0] = 2; states [5][1] = -1; 
//	states [6][0] = 3; states [6][1] = 1; 
//	states [7][0] = 3; states [7][1] = -1;

	// Assigns the energies of each orbital to the appropriate index in the array
	// using the numbering scheme shown above.  It has to be an array of double
	// objects because d is a symbol.
//	double * energies = new double [8];
//	energies [0] = p0; energies [1] = p0; 
//	energies [2] = p1; energies [3] = p1;
//	energies [4] = p2; energies [5] = p2;
//	energies [6] = p3; energies [7] = p3;

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
//	double ** basis = new double * [6];
//	for (int i = 0; i < 6; i ++){
//		basis [i] = new double [4];
//	}
//	basis [0][0] = 0; basis [0][1] = 1; basis [0][2] = 2; basis [0][3] = 3;
//	basis [1][0] = 0; basis [1][1] = 1; basis [1][2] = 4; basis [1][3] = 5;
//	basis [2][0] = 0; basis [2][1] = 1; basis [2][2] = 6; basis [2][3] = 7;
//	basis [3][0] = 2; basis [3][1] = 3; basis [3][2] = 4; basis [3][3] = 5;
//	basis [4][0] = 2; basis [4][1] = 3; basis [4][2] = 6; basis [4][3] = 7;
//	basis [5][0] = 4; basis [5][1] = 5; basis [5][2] = 6; basis [5][3] = 7;

	// Fills the Hamiltonian with the appropriate values. The values are
	// found by evaulating the expression <PhiI|HI|PhiJ>.   However, the
	// program does not directly analyze the expression.  The values for
	// all possible combinations of PhiI and PhiJ are known, so these
	// values are directly assigned to the appropriate matrix elements.

	// i and j cycle through all of the basis states and compares 
	// them.  There are a total of 36 comparisons made.
//	for (int i = 0; i < size; i ++){
//		for (int j = 0; j < size; j ++){

			// Gets the occupied states from the basis cases which are
			// being considered.
//			int a, b, c, d, w, x, y, z;

//			a = basis [i][0];	w = basis [j][0];
//			b = basis [i][1];	x = basis [j][1];
//			c = basis [i][2];	y = basis [j][2];
//			d = basis [i][3];	z = basis [j][3];

			// This test to see if all eight values are the same
			// i.e. the states being described have all four particles
			// in the exact same locations
//			if (a == w && b == x && c == y && d == z){
			    // Extracts the information for all four particles
//				double ea, eb, ec, ed;
//				ea = energies [a];
//				eb = energies [b];
//				ec = energies [c];
//				ed = energies [d];

			    // Calculates the matrix element by adding the energies
			    // of all four particles and then subtracting the
			    // constant g from the total energy
//				double total = ea + eb + ec + ed - g;

			    // Sets the matrix element to the value calcualted above
//				H0[i][j] = total;
//			}

			// The next four conditions find all conditions where two of the
			// particles are in the same locations, but two of the particles
			// are in different locations.  For all of these conditions, the matrix
			// element is set to g/2.

			// The value is g/2 becuase only one pair of particles is participating in
			// the interaction and g is defined as the interaction strength between two
			// pairs of particles.
//			else if (a == w && b == x && c != y && d != z){
//				H0[i][j] = g / 2;
//			}
			
//			else if (a == y && b == z && c != w && d != x){

//				H0[i][j] = g / 2;
//			}

//			else if (c == w && d == x && a != y && b != z){


//				H0[i][j] = g / 2;
//			}

//			else if (c == y && d == z && a != w && b != x){

//				H0[i][j] = g / 2;
//			}
			
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
//			else{

//				H0[i][j] = 0;
//			}
//		}
//	}

	// Returns the complete Hamiltonian
//	return H0;
//}
