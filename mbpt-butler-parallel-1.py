#########################################################################################
# MBPT-Butler-Parallel-1.py
# Version 1.1
# June 27th, 2018
# A code which applies second order many-body pertubation theory to the paring 
# problem.  This paring problem has eight states with four particles, giving a 
# total number of six possible configurations.  An attempt to shorten the runtime
# is made by utilizing mpi4py to parallelize certain task to that they run concurrently
#
# This version of the program is designed to be run with exactly four cores.  Running
# the program with more cores will not cause an error, but running the program with less 
# cores will cause the program to terminate.
#
# To run the program using four cores, use the following command:
# mpiexec -n 4 python3 mbpt-butler-parallel-1.py
#
# For more information see Lecture Notes in Physics #936 Chapter 8 Section 5.
#
# TO-DO
# 1. Update documentation
# 2. Perform time test
#########################################################################################

#########################################################################################
# Methods:
# printHamiltonian (H, n): takes in a nested, square array and the number of 
#	columns/rows and prints the matrix in a nicely formatted array
# buildHamiltonian: Calcualtes the Hamiltonian for the pairing problem for a system 
#	of N=8 and four particles (Fermions).
# assymmetrized (w, x, y, z): Takes in the indices of four states and returns the two
# 	body antisymmetrized matrix element calculated with those states
# denominator (holes, particles): takes in a tuple of hole states and a tuple of particle
# 	states and returns the denominator of the equations defined below, calculated
#	with the energies of the inputted states.
# insertValues (symbolic_energy, values, variable): takes in a symbolic expression and 
#	returns a list of numbers calculates by replacing the symbolic variable with each
# 	member of the list values
# insertValue (symbolic_energy, value variable): takes in a symbolic expression, a 
#	double, and a symbolic variable and returns the result of substituting value
#	into variable in symbolic_energy.
#########################################################################################

#########################################################################################
# Global Variables:
# d: a symbolic variable representing the spacing between the energy levels.  Does not
#	currently have a value, but generally set to one.
# g: a symbolic variable representing the interactions strength between the particles.
#	Calculations are performed with a symbolic g, but at the end various values of g
#	are plugged in between -1 and 1, in increments of 0.1 in order to get data for
#	a graph.
# states: a list of tuples containing the characteristic values of the eight possible
#	energy states of the system.  For each states, the tuple contains 3 numbers.  The
#	first number is the p value, which refers to the energy level.  The second number
#	is the spin of the state.  Due to the Pauli Exclusion Principle, there can only
#	be one spin up (1) and one spin down (-1) state per energy level.  The third 
#	number is the energy of the state, calculated by the formula E = p*d.
# basis: a list of tuples that represents the six possible arrangements of particles 
#	that can be created without breaking pairs.  The numbering of the arrangements
#	was set in class.  Each tuple of four numbers tells which four states are 
#	occupied by particles in the particular arrangement.  The arrangements are shown
#	below.
# states_below_Fermi: a tuple which list the states that are below the Fermi level in 
#	this system.  
# states_above_Fermi: a tuple which list the states that are above the Fermi level in
#	this system.
#########################################################################################

#################################################
#                                               #
#                 IMPORTS                       #
#                                               #
#################################################

# Allows for the symbolic representation of d and g without
# assigning them a specific value
from sympy import *
# The product tool allows for the replacement of nested for loops.
# See the not below.
from itertools import product
# Allows for graphing capabilities 
import matplotlib.pyplot as plt
# Allows for the parallelization of the code
from mpi4py import MPI
# Allows for the code to be terminated if not enough processes are used to run the code
from sys import exit

from time import clock


#####################################################################
# Itertools.Product:
# Replaces nested for loops.  For example, given the tuples 
# one = (0, 1, 2, 3) and two = (4, 5, 6, 7), the following nested
# for loop:
# for a in one:
#	for b in two:
#		print (a, b)
# prints the pairs of numbers 0, 4; 0, 5; 0, 6; 0, 7; 1, 4; 1, 5; etc.
# The book implementation of this code (found on the GitHub at:
# https://github.com/ManyBodyPhysics/LectureNotesPhysics/tree/master/Programs
# uses four, and even six, nested for loops, which makes the code hard 
# to read and generally ugly.  For example to loop over the tuple one
# twice and the tuple two twice, the for loop structure will look like:
# for a in one:
#	for b in one:
#		for c in two:
#			for d in two:
# Whereas the itertools.product version of the code would look like:
# for a, b, c, d in product (one, two, three, four):, which is much
# shorter and neater.  Experimentation needs to occur to see if it
# is actually faster, because looks do not matter if the implementation
# sucks.  Preliminary investigation does seem to indicate the it at
# least as fast, if not a little faster than the nested for loops.
#####################################################################

#################################################
#                                               #
#             Global Variables                  #
#                                               #
#################################################
# The level spacing 
d = Symbol("d")
# The constant g, which represents the interaction strength
# between two pairs of particles
g = Symbol("g")


# A list of all the quantities related to the states
# (p, spin, energy)
# p
# _ _  p = 3
# _ _  p = 2
# _ _  p = 1
# _ _  p = 0
# Spin
# up  ms = 1
# down  ms = -1
# Energy
# E = p*d

# Due to the Pauli Exclusion Principle, two particles
# with the same p value must have different spins

# The numbering of the states will be as followed, defined
# by the indexing of the list:
# _ _  6 7
# _ _  4 5
# _ _  2 3
# _ _  0 1

states = [(0, 1, 0), (0, -1, 0),
	  (1, 1, d), (1, -1, d),
	  (2, 1, 2*d), (2, -1, 2*d),
	  (3, 1, 3*d), (3, -1, 3*d)
]

# A list of all possible arrangements of particles
# Phi0
# _ _
# _ _
# * *
# * *
# Phi1
# _ _
# * *
# _ _
# * *
# Phi2
# * *
# _ _
# _ _
# * *
# Phi3
# _ _
# * *
# * *
# _ _
# Phi4
# * *
# _ _
# * *
# _ _
# Phi5
# * *
# * *
# _ _
# _ _
basis = [(0, 1, 2, 3), # Phi0
	 (0, 1, 4, 5), # Phi1
	 (0, 1, 6, 7), # Phi2
	 (2, 3, 4, 5), # Phi3
	 (2, 3, 6, 7), # Phi4
	 (4, 5, 6, 7)  # Phi5
]

# Since Phi0 is defined as the ground state of this system, the Fermi level is defined 
# right above the highest occupied states.  So in this system, states that are below
# the Fermi level are 0, 1, 2, and 3.  The states above the Fermi level are 4, 5, 6, 
# and 7.

# The states that are below the Fermi level
# i.e. The occupied states in Phi0
states_below_Fermi = (0, 1, 2, 3)

# The states that are below the Fermi level
# i.e. The unoccupied states in Phi0
states_above_Fermi = (4, 5, 6, 7)

#################################################
#                                               #
#              MPI Initialized                  #
#                                               #
#################################################
# Shortcut so that MPI.COMM_WORLD does not need to be typed everytime a MPI command is 
# used
comm = MPI.COMM_WORLD

# Gets the total number of processes that are used to run the program
num_procs = comm.Get_size ()

# Gets the number of the particular process running this code
rank = comm.Get_rank ()

# This code is designed to be run with exactly four processes.  If there are not at 
# least four processes, then not all results will be calculated.  More than four 
# processes does not cause a problem; the extra process will just not have a job.
# However, any less than four processes will cause the wrong results to be computed.
# This if statements terminates the code if it is ran with less than four processes.
if num_procs < 4:
	# Prints an error message so the user knows why the code has been terminated.
	print ("There are not enough processes to run this code.")
	print ("This code is designed to be run with four processes.")
	print ("The code will now terminate.")
	# Terminates the program
	exit ()


#################################################
#                                               #
#                 assymmetrized                 #
#                                               #
#################################################
def assymmetrized (w, x, y, z):
	"""
	Input:
		w, x, y, z (integers): these integers refer to the indices in the states
			list.  They represent particular states in the system that are
			either occupied by holes or by particles.

	Output:
		matrix_element (a symbol):  the result of calculating the 
			anti-symmetrized matrix element calculated given the four 
			inputted states.
	
	Calculates the anti-symmetrized matrix element for the four inputted states.
	"""
	# Extracts the attributes of each state from the states list.  The first number 
	# is the p value, the second number is the spin, and the third number is the
	# energy of the state.
	pw, sw, ew = states [w]
	px, sx, ex = states [x]
	py, sy, ey = states [y]
	pz, sz, ez = states [z]

	# Where the matrix element will be stored when calcualted.  It defaults to 
	# "none" so that it can be determined if any states slip through the if/elif
	# statements.  However, every combination of w, x, y, and z should be filtered
	# into one of the statements.
	matrix_element = "none"

	# w and x should be on the same energy level.  Same for y and z since pairs of
	# particles can not be broken in this model.  So if either w and x are on 
	# different levels or y and z are on different levels, then matrix element
	# is set to zero since the situation does not make sense in the model.  This way
	# it does not contribute to the sum that calculates the energy.
	if pw != px or py != pz:
		matrix_element = 0 

	# w and x should have different/opposite spins.  The same for y and z.  Since w/x
	# and y/z are on the same energy level, due to the Pauli Exclusion Principle, 
	# they cannot have the same spin.  Since that situation does not make since in the
	# model, the matrix element is set to zero so it will not contribute to the sum 
	# that calculates the energy.
	elif sw == sx or sy == sz:
		 matrix_element = 0

	# This is a valid arrangement of the particles, so this matrix element
	# contributes to the energy sum.  This is the case where the spins line up.  For
	# example, all the up spins on the left and all the down spins on the right, i.e.
	# _ _
	# u d
	# _ _
	# u d
	# In the case, the matrix element is set to -g/2.  EXPLAIN WHY.
	elif sw == sy and sx == sz:
		matrix_element = (-g) / 2

	# This is a valid arrangement of the particles, so this matrix element
	# contributes to the energy sum.  This is the case where the spins do not line	
	# up.  For example:
	# _ _
	# u d
	# _ _
	# d u
	# In this case, the matrix element is set to g/2.  EXPLAIN WHY. 
	elif sw == sz and sx == sy:
		matrix_element = g / 2

	# Returns the value of the matrix element to be used in the calculations
	return matrix_element

	


#################################################
#                                               #
#                  denominator                  #
#                                               #
#################################################
def denominator (holes, particles):
	"""
	Input: 
		holes (a tuple of integers): For the particular situation being examined, 
			a list of the states which are below the Fermi level 
			(defined above)
		particles (a tuple of integers): For the particular situation being 
			examined, a list of the states which are above the Fermi level.
	
	Output:
		denom (a double): The results of calcualting the denominator of the 
			equations (1), (3), (4), or (5).  

	Calculates the denominator of the Equations (1), (3), (4), and (5), which are
	defined below.  It is calculated by subtracting the energies of the states that 
	contain particles above the Fermi level and adding the energies of the 
	holes that are below the Fermi level.
	"""
	# Stores the result of calcualting the denominator
	denom = 0

	# For each state in the holes tuple, the energy of that state is added to denom.
	# In the book code, the p value is used in this calculation, but that is 
	# assuming an energy spacing value, d, of 1, since E = p*d. This way of defining
	# the energy allows for the value of d to be set to any value.

	for h in holes:
		# Based on the way states is defined above, states[h] returned a
		# tuple of three numbers.  The first one is the p value of the 
		# state, the second is the spin value of the state, and the third 
		# is the energy value of the state.
		ph, sh, eh = states [h]
		denom = denom + eh

	# For each state in the particles tuple, the energy of that state is added to 
	# denom.  
	
	# The second process is used to calculate the denomination contributoin from the 
	# particles.  This runs at the same time that the first process is calculating
	# the contribution from the holes.

	for p in particles:
		# Based on the way states is defined above, states[h] returned a
		# tuple of three numbers.  The first one is the p value of the 
		# state, the second is the spin value of the state, and the third
		# is the energy value of the state.
		pp, sp, ep = states [p]
		denom = denom - ep


	# Returned the value of the denominator to be used in the equations
	return denom


#################################################
#                                               #
#             insertValues                      #
#                                               #
#################################################
def insertValues (symbolic_energy, values, variable): 
	"""
	Input:
		symbolic_energy (a symbolic expression): An expression containing one or
			more symbolic variables to which one of the symbolic variables
			needs to have a numeric value assigned.
		values (a list): the values that need to be plugged into the given 
			symbolic variable
		variable (a symbol): the symolic variable that is to be replaced with the
			numbers given in the values list
	Output:
		numeric_energies (a list): a list of numbers which results when values is
			plugged into the symbolic_energy expression

	Takes a symbolic expression for the energy and replaces one of the symbolic 
	variables with a list of given values.  It returns a list of numbers which 
	result from the substitution.
	"""
	# Where the numeric energies will be stored.
	numeric_energies = []

	# Cycles through the values to be substituted.  For each value it calculates
	# the result obtained by plugging the value into the symbolic expression for the
	# given variable.  It then takes the calculated value and appends it to the 
	# numeric_energies list.
	for value in values:
		numeric_energies.append (symbolic_energy.subs (variable, value))

	# The result of the substitutions
	return numeric_energies

#################################################
#                                               #
#                  insertValue                  # 
#                                               #
#################################################
def insertValue (symbolic_energy, value, variable):
	"""
	Input:
		symbolic_energy (a symbolic expression): An expression containing one or
			more symbolic variables to which one of the symbolic variables
			needs to have a numeric value assigned.
		value (a ldouble): the value that needs to be plugged into the given 
			symbolic variable
		variable (a symbol): the symolic variable that is to be replaced with the
			numbers given in the value
	Output:
		unnamed (a double): the result of substituting the given value into the 
			specified variable
	
	Takes a symbolic expression and replaces a given variable with a specified value 
	and returns the result of the substitution.
	"""
	return symbolic_energy.subs(variable, value)

#########################################################################################
#
# End of methods, start of MPI test code
#
#########################################################################################
if rank == 0:
	start_time = clock ()

# The following equations are used to calculate the correlation energy of the pairing 
# model using second and third order many-body pertubation theory.   They are derived
# from the Goldstone diagrams which are found on page 332 of Lecture Notes in Physics 
# #936.
#
# (1) = (1/(2^2))*SUM(i,j <= F)SUM(a,b > F) <ij|v|ab><ab|V|ij>
#                                           -------------------
#                                              (Ei+Ej-Ea-Eb)
#
# (3) = SUM(i,i,k <= F)SUM(a,b,c > F) <ij|V|ab><bk|V|ic><ac|V|ik>
# 				    ----------------------------
# 	                            (Ei+Ej-Ea-Eb)(Ei+Ek-Ez-Ec)
#
# (4) = (1/(2^3)) SUM(i,j <= F)SUM(a,b,c,d > F) <ij|V|cd><cd|V|ab><ab|V|ij>	
# 					      ----------------------------
# 					       (Ei+Ej-Ec-Ed)(Ei+Ej-Ea-Eb)
#
# (5) = (1/(2^3)) SUM(i,j,k,l <= F)SUM(a,b > F) <ab|V|kl><kl|V|ij><ij|V|av>
# 					      ----------------------------
# 					      (Ei+Ej-Ea-Eb)(Ek+El-Ea-Eb)
#
# The correlation energy calculated using second order many-body pertubation theory is 
# simply the result of analyzing Equation (1).  The correlation energy calculated using
# third order many-body pertubation theory is the summation of the results of Equations
# (1), (3), (4), and (5).
#
# The results of these four equations are calculated using a parallel implementation.  
# This code is designed to be run with four processes: one process per equation.  The 
# first process computes the result of Equation (1), the second process computes the 
# result of Equation (3), the third process computes the result of Equation (4), and the
# fourth and final process computes the result of Equation (5).  The individual results
# from each process are the combined at the end to get to correlation energies.

	


# This will hold the result of each energy calculation by each process.  Each process
# will store a different result in this variable because each process is analyzing a 
# different equation
energy = 0

# From here on out, the letters at the beginning of the alphabet 
# (a, b, c, d, ...) represent particles that are above the Fermi level, i.e.
# the states 4, 5, 6, and 7.  Letters in the middle of the alphabet 
# (i, j, k, l, ...) represent holes that are below the Fermi level, i.e.
# the states 0, 1, 2, and 3.

# This calculates the result from Equation (1).  Rank 0 is the first process since the
# numbering starts at 0.
if rank == 0:
	# Replaces the structure of four nested for loops in the book code.  Combines
	# every combination of a, b, i, and j into the equation and sums the results.
	for a, b, i, j in product (states_above_Fermi, states_above_Fermi,		
		states_below_Fermi, states_below_Fermi):

		# Calculates one term of the double sum.  
		# The factor of 1/4 is actually two factors of 1/2 which come from
		# the fact that there are two equivilant pairs of fermions starting 
		# starting at the same vertex and ending at the same vertex (LNP936).
		# See the Goldstone diagrams on page 332.
		energy += ((1/4) * assymmetrized (a, b, i, j) 
			* assymmetrized (i, j, a, b) / denominator ((i,j), (a,b)))


# This calculates the results from equation (3). 
elif rank == 1:
	# The product structure replaces six nested for loops from the book code.  It is
	# used for equations (3) - (5) to remove the six nested loop structures.
	for a, b, c, i, j, k in product (states_above_Fermi, states_above_Fermi, 
		states_above_Fermi, states_below_Fermi, states_below_Fermi, 
		states_below_Fermi):

		energy += (assymmetrized (i, j, a, b) * assymmetrized (a, c, j, k) 
			* assymmetrized (b, k, c, i) / denominator ((i, j), (a, b)) 
			/ denominator((k, j), (a, c)))


# This structure calculates equation (4)
elif rank == 2:
	for a, b, c, d, i, j in product (states_above_Fermi, states_above_Fermi,
		states_above_Fermi, states_above_Fermi, states_below_Fermi,
		states_below_Fermi):

		energy += ((1/8) * assymmetrized (i, j, a, b) 
			* assymmetrized (a, b, c, d) 
			* assymmetrized (c, d, i, j) / denominator ((i, j), (a, b))  
			/ denominator((i, j), (c, d)))


# This structure calculates equation (5)
elif rank == 3:
	for a, b, i, j, k, l in product (states_above_Fermi, states_above_Fermi,
		states_below_Fermi, states_below_Fermi, states_below_Fermi,
		states_below_Fermi):
	
		energy += ((1/8) * assymmetrized (i, j, a, b) 
			* assymmetrized (k, l, i, j) 
			* assymmetrized (a, b, k, l) / denominator ((i, j), (a, b)) 
			/denominator ((k, l), (a, b)))

# This handles the case when there are too many processes.  Only four are needed.
else:
	print ("Too many processes for design")






#################################################
#                                               #
#       	     TEST                       #
#                                               #
#################################################
# The values of g (the interaction strength) which will be plugged into the results from
# mbpt_2 and mbpt_3 to graph the correlation energy has a function of interaction 
# strength.  Values from -1.0 to 1.0 were chosen, with an increment of 0.1.
g_values = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3,
	0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# The value of d (the energy level spacing).  It will generally be set to 1.
d_value = 1

# This method gathers the value of energy from each of the processes and stores them in
# a list ordered such that the result from rank 0 is at index 0, the result from rank 1
# is at index 1, and so on.

# This gathered energies are only given to process zero (indicated by "root = 0").  All
# other processes will have a blank list.  So from here on out, only rank 0 will be used.
correlation_energies = comm.gather (energy, root = 0)

if rank == 0:
	# Gets the result for the calculation of the correlation energy using second
	# order many-body pertubation theory.  Since the second order result is only
	# the result of Equation (1), it only needs the result calculated by rank 0, 
	# which was responsible for calculating the result of Equation (1)
	E_2 = correlation_energies [0]

	# Gets the result for the calculation of the correlation energy using third 
	# order many-body pertubation theory.  Since the third order result is the sum
	# of the results from Equations (1), (3), (4), and (5) the entire array needs to
	# be summed to get the final answer.
	E_3 = sum (correlation_energies)


	# Substitutes the value of d into the symbolic correlation energy expressions.  The 
	# results will still be a symbolic expression since g has not be substituted yet.
	E_2 = insertValue (E_2, d_value, d)
	E_3 = insertValue (E_3, d_value, d)


	# Substitutes the values of g into the symbolic correlation energy expressions.  The
	# results will be lists of correlation energies, each calculated with a different g value
	E_2 = insertValues (E_2, g_values, g)
	E_3 = insertValues (E_3, g_values, g)

	print ("--- %s seconds ---" % (clock () - start_time))

#################################################
#                                               #
#                   GRAPHING                    #
#     will be commented out before time test    #
#                                               #
#################################################
#if rank == 0:
#	plt.axis([-1,1,-0.5,0.05])
#	plt.xlabel(r'Interaction strength, $g$', fontsize=16)
#	plt.ylabel(r'Correlation energy', fontsize=16)
#	#exact = plt.plot(ga, exact,'b-*',linewidth = 2.0, label = 'Exact')
#	mbpt2 = plt.plot(g_values, E_2,'r:.', linewidth = 2.0, label = 'MBPT2')
#	mbpt3 = plt.plot(g_values, E_3, 'm:v',linewidth = 2.0, label = 'MBPT3')
#	plt.legend()
#	plt.savefig('perturbationtheory.pdf', format='pdf')
#	plt.show()




#########################################################################################
#
# Old code, no longer needed for successful execution of the code
#
#########################################################################################

#################################################
#                                               #
#             printHamiltonian                  #
#                                               #
#################################################
# def printHamiltonian (H, n):
"""
Input:
	H (a nested array): A Hamiltonian matrix (MUST be square)
	n (an integer): The dimension of the Hamiltonian matrix,
	    can either be the number of rows of columns since it is 
            a square matrix.
Output:
	None.

Takes the nested array given in the parameters and prints it in a nicely
formatted way.
"""

# Where the output to be printed is stored.  The final, formated string will have
# n lines.  Each line will have n elements, each separated by a tab.  For
# example, for the matrix [[a, b, c], [d, e, f], [g, h, i]], the formatted, 
# printed output will look like:
# a	b	c
# d	e	f
# g	h	i

#	output = ""

# Cycles through the n rows and n columns of the matrix and creates a formatted 
# string of all the matrix elements.  i cycles through the rows and j cycles 
# through the columns.
#	for i in range (0, n):
#		for j in range (0, n):
		# Appends the new matrix element of the formatted string 
		# with a tab afterwards
#			output = output + str(H[i][j]) + "\t"

	# Adds a new line after an entire row has been added to the 
	# formatted string
#		output += "\n"

# Prints the matrix as a nicely formatted string
#	print (output)

#################################################
#                                               #
#              buildHamiltonian                 #
#                                               #
#################################################
#def buildHamiltonian ():
"""
Input:
	None.
Output:
	H0 (a nested array): the Hamiltonian of the pairing problem.  
		It will be a 6x6 matrix.

Calcualtes the Hamiltonian for the pairing problem for a system of N=8 
and four particles (Fermions).

Currently d (the level spacing) and g (the interaction strength) are
symbolic and do not have specific values.
"""
# Sets up the empty 6x6 array which will be filled
# with the elements of the Hamilitonian below
#	H0 = [["None","None","None","None","None","None"],
#	      ["None","None","None","None","None","None"],
#	      ["None","None","None","None","None","None"],
#	      ["None","None","None","None","None","None"],
#	      ["None","None","None","None","None","None"],
#	      ["None","None","None","None","None","None"]
#	]


# Fills the Hamiltonian with the appropriate values. The values are
# found by evaulating the expression <PhiI|HI|PhiJ>.   However, the
# program does not directly analyze the expression.  The values for
# all possible combinations of PhiI and PhiJ are known, so these
# values are directly assigned to the appropriate matrix elements.

# i and j cycle through all of the basis states and compares 
# them.  There are a total of 36 comparisons made.
#	for i in range (0,6):
#		for j in range (0,6):
	
		# Removes the numbers from the basis tuple.  The four
		# numbers identify where the particles in that particular
		# basis are.
#			a, b, c, d = basis[i]
#			w, x, y, z = basis[j]
	
		# This test to see if all eight values are the same
		# i.e. the states being described have all four particles
		# in the exact same locations
#			if a == w and b == x and c == y and d == z:

		    # Extracts the information for all four particles
#			    pa, ma, ea = states[a]
#			    pb, mb, eb = states[b]
#			    pc, mc, ec = states[c]
#			    pd, md, ed = states[d]
		    
		    # Calculates the matrix element by adding the energies
		    # of all four particles and then subtracting the
		    # constant g from the total energy
#			    total = ea + eb + ec + ed - g
		    
		    # Sets the matrix element to the value calcualted above
#			    H0[i][j] = total
		    
	
		# The next four conditions find all conditions where two of the
		# particles are in the same locations, but two of the particles
		# are in different locations.  For all of these conditions, the
		# matrix element is set to g/2.

		# The value is g/2 becuase only one pair of particles is
		# participating in the interaction and g is defined as the
		# interaction strength between two pairs of particles.
#			elif a == w and b == x and c != y and d != z:
		    
#			    H0[i][j] = g/2
		      
#			elif a == y and b == z and c != w and d != x:

#			    H0[i][j] = g/2
		    
#			elif c == w and d == x and a != y and b != z:

#			    H0[i][j] = g/2
		    
#			elif c == y and d == z and a != w and b != x:

#			    H0[i][j] = g/2

		# The else case is the case where none of the particles are 
		# in the same position across the two basis states being 
		# compared.  In this case the matrix element is 0 due to the
		# Condon-Slater rule.  

		# Condon-Slater Rule:  with two-body interactions, an integral
		# of the type <PhiI|HI|PhiJ> is non-zero if PhiI and PhiJ differ 
		# at most by two single particle states

		# It is because of the Condon-Slater rule that one-body 
		# interactions for this case are not considered because they are
		# zero.  For one-body interactions, the Condon-Slater Rule 
		# requires that PhiI and PhiJ differ by no more than one single
		# particle state, which would mean that no excitations could
		# happen              
#			else:
#			    H0[i][j] = 0

# Returns the complete Hamilitonian 
#	return H0

