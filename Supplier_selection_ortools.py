# Kapil Sethi, Masters of Operational Research 2025, University of Delhi, India
# email id : kapil.du.or.25@gmail.com
# This code is a solution to the Supplier Selection Problem using Google OR-Tools.

from ortools.linear_solver import pywraplp
import numpy as np
#storing data
discounts = [
    [7, 22, 25, 34, 35],
    [30, 18, 18, 31, 26],
    [21, 30, 34, 12, 14],
    [27, 29, 30, 60, 6],
    [31, 25, 10, 13, 30],
    [23, 32, 15, 60, 9],
    [6, 21, 18, 60, 28],
    [17, 18, 6, 8, 31]
]


default_prob = [0.30091579, 0.72152552, 0.11573458, 0.20984717, 0.83472454]
cost_of_risk = 100000
demand = [592, 446, 548, 647, 245, 797, 603, 401]
list_price = [870, 630, 960, 400, 980, 510, 830, 550]
fixed_admin_cost = 50000
reliability = 0.8
num_product = len(discounts)
num_supplier = len(discounts[1])


# Create the mip solver with the SCIP backend.
solver = pywraplp.Solver.CreateSolver('SCIP')

#Create the decision variables
infinity = solver.infinity()

y = {} #binary decision variable
for j in range(num_supplier):
    y[j] = solver.IntVar(0, 1, 'y[%i]' % (j+1))
print(y)

x_var = {} #Decision variable for the quantity of ith Product from jth Supplier
k = 0
for i in range(num_product):
  x_var[i] = [solver.NumVar(0, infinity, 'X[%d][%d]' %((i+1),(j+1))) for j in range(num_supplier)]


print(x_var)
print(type(x_var))

import pandas as pd
pd_frame = pd.DataFrame.from_dict(x_var)
print(pd_frame.transpose())

print('Number of variables =', solver.NumVariables())

#Create the constraints

# Creating reliability constraint
for i in range(num_product):
  for j in range(num_supplier):
    solver.Add(x_var[i][j] <= reliability*demand[i]*y[j])


#Creating demand satisfaction constraint
for i in range(num_product):
  expr = [x_var[i][j] for j in range(num_supplier)]
  solver.Add(sum(expr) >= demand[i])


print('Number of constraints =', solver.NumConstraints())

objective_terms = []
for i in range(num_product):
    for j in range(num_supplier):
        objective_terms.append((1-(discounts[i][j])/100) * list_price[i] * x_var[i][j])

for j in range(num_supplier):
    objective_terms.append(fixed_admin_cost  * y[j])

for k in range(num_supplier):
    objective_terms.append(cost_of_risk  * y[k] * default_prob[k])


solver.Minimize(solver.Sum(objective_terms))

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
  print('Minimum Cost Incurred to Satisfy Demand = ', solver.Objective().Value())
  print()
  for j in range(num_supplier):
    print(y[j] , ' = ', y[j].solution_value())
  for i in range(num_product):
    print()
    for j in range(num_supplier):
      if(round(x_var[i][j].solution_value(),2) == -0.0):
        print(x_var[i][j], " = ", 0.0 , "| ",end=" ")
      else:
        print(x_var[i][j], " = ", round(x_var[i][j].solution_value(),2), "| ",end=" ")

else:
        print('The problem does not have an optimal solution.')
