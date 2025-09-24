from pulp import *

x1 = LpVariable("x1")
x2 = LpVariable("x2",0, cat="Integer")
x3 = LpVariable("x3",0, cat="Integer")
x4 = LpVariable("x4",0, cat="Integer")
x5 = LpVariable("x5",0, cat="Integer")

prob = LpProblem("Exercise_1", LpMaximize)

prob += 3*x1+11*x2+9*x3-x4-29*x5
prob += x2 + x3 + x4 - 2*x5 <=4
prob += x1 - x2 + x3 + 2*x4 + x5 >= 0
prob += x1 + x2 + x3 - 3*x5 <= 1

status = prob.solve()
print(LpStatus[status])

for var in prob.variables(): print(value(var))
print('\n')
print(value(x2)+value(x3)+value(x4)-2*value(x5))
print(value(x1)-value(x2)+value(x3)+2*value(x4)+value(x5))
print(value(x1)+value(x2)+value(x3)-3*value(x5))
print('\n')
print(prob.objective.value())