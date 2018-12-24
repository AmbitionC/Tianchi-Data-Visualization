# coding:utf-8

'''
# Author: chenhao
# Date: Aug.8.2018
# Description: site selection using Genetic Algorithm(GA) to draw the procedure
# Using GAFT
'''


#########################################################################
# 一维函数寻优过程，函数方程为f(x) = x + 10sin(5x) + 7cos(4x)
#########################################################################

'''
Find the global maximum for function: f(x) = x + 10sin(5x) + 7cos(4x)
'''
'''
from math import sin, cos

from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore

# Define population.
indv_template = BinaryIndividual(ranges=[(0, 10)], eps=0.001)
population = Population(indv_template=indv_template, size=30).init()

# Create genetic operators.
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return x + 10*sin(5*x) + 7*cos(4*x)

# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmax)
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)

if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)
'''

'''
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt

best_fit = [
    (0, [7.84423828125], 24.827058441949525),
    (1, [7.84423828125], 24.827058441949525),
    (2, [7.862548828125], 24.84926580952249),
    (3, [7.862548828125], 24.84926580952249),
    (4, [7.862548828125], 24.84926580952249),
    (5, [7.861328125], 24.85156036261068),
    (6, [7.861328125], 24.85156036261068),
    (7, [7.861328125], 24.85156036261068),
    (8, [7.861328125], 24.85156036261068),
    (9, [7.861328125], 24.85156036261068),
    (10, [7.861328125], 24.85156036261068),
    (11, [7.861328125], 24.85156036261068),
    (12, [7.861328125], 24.85156036261068),
    (13, [7.861328125], 24.85156036261068),
    (14, [7.861328125], 24.85156036261068),
    (15, [7.861328125], 24.85156036261068),
    (16, [7.861328125], 24.85156036261068),
    (17, [7.861328125], 24.85156036261068),
    (18, [7.861328125], 24.85156036261068),
    (19, [7.861328125], 24.85156036261068),
    (20, [7.861328125], 24.85156036261068),
    (21, [7.861328125], 24.85156036261068),
    (22, [7.861328125], 24.85156036261068),
    (23, [7.861328125], 24.85156036261068),
    (24, [7.861328125], 24.85156036261068),
    (25, [7.861328125], 24.85156036261068),
    (26, [7.861328125], 24.85156036261068),
    (27, [7.861328125], 24.85156036261068),
    (28, [7.861328125], 24.85156036261068),
    (29, [7.861328125], 24.85156036261068),
    (30, [7.861328125], 24.85156036261068),
    (31, [7.861328125], 24.85156036261068),
    (32, [7.861328125], 24.85156036261068),
    (33, [7.861328125], 24.85156036261068),
    (34, [7.861328125], 24.85156036261068),
    (35, [7.861328125], 24.85156036261068),
    (36, [7.861328125], 24.85156036261068),
    (37, [7.861328125], 24.85156036261068),
    (38, [7.861328125], 24.85156036261068),
    (39, [7.861328125], 24.85156036261068),
    (40, [7.861328125], 24.85156036261068),
    (41, [7.861328125], 24.85156036261068),
    (42, [7.861328125], 24.85156036261068),
    (43, [7.861328125], 24.85156036261068),
    (44, [7.861328125], 24.85156036261068),
    (45, [7.861328125], 24.85156036261068),
    (46, [7.861328125], 24.85156036261068),
    (47, [7.861328125], 24.85156036261068),
    (48, [7.861328125], 24.85156036261068),
    (49, [7.861328125], 24.85156036261068),
    (50, [7.861328125], 24.85156036261068),
    (51, [7.861328125], 24.85156036261068),
    (52, [7.861328125], 24.85156036261068),
    (53, [7.861328125], 24.85156036261068),
    (54, [7.861328125], 24.85156036261068),
    (55, [7.861328125], 24.85156036261068),
    (56, [7.861328125], 24.85156036261068),
    (57, [7.861328125], 24.85156036261068),
    (58, [7.861328125], 24.85156036261068),
    (59, [7.861328125], 24.85156036261068),
    (60, [7.861328125], 24.85156036261068),
    (61, [7.861328125], 24.85156036261068),
    (62, [7.861328125], 24.85156036261068),
    (63, [7.861328125], 24.85156036261068),
    (64, [7.85400390625], 24.85400381646418),
    (65, [7.85400390625], 24.85400381646418),
    (66, [7.85400390625], 24.85400381646418),
    (67, [7.85400390625], 24.85400381646418),
    (68, [7.85400390625], 24.85400381646418),
    (69, [7.855224609375], 24.85494496737466),
    (70, [7.855224609375], 24.85494496737466),
    (71, [7.8564453125], 24.855346706995128),
    (72, [7.8564453125], 24.855346706995128),
    (73, [7.8564453125], 24.855346706995128),
    (74, [7.8564453125], 24.855346706995128),
    (75, [7.8564453125], 24.855346706995128),
    (76, [7.8564453125], 24.855346706995128),
    (77, [7.8564453125], 24.855346706995128),
    (78, [7.8564453125], 24.855346706995128),
    (79, [7.8564453125], 24.855346706995128),
    (80, [7.8564453125], 24.855346706995128),
    (81, [7.8564453125], 24.855346706995128),
    (82, [7.8564453125], 24.855346706995128),
    (83, [7.8564453125], 24.855346706995128),
    (84, [7.8564453125], 24.855346706995128),
    (85, [7.8564453125], 24.855346706995128),
    (86, [7.8564453125], 24.855346706995128),
    (87, [7.8564453125], 24.855346706995128),
    (88, [7.8564453125], 24.855346706995128),
    (89, [7.8564453125], 24.855346706995128),
    (90, [7.8564453125], 24.855346706995128),
    (91, [7.8564453125], 24.855346706995128),
    (92, [7.8564453125], 24.855346706995128),
    (93, [7.8564453125], 24.855346706995128),
    (94, [7.8564453125], 24.855346706995128),
    (95, [7.8564453125], 24.855346706995128),
    (96, [7.8564453125], 24.855346706995128),
    (97, [7.8564453125], 24.855346706995128),
    (98, [7.8564453125], 24.855346706995128),
    (99, [7.8564453125], 24.855346706995128),
]

for i, (x,), y in best_fit:
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111)
    f = lambda x: x + 10 * sin(5 * x) + 7 * cos(4 * x)
    xs = np.linspace(0, 10, 1000)
    ys = [f(i) for i in xs]
    ax.plot(xs, ys)
    ax.scatter([x], [y], facecolor='r', s=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    fig.savefig('{}.png'.format(i))
    print('save {}.png'.format(i))
    plt.close(fig)
    
    
'''

'''
from math import sin, cos
import numpy as np
import matplotlib.pyplot as plt

steps, variants, fits = list(zip(*best_fit))
best_step, best_v, best_f = steps[-1], variants[-1][0], fits[-1]

fig = plt.figure()

ax = fig.add_subplot(211)
f = lambda x: x + 10*sin(5*x) + 7*cos(4*x)
x = np.linspace(0, 10, 1000)
y = [f(i) for i in x]
ax.plot(x, y)
ax.scatter([best_v], [best_f], facecolor='r')
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = fig.add_subplot(212)
ax.plot(steps, fits)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')

# Plot the maximum.
ax.scatter([best_step], [best_f], facecolor='r')
ax.annotate(s='x: {:.2f}\ny:{:.2f}'.format(best_v, best_f),
                                           xy=(best_step, best_f),
                                           xytext=(best_step-0.3, best_f-0.3))


plt.show()
'''

###########################################################################################
# 二维函数寻优，函数方程为 f(x) = y*sim(2*pi*x) + x*cos(2*pi*y)
###########################################################################################

'''
# 二维函数采用遗传算法寻优过程
from math import sin, cos, pi

from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

# Define population.
indv_template = BinaryIndividual(ranges=[(-2, 2), (-2, 2)], eps=0.001)
population = Population(indv_template=indv_template, size=50).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x, y = indv.solution
    return y*sin(2*pi*x) + x*cos(2*pi*y)

if '__main__' == __name__:
    engine.run(ng=100)
'''

'''
#画出进化曲线
import matplotlib.pyplot as plt

from best_fit import best_fit

steps, variants, fits = list(zip(*best_fit))
best_step, best_v, best_f = steps[-1], variants[-1], fits[-1]

fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(steps, fits)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')

# Plot the maximum.
ax.scatter([best_step], [best_f], facecolor='r')
ax.annotate(s='x: [{:.2f}, {:.2f}]\ny:{:.2f}'.format(*best_v, best_f),
                                                     xy=(best_step, best_f),
                                                     xytext=(best_step, best_f-0.1))


plt.show()
'''

'''
# 画出二维函数面
import os

import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt

from best_fit import best_fit

for i, (x, y), z in best_fit:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([x], [y], [z], zorder=99, c='r', s=100)

    x, y = np.mgrid[-2:2:100j, -2:2:100j]
    z = y*np.sin(2*np.pi*x) + x*np.cos(2*np.pi*y)
    ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap=plt.cm.bone_r)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if not os.path.exists('./surfaces'):
        os.mkdir('./surfaces')
    fig.savefig('./surfaces/{}.png'.format(i))
    print('save ./surfaces/{}.png'.format(i))
    plt.close(fig)
'''



###########################################################################################
# 二维函数寻优，函数方程为前两个函数
###########################################################################################


# 二维函数采用遗传算法寻优过程
from math import exp

from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitBigMutation

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput

# Define population.
indv_template = BinaryIndividual(ranges=[(0, 40), (0, 40), (0, 40), (0, 40), (0, 40)], eps=0.001)
population = Population(indv_template=indv_template, size=50).init()

# Create genetic operators.
#selection = RouletteWheelSelection()
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitBigMutation(pm=0.1, pbm=0.55, alpha=0.6)

# Create genetic algorithm engine.
# Here we pass all built-in analysis to engine constructor.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[ConsoleOutput, FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x1, x2, x3, x4, x5 = indv.solution
    # 输入相应的权重
    W1 = 0.5
    W2 = 0.25
    W3 = 0.25
    # 输入相应的需求强度、修建成本、区域内已有设施数量

    # alpha1i表示需求强度，系数与住宅区数量成正比
    alpha11 = 33
    alpha12 = 14
    alpha13 = 4
    alpha14 = 17
    alpha15 = 11

    # alpha2i表示修建成本，系数与绿化面积成反比
    alpha21 = 62
    alpha22 = 42
    alpha23 = 10
    alpha24 = 49
    alpha25 = 25

    # alpha3i表示已有设施影响程度，系数与区域内已有设施数量统计成正比
    alpha31 = 52
    alpha32 = 12
    alpha33 = 5
    alpha34 = 30
    alpha35 = 25

    x1 = 40 - x2 - x3 - x4 - x5
    x2 = 40 - x1 - x3 - x4 - x5
    x3 = 40 - x1 - x2 - x4 - x5
    x4 = 40 - x1 - x2 - x3 - x5
    x5 = 40 - x1 - x2 - x3 - x4

    y = W1 * (alpha11 * exp(-0.5 * x1) + alpha12 * exp(-0.5 * x2) + alpha13 * exp(-0.5 * x3) + alpha14 * exp(
        -0.5 * x4) + alpha15 * exp(-0.5 * x5)) + W2 * (
    alpha21 * exp(-0.5 * x1) + alpha22 * exp(-0.5 * x2) + alpha23 * exp(-0.5 * x3) + alpha24 * exp(
        -0.5 * x4) + alpha25 * exp(-0.5 * x5)) - W3 * (
    alpha31 * exp(-0.5 * x1) + alpha32 * exp(-0.5 * x2) + alpha33 * exp(-0.5 * x3) + alpha34 * exp(
        -0.5 * x4) + alpha35 * exp(-0.5 * x5))


    #y = 0.3*80*(exp(-0.5*x1)) - 0.3*30*(exp(-0.5*x1)) - 0.3*30*(exp(-0.5*x1)) + 0.3*60*(exp(-0.5*x2)) - 0.3*40*(exp(-0.5*x2)) - 0.3*70*(exp(-0.5*x2)) + 0.3*40*(exp(-0.5*x3)) - 0.3*30*(exp(-0.5*x3)) - 0.3*90*(exp(-0.5*x3))

    return y

if '__main__' == __name__:
    engine.run(ng=100)


import matplotlib.pyplot as plt

from best_fit import best_fit

steps, variants, fits = list(zip(*best_fit))
best_step, best_v, best_f = steps[-1], variants[-1], fits[-1]

fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(steps, fits)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')


# Plot the maximum.
ax.scatter([best_step], [best_f], facecolor='r')
#ax.annotate(s='x: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]\ny:{:.2f}'.format(*best_v, best_f), xy=(best_step, best_f), xytext=(best_step, best_f-0.1))


plt.show()



'''
# 二维函数采用遗传算法寻优过程
from math import exp

from math import sin, cos

from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore

# Define population.
indv_template = BinaryIndividual(ranges=[(0, 40)], eps=0.001)
population = Population(indv_template=indv_template, size=30).init()

# Create genetic operators.
selection = TournamentSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine.
engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])

# Define fitness function.
@engine.fitness_register
def fitness(indv):
    x, = indv.solution
    return 0.3*30*(exp(-0.5*x)) - 0.3*90*(exp(-0.5*x)) + 0.3*60*(1/(exp(-0.5*x)))

# Define on-the-fly analysis.
@engine.analysis_register
class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    interval = 1
    master_only = True

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmax)
        self.logger.info(msg)

    def finalize(self, population, engine):
        best_indv = population.best_indv(engine.fitness)
        x = best_indv.solution
        y = engine.ori_fmax
        msg = 'Optimal solution: ({}, {})'.format(x, y)
        self.logger.info(msg)

if '__main__' == __name__:
    # Run the GA engine.
    engine.run(ng=100)


import numpy as np
import matplotlib.pyplot as plt
from best_fit import best_fit

steps, variants, fits = list(zip(*best_fit))
best_step, best_v, best_f = steps[-1], variants[-1][0], fits[-1]

fig = plt.figure()

ax = fig.add_subplot(211)
f = lambda x: 0.3*30*(exp(-0.5*x)) - 0.3*90*(exp(-0.5*x)) + 0.3*60*(1/(exp(-0.5*x)))
x = np.linspace(0, 40, 1000)
y = [f(i) for i in x]
ax.plot(x, y)
ax.scatter([best_v], [best_f], facecolor='r')
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = fig.add_subplot(212)
ax.plot(steps, fits)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')

# Plot the maximum.
ax.scatter([best_step], [best_f], facecolor='r')
ax.annotate(s='x: {:.2f}\ny:{:.2f}'.format(best_v, best_f),
                                           xy=(best_step, best_f),
                                           xytext=(best_step-0.3, best_f-0.3))


plt.show()
'''