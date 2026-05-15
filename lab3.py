import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

N = 1000

class LinearProgram:
    def __init__(self, target_coef, le_coef=None, le_limit=None, eq_coef=None, eq_limit=None, lower_bound=[0], upper_bound=None):
        self.target_coef = target_coef
        self.le_coef = le_coef
        self.le_limit = le_limit
        self.eq_coef = eq_coef
        self.eq_limit = eq_limit
        self.lower_bound = lower_bound
        if upper_bound == None:
            self.upper_bound = [None]
        else:
            self.upper_bound = upper_bound

    def __str__(self):
        return (f'''Linear Program:\n
            Target coefficients:\n {str(self.target_coef)}\n
            Lower than or equal to coefficients:\n {str(self.le_coef)}\n
            Lower than or equal to limits:\n {str(self.le_limit)}\n
            Equal to coefficients:\n {str(self.eq_coef)}\n
            Equal to limits:\n {str(self.eq_limit)}\n
            Lower bound:\n {str(self.lower_bound)}\n
            Upper bound:\n {str(self.upper_bound)}\n''')

    def solve(self):
        c = self.target_coef
        A_ub = self.le_coef
        b_ub = self.le_limit
        A_eq = self.eq_coef
        b_eq = self.eq_limit
        bounds = []
        for i in range(len(self.lower_bound)):
            bounds.append([self.lower_bound[i], self.upper_bound[i]])

        try:
            return linprog(
                c = c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq,
                b_eq = b_eq, bounds = bounds, method = 'highs')

        except:
            return "Error - Linprog not solvable"

def main():
    # Simple profit maximization
    simple_profit = simple_prob()
    result = simple_profit.solve()
    print(f'''Simple profit maximization:
        Variables:\n {result.x}\n
        Result:\n {np.sum(result.x * np.array([490, 640, 470])):.2f}\n''')
        # Values for profit function taken from report

    # Dual problem
    dual_problem = dual_prob()
    result = dual_problem.solve()
    print(f'''Dual problem:
        Variables:\n {result.x}\n
        Result:\n {np.sum(result.x * np.array([930, 800, -100, 200])):.2f}\n''')
        # Values for result taken directly from report

    # Stability analysis
    stab_analysis = stability_prob()
    print(stab_analysis)

    # Margin analysis
    margin_analysis = margin_prob()
    result = margin_analysis.solve()
    print(f'''Margin analysis
        Variables:\n {result.x}\n
        Result:\n {np.sum(result.x * np.array([490, 640, 470])):.2f}\n''')
        # Values for result taken directly from report

    # Statistical analysis
    stat_problems = stat_model()
    solutions = np.empty((N, 3))
    for i in range(len(stat_problems)):
        solutions[i] = stat_problems[i].solve().x

    counts, bins = np.histogram(solutions[:,0], bins=100)
    plt.stairs(counts, bins, label="Värden A", color="red", fill=True, zorder=10)
    counts, bins = np.histogram(solutions[:,1], bins=100)
    plt.stairs(counts, bins, label="Värden B", color="limegreen",fill=True)
    counts, bins = np.histogram(solutions[:,2], bins=100)
    plt.stairs(counts, bins, label="Värden C", color="blue", fill=True)
    plt.ylabel("Instanser av värde")
    plt.xlabel("Antal producerade enheter")
    plt.legend()
    plt.savefig(f"figures/{N}-iter-histogram.png", dpi=400)
    plt.clf
    
    print(f'''Means:\n
        Mean of A: {np.mean(solutions[:, 0]):.2f}\n
        Mean of B: {np.mean(solutions[:, 1]):.2f}\n
        Mean of C: {np.mean(solutions[:, 2]):.2f}\n
        ''')

    print(f'''Standard deviations:\n
        Standard deviation of A: {np.std(solutions[:, 0]):.2f}\n
        Standard deviation of B: {np.std(solutions[:, 1]):.2f}\n
        Standard deviation of C: {np.std(solutions[:, 2]):.2f}\n
        ''')

    # a = 0
    # b = 0
    # i = 0
    # for b in solutions[:, 1]:
    #     if b == 310:
    #         print(solutions[i, :])
    #     i = i + 1
    # print(f"= 200: {a}\n= 0: {b}")

def simple_prob():
    # Directly from report
    target_coef = np.array([
        -490,
        -640,
        -470
    ])
    le_coef = np.array([
        [2, 3, 2],
        [3, 1, 2],
        [0, -1, 0],
        [0, 0, 1]]
        )
    le_limit = np.array([
        930,
        800,
        -100,
        200
    ])

    return LinearProgram(target_coef = target_coef,
        le_coef = le_coef,
        le_limit = le_limit,)

def dual_prob():
    # Directly from report
    target_coef = np.array([
        930,
        800,
        -100,
        200
    ])
    le_coef = np.array([
        [-2, -3, 0, 0],
        [-3, -1, 1, 0],
        [-2, -2, 0, -1]
    ])
    le_limit = np.array([
        -490,
        -640,
        -470
    ])

    return LinearProgram(target_coef = target_coef,
        le_coef = le_coef,
        le_limit = le_limit,)

def stability_prob():
    baseline = simple_prob().solve()
    increasing = [False, True]
    limit_list = []
    costs = [(2 * 75 + 3 * 220 + 10), (3 * 75 + 1 * 220 + 10), (2 * 75 + 2 * 220 + 10)]
    
    for i in range(len(simple_prob().solve().x)):
        for is_increasing in increasing:
            curr_solution = baseline
            curr_program = simple_prob()
            while np.array_equal(curr_solution.x, baseline.x):
                if is_increasing:
                    # Switched signs
                    if not(curr_program.target_coef[i] <= -2000):
                        curr_program.target_coef[i] = curr_program.target_coef[i] - 1
                    else: 
                        print("Broken due to end of limit, upper")
                        break
                elif not(is_increasing):
                    # Switched signs
                    if not(curr_program.target_coef[i] >= 0):
                        curr_program.target_coef[i] = curr_program.target_coef[i] + 1
                    else:
                        print("Broken due to end of limit, lower")
                        break
                curr_solution = curr_program.solve()
            print(f'''Value {i} broke, compare solutions: {curr_solution.x}, {baseline.x}''')
            limit_list.append((-curr_program.target_coef[i]) + costs[i])

    return limit_list

def margin_prob():
    # Directly from report
    target_coef = np.array([
        -490,
        -640,
        -470
    ])
    le_coef = np.array([
        [2, 3, 2],
        [3, 1, 2],
        [0, -1, 0],
        [0, 0, 1]]
        )
    le_limit = np.array([
        930,
        (800 + 400),
        -100,
        200
    ])

    return LinearProgram(target_coef = target_coef,
        le_coef = le_coef,
        le_limit = le_limit,)

def stat_model():
    init_problem = simple_prob()

    # Get random values for A, B, C
    rng = np.random.default_rng()
    prices_A = rng.normal(loc = 1310, scale = 100, size = N)
    prices_B = rng.normal(loc = 1095, scale = 50, size = N)
    prices_C = rng.normal(loc = 1070, scale = 30, size = N)

    linear_programs = [None] * N

    for i in range(N):
        target_coef = np.array([
        -(prices_A[i] - 2 * 75 - 3 * 220 - 10),
        -(prices_B[i] - 3 * 75 - 1 * 220 - 10),
        -(prices_C[i] - 2 * 75 - 2 * 220 - 10)
        ])
        linear_programs[i] = LinearProgram(
            target_coef = target_coef,
            le_coef = init_problem.le_coef,
            le_limit = init_problem.le_limit)

    return linear_programs

main()