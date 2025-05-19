from qiskit_aer import Aer
from qiskit_algorithms.utils import algorithm_globals

from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
from qiskit.algorithms.optimizers import COBYLA

# 1. Sabit seed
algorithm_globals.random_seed = 42

# 2. Optimizasyon problemi tanımı
problem = QuadraticProgram()
problem.binary_var('bor')        # x: bor yoğunluğu (0=düşük, 1=yüksek)
problem.binary_var('temp')       # y: sıcaklık (0=düşük, 1=yüksek)
problem.binary_var('neutron')    # z: nötron akısı (0=normal, 1=yüksek)

# 3. Maliyet fonksiyonu (minimize edilecek)
# Cost = 1*bor + 1.5*temp + 2*neutron + bor*neutron + temp*neutron
problem.minimize(
    linear=[1, 1.5,  2],
    quadratic={('bor', 'neutron'): 1, ('temp', 'neutron'): 1}
)

# 4. QAOA kur
optimizer = COBYLA()
qaoa = QAOA(optimizer=optimizer, reps=1, quantum_instance=Aer.get_backend('qasm_simulator'))

# 5. Çözümle
meo = MinimumEigenOptimizer(qaoa)
result = meo.solve(problem)

# 6. Sonuç
print("\n🧠 Optimal Settings for Reactor (Binary Form):")
print("bor (0=low, 1=high)     :", int(result.x[0]))
print("temp (0=low, 1=high)    :", int(result.x[1]))
print("neutron (0=low, 1=high) :", int(result.x[2]))
print("Minimum cost (risk score):", result.fval)