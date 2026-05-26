import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Límites de búsqueda para la función Sphere
BOUND_MIN = -5.12
BOUND_MAX = 5.12

# Función Matemática Sphere: f(x) = sum(x_i^2)
def sphere(x):
    return np.sum(x**2)

class uSADE_Sphere:
    def __init__(self, D, NP=5, max_fes=300, strategy='DE/rand/1', restart_iters=10, e=1, Fl=0.1, Fu=0.9, tau1=0.1, tau2=0.1):
        self.D = D                      
        self.NP = NP                    
        self.max_fes = max_fes 
        self.strategy = strategy
        self.restart_iters = restart_iters 
        self.e = e                      
        self.Fl = Fl                    
        self.Fu = Fu                    
        self.tau1 = tau1
        self.tau2 = tau2
        
        # Guardaremos el historial de las poblaciones
        self.history_pop = [] 

    def optimize(self):
        pop = np.random.uniform(BOUND_MIN, BOUND_MAX, (self.NP, self.D))
        fitness = np.array([sphere(ind) for ind in pop])
        
        F = np.random.uniform(self.Fl, self.Fl + self.Fu, self.NP)
        Cr = np.random.rand(self.NP)
        
        fes = self.NP 
        t = 1 
        
        self.history_pop.append(pop.copy())
        
        while fes < self.max_fes:
            if t % self.restart_iters == 0:
                sort_idx = np.argsort(fitness)
                pop, fitness = pop[sort_idx], fitness[sort_idx]
                F, Cr = F[sort_idx], Cr[sort_idx]
                
                for i in range(self.e):
                    if np.random.rand() < self.tau1: F[i] = self.Fl + np.random.rand() * self.Fu
                    if np.random.rand() < self.tau2: Cr[i] = np.random.rand()
                        
                for i in range(self.e, self.NP):
                    if fes >= self.max_fes: break
                    pop[i] = np.random.uniform(BOUND_MIN, BOUND_MAX, self.D)
                    fitness[i] = sphere(pop[i])
                    fes += 1
                    F[i] = np.random.uniform(self.Fl, self.Fl + self.Fu)
                    Cr[i] = np.random.rand()
            
            best_idx = np.argmin(fitness)
            
            for i in range(self.NP):
                if fes >= self.max_fes: break 
                
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                r1, r2, r3 = idxs[:3] 
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F[i] * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F[i] * (pop[r1] - pop[r2])
                
                V = np.clip(V, BOUND_MIN, BOUND_MAX)
                
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= Cr[i]) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                
                f_U = sphere(U)
                fes += 1 
                
                if f_U <= fitness[i]:
                    pop[i], fitness[i] = U, f_U
            t += 1
            self.history_pop.append(pop.copy())
                
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]

def create_animation(history, strategy_name, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear malla de fondo para la función Sphere (2D)
    x = np.linspace(BOUND_MIN, BOUND_MAX, 200)
    y = np.linspace(BOUND_MIN, BOUND_MAX, 200)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    # Dibujar gráfico de calor (cmap 'magma' pone los valores bajos en color oscuro casi negro)
    c = ax.pcolormesh(X, Y, Z, cmap='magma', shading='auto', zorder=0)
    fig.colorbar(c, ax=ax, label='Valor de f(x)')
    
    # Individuos normales (blancos para que resalten sobre el mapa de calor oscuro/morado)
    scatter_normal = ax.scatter([], [], c='white', s=50, edgecolors='black', zorder=5, label='Individuos')
    # Mejor individuo (estrella cyan o verde claro)
    scatter_best = ax.scatter([], [], c='cyan', s=200, marker='*', edgecolors='black', zorder=6, label='Mejor')
    
    ax.set_title(f'$\mu$SADE ({strategy_name}) en la Función Sphere')
    ax.set_xlim(BOUND_MIN, BOUND_MAX)
    ax.set_ylim(BOUND_MIN, BOUND_MAX)
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    ax.legend(loc='upper right')
    
    def init():
        scatter_normal.set_offsets(np.empty((0, 2)))
        scatter_best.set_offsets(np.empty((0, 2)))
        return scatter_normal, scatter_best
        
    def animate(i):
        pop = history[i]
        fitnesses = [sphere(ind) for ind in pop]
        best_idx = np.argmin(fitnesses)
        
        best_ind = pop[best_idx, :2]
        others = np.delete(pop[:, :2], best_idx, axis=0)
        
        scatter_normal.set_offsets(others)
        scatter_best.set_offsets([best_ind])
        
        ax.set_title(f'$\mu$SADE ({strategy_name}) - Generación {i}\nMejor fitness: {fitnesses[best_idx]:.4f}')
        return scatter_normal, scatter_best
        
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=150, blit=True)
    
    print(f"Guardando {filename}...")
    ani.save(filename, writer='pillow', fps=5)
    plt.close()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prueba 1: DE/rand/1
    print("Iniciando optimización con DE/rand/1...")
    usade_rand = uSADE_Sphere(D=2, NP=5, max_fes=300, strategy='DE/rand/1')
    best_ind_rand, best_fit_rand = usade_rand.optimize()
    filename_rand = os.path.join(script_dir, 'usade_sphere_rand1.gif')
    create_animation(usade_rand.history_pop, 'DE/rand/1', filename_rand)
    
    # Prueba 2: DE/best/1
    print("Iniciando optimización con DE/best/1...")
    usade_best = uSADE_Sphere(D=2, NP=5, max_fes=300, strategy='DE/best/1')
    best_ind_best, best_fit_best = usade_best.optimize()
    filename_best = os.path.join(script_dir, 'usade_sphere_best1.gif')
    create_animation(usade_best.history_pop, 'DE/best/1', filename_best)
    
    print("¡Animaciones guardadas exitosamente en la carpeta extra!")
