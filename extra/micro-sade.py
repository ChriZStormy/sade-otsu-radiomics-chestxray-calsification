class uSADE_MultiOtsu:
    def __init__(self, D, NP=5, max_fes=3000, strategy='DE/rand/1', restart_iters=10, e=1, Fl=0.1, Fu=0.9, tau1=0.1, tau2=0.1):
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

    def optimize(self, image):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        total_pixels = image.size
        
        pop = np.sort(np.random.uniform(0, 255, (self.NP, self.D)), axis=1)
        fitness = np.array([calculate_fitness(ind, hist, total_pixels) for ind in pop])
        
        F = np.random.uniform(self.Fl, self.Fl + self.Fu, self.NP)
        Cr = np.random.rand(self.NP)
        
        fes = self.NP 
        t = 1 
        
        convergence_fitness = [np.min(fitness)]
        convergence_fes = [fes]
        
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
                    pop[i] = np.sort(np.random.uniform(0, 255, self.D))
                    fitness[i] = calculate_fitness(pop[i], hist, total_pixels)
                    fes += 1
                    F[i] = np.random.uniform(self.Fl, self.Fl + self.Fu)
                    Cr[i] = np.random.rand()
            
            best_idx = np.argmin(fitness)
            
            for i in range(self.NP):
                if fes >= self.max_fes: break 
                
                # ... código anterior ...
                idxs = [idx for idx in range(self.NP) if idx != i]
                np.random.shuffle(idxs)
                
                r1, r2, r3 = idxs[:3] 
                
                if self.strategy == 'DE/rand/1':
                    V = pop[r1] + F[i] * (pop[r2] - pop[r3])
                elif self.strategy == 'DE/best/1':
                    V = pop[best_idx] + F[i] * (pop[r1] - pop[r2])
                # ... código siguiente ...
                
                V = np.sort(np.clip(V, 0, 255))
                
                j_rand = np.random.randint(self.D)
                mask = (np.random.rand(self.D) <= Cr[i]) | (np.arange(self.D) == j_rand)
                U = np.where(mask, V, pop[i])
                
                U = np.sort(U) 
                
                f_U = calculate_fitness(U, hist, total_pixels)
                fes += 1 
                
                if f_U <= fitness[i]:
                    pop[i], fitness[i] = U, f_U
            t += 1
            convergence_fitness.append(np.min(fitness))
            convergence_fes.append(fes)
                
        best_idx = np.argmin(fitness)
        return np.round(pop[best_idx]).astype(int), fitness[best_idx], (convergence_fes, convergence_fitness)