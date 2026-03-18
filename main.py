import numpy as np
 
class EWA:
    """Exponential Weights Algorithm (Hedge)."""
    def __init__(self, n_experts, lr=None):
        self.n=n_experts; self.w=np.ones(n_experts)/n_experts
        self.lr=lr; self.t=0
    def predict(self): return np.random.choice(self.n, p=self.w)
    def update(self, losses):
        self.t+=1; lr=self.lr if self.lr else np.sqrt(np.log(self.n)/self.t)
        self.w*=np.exp(-lr*losses); self.w/=self.w.sum()
 
class FollowTheRegularisedLeader:
    """FTRL with quadratic regulariser (equiv to online GD)."""
    def __init__(self, dim, lr=1.0):
        self.dim=dim; self.lr=lr; self.sum_grads=np.zeros(dim)
    def predict(self): return -self.lr*self.sum_grads/max(1,np.linalg.norm(self.sum_grads))
    def update(self, grad): self.sum_grads+=grad
 
T,n_experts,dim=500,5,10; np.random.seed(42)
ewa=EWA(n_experts,lr=0.1); ftrl=FTRL(dim)
ewa_loss=ftrl_loss=0
best_expert_loss=np.zeros(n_experts)
for t in range(T):
    expert_losses=np.random.rand(n_experts)
    e=ewa.predict(); ewa_loss+=expert_losses[e]; ewa.update(expert_losses)
    best_expert_loss+=expert_losses
    g=np.random.randn(dim); ftrl_loss+=np.random.rand(); ftrl.update(g)
 
print(f"EWA cumulative loss: {ewa_loss:.1f}")
print(f"Best expert loss:    {min(best_expert_loss):.1f}")
print(f"EWA regret: {ewa_loss-min(best_expert_loss):.1f}")
print(f"Theoretical O(sqrt(T*log n)): {np.sqrt(T*np.log(n_experts)):.1f}")
 
class FTRL:
    def __init__(self, dim, lr=1.0): self.s=np.zeros(dim); self.lr=lr
    def predict(self): return -self.lr*self.s
    def update(self, g): self.s+=g
