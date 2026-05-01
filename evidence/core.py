# 文件路径: evidence/core.py
import torch

# ==============================================================================
# Jeffrey & Wandelt (2023) Evidence Network Core Math
# ==============================================================================

def l_pop_transform(x, alpha=2.0):
    """
    Eq (27): J_alpha(x) = x + x * |x|^(alpha-1)
    Transforms the raw network output f(x) into Log Bayes Factor space.
    """
    return x + x * torch.abs(x).pow(alpha - 1)

def one_pop_exponential_loss(f_x, targets, alpha=2.0, c=0.0):
    """
    Eq (28): V(f(x), m) = exp( (0.5 - m) * J(f(x)) )
    
    Args:
        f_x: Raw output from the neural network [Batch, 1]
        targets: Binary labels (0 or 1) [Batch, 1]
        alpha: l-POP hyperparameter (default 2.0)
    """
    # Ensure targets match f_x shape
    if targets.shape != f_x.shape:
        targets = targets.view_as(f_x)
        
    J_val = l_pop_transform(f_x, alpha)
    
    # Calculate the exponent term
    # If target=1 (Model 1), term = -0.5 * J
    # If target=0 (Model 0), term = +0.5 * J
    term = (0.5 - targets) * (J_val+ c)
    
    # === SAFETY CLAMP ===
    # Prevents exp(88) -> inf -> NaN. 
    # This is a numerical stability fix. Clamping at +/- 20 is safe
    # because e^20 is huge enough to drive gradients effectively.
    term = torch.clamp(term, min=-20.0, max=10.0)
    
    loss = torch.exp(term)
    return torch.mean(loss)

def compute_posterior(f_x, alpha=2.0,c=0.0):
    """
    Eq (34): p(M1|x) = Sigmoid( J(f(x)) )
    
    Used for:
    1. Calculating Accuracy
    2. Plotting Histograms
    3. Coverage Tests
    """
    J_val = l_pop_transform(f_x, alpha)
    return torch.sigmoid(J_val + c)