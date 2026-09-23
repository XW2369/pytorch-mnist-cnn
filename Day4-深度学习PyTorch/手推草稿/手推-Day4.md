δ₂ = dL/dz₂ = 2*(z₂ - y)/(n * d_out)
→ dW₂ = a₁ᵀ @ δ₂
→ db₂ = δ₂.sum(axis=0)
→ da₁ = δ₂ @ W₂ᵀ
→ δ₁ = da₁ * (1 - a₁**2)
→ dW₁ = a₀ᵀ @ δ₁
→ db₁ = δ₁.sum(axis=0)
