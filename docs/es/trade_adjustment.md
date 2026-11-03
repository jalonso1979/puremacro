> 🇬🇧 [English](../trade_adjustment.md) · 🇪🇸 Español

# Abastecimiento dinámico condicionado

`solve_sourcing_adjustment` elige cantidades por origen para producir una unidad
de un compuesto CES en cada fecha, dados precios de compra exógenos. Minimiza

$$\sum_{t=0}^{T-1}\delta^t\left[p_t'q_t+
\frac{\kappa}{2}\sum_o\frac{(q_{to}-q_{t-1,o})^2}{\alpha_o}\right]$$

sujeto a

$$F(q_t)=\left[\sum_o\alpha_o(q_{to}/\alpha_o)^{(\sigma-1)/\sigma}
\right]^{\sigma/(\sigma-1)}\ge1.$$

Para `sigma=1` se utiliza el límite Cobb–Douglas. Las participaciones son
positivas y suman uno. Los precios iniciales son uno y las cantidades iniciales
coinciden por defecto con las participaciones. La tecnología es cóncava y
el problema de costos es convexo. Se permite desechar producción excedente.
Después de la última fecha no existe requisito de producción ni costo de
ajuste; el horizonte es parte del modelo.

```python
import pandas as pd
from puremacro.trade import solve_sourcing_adjustment

prices = pd.DataFrame({"nacional": [1., 1., 1., 1.],
                       "importado": [1., 1.3, 1.3, 1.]})
result = solve_sourcing_adjustment(
    prices, pd.Series({"nacional": .7, "importado": .3}),
    elasticity=3., adjustment_cost=2., discount=.99,
)
print(result.accounts)
print(result.diagnostics)
```

Las cuentas separan compras, recursos de ajuste, costo total, producción e
índice CES sin fricciones. La aceptación exige factibilidad, estacionariedad
KKT reconstruida y complementariedad. Una optimización fallida o un límite
artificial de cantidades sin resolver genera un error.

Este mecanismo no determina salarios, producción nacional, balanza comercial,
movilidad laboral ni reciclaje de la recaudación arancelaria. Trasladar costos
unitarios a precios al consumidor es un supuesto explícito del
[estudio de hogares](household_policy_studies.md). El
[equilibrio MRIO dinámico](trade_dynamic.md) tiene otro cierre económico.
