# Metodología y Decisiones - Modelado de Riesgo de Crédito (Punto 2)

## 1. Filtrado para los créditos que incumplieron (Clasificación - PD)

### Construcción de la variable objetivo dicotómica
Se creó una variable denominada `incumplimiento`, definida de la siguiente manera:
- **1 (Incumplimiento):** créditos clasificados como `Charged Off` o `Default`.
- **0 (No incumplimiento):** créditos clasificados como `Fully Paid`.

Se excluyeron los estados `Current`, `In Grace Period`, `Late (16-30 days)` y `Late (31-120 days)`, debido a que corresponden a créditos cuyo resultado final todavía es incierto.

---

## 2. Construcción de la variable LGD (Punto de Regresión)

### 2.1 Definición y Población Objetivo
La severidad de la pérdida o **Loss Given Default (LGD)** representa el porcentaje del monto expuesto que la entidad financiera no logra recuperar una vez el deudor entra en incumplimiento.
- **Población para la regresión:** El modelo de regresión se entrena **únicamente sobre el subconjunto de créditos que incumplieron** (`incumplimiento == 1`, es decir, `loan_status` en `Charged Off` o `Default`, totalizando 46,467 registros en la base de datos). 
- **Justificación:** Conceptualmente, LGD es una variable condicional al incumplimiento ($E[\text{LGD} \mid \text{Default} = 1]$). Los créditos pagados (`Fully Paid`) no entran en este modelo de severidad, pues su probabilidad de no-pérdida se modela en la etapa de clasificación (PD).

### 2.2 Variables de la Base de Datos para el Cálculo de LGD
1. **`funded_amnt`**: Monto financiado (proxy de la Exposición al Incumplimiento o EAD base).
2. **`total_rec_prncp`**: Capital pagado a la fecha por el prestatario.
3. **`recoveries`**: Recuperaciones brutas posteriores al castigo (*post charge off gross recovery*).
4. **`collection_recovery_fee`**: Costos de cobranza judicial/extrajudicial a deducir de las recuperaciones.

#### Análisis sobre la variable `out_prncp` (Capital pendiente):
En los créditos **`Charged Off`** (45.248 créditos, 97.4% de los incumplidos), `out_prncp` es **0.00 en el 100% de los casos**. Al castigar el crédito como incobrable (*write-off*), el saldo pendiente se traslada a cuentas de orden. Por ello, **no se utiliza `out_prncp` para la LGD**, sino `total_rec_prncp` y `recoveries` netas.

### 2.3 Formulación Matemática de la LGD
Enfoque de pérdida neta sobre capital:
$$RR = \frac{\text{total\_rec\_prncp} + (\text{recoveries} - \text{collection\_recovery\_fee})}{\text{funded\_amnt}}$$

$$\text{LGD} = 1 - RR = \frac{\text{funded\_amnt} - \text{total\_rec\_prncp} - (\text{recoveries} - \text{collection\_recovery\_fee})}{\text{funded\_amnt}}$$

### 2.4 Decisiones Metodológicas Adicionales
1. **Acotamiento:** Truncamiento estricto a $[0, 1]$: $\text{LGD}_{\text{final}} = \min(\max(\text{LGD}, 0), 1)$.
2. **Fuga de información (*Data Leakage*):** Las variables de recuperación son *ex-post* y no se usan como predictoras; solo se usan variables disponibles a la originación.
3. **Significancia por Grado (`grade`):** Mediante ANOVA y Tukey HSD se colapsaron las categorías `F` y `G` en un único grupo `{F, G}` por no presentar diferencia significativa ($p = 0.1424$).
