# Metodología y Decisiones - Modelado de Riesgo de Crédito (Punto 2)

## 1. Definición de Incumplimiento y Poblaciones (PD vs. LGD)

### 1.1 Codificación del Incumplimiento en Clasificación (PD — Punto 1)
En concordancia estricta con el **Punto 1** y los estándares regulatorios de Basilea (mora superior a 90 días o inviabilidad de cobro), la variable objetivo binaria de incumplimiento ($y$) se define como:
- **1 (Incumplimiento / Malo):** Créditos clasificados como `Charged Off`, `Default`, `Late (31-120 days)` y `Does not meet the credit policy. Status:Charged Off`.
- **0 (Cumplimiento / Bueno):** Créditos clasificados como `Fully Paid` y `Does not meet the credit policy. Status:Fully Paid`.

Los estados de resultado aún no definitivo o no madurado (`Current`, `Issued`, `In Grace Period` y `Late (16-30 days)`) se excluyen en la etapa de modelado de PD por presentar censura a la derecha.

### 1.2 Muestra de Entrenamiento para Severidad (LGD — Punto 2)
Para el modelo de severidad o **Loss Given Default (LGD)**:
- **Población para la regresión:** Se entrena **únicamente sobre el subconjunto de créditos castigados o en default definitivo** (`Charged Off` y `Default`, totalizando 46.467 registros en la base global).
- **Justificación metodológica:** Conceptualmente, la LGD es condicional a la materialización completa del impago ($\mathbb{E}[\mathrm{LGD} \mid \mathrm{Default} = 1]$). Los créditos pagados (`Fully Paid`) no entran al modelo de severidad pues su riesgo se modela en PD. Los créditos `Late (31-120 days)`, aunque clasificados como malos en PD, aún se encuentran en gestión temprana de cobro y no han culminado su proceso de recuperación ni de castigo contable.

---

## 2. Construcción de la Variable Objetivo LGD

### 2.1 Variables de la Base de Datos
1. **`funded_amnt`**: Monto financiado efectivamente desembolsado (base de la exposición inicial $EAD_{proxy}$).
2. **`total_rec_prncp`**: Capital principal efectivamente amortizado y recuperado por la entidad a la fecha.
3. **`recoveries` y `collection_recovery_fee`**: Recuperaciones posteriores al castigo y costos judiciales/extrajudiciales de cobranza.

#### Análisis sobre la variable `out_prncp` (Capital pendiente):
En los créditos **`Charged Off`** (45.248 créditos, 97,4% de los incumplidos), `out_prncp` es **0.00 en el 100% de los casos**. Al castigar el crédito como incobrable (*write-off*), el saldo pendiente se traslada formalmente a cuentas de orden fuera de balance. Por tanto, **no se puede utilizar `out_prncp` para la LGD ni para la EAD**, utilizándose en su lugar la base de capital financiado y recuperado.

### 2.2 Formulación Matemática Implementada (LGD Proxy de Capital)
En el código del proyecto (`scripts/parte2_lgd/preparar_lgd.py` y modelos neuronales), la severidad observada se modela a partir de la tasa de recuperación del capital principal prestado:

$$\mathrm{RR}_{\mathrm{principal}} = \frac{\texttt{total\_rec\_prncp}}{\texttt{funded\_amnt}}$$

$$\mathrm{LGD}_{\mathrm{proxy}} = 1 - \mathrm{RR}_{\mathrm{principal}} = \frac{\texttt{funded\_amnt} - \texttt{total\_rec\_prncp}}{\texttt{funded\_amnt}}$$

> [!NOTE]
> **Relación con la fórmula financiera extendida:** En la literatura teórica de riesgo, la tasa de recuperación total puede incluir recuperaciones brutas ex-post netas de gastos de cobranza:
> $$RR_{\mathrm{teorico}} = \frac{\texttt{total\_rec\_prncp} + (\texttt{recoveries} - \texttt{collection\_recovery\_fee})}{\texttt{funded\_amnt}}$$
> No obstante, la métrica estándar y reproducible adoptada en este proyecto para la regresión neuronal acotada en $[0, 1]$ es el **LGD proxy sobre capital no recuperado** ($\mathrm{LGD}_{\mathrm{proxy}} = 1 - \frac{\texttt{total\_rec\_prncp}}{\texttt{funded\_amnt}}$), garantizando consistencia dimensional con la EAD definida como capital desembolsado (`funded_amnt`) y evitando distorsiones por flujos de cobranza externa tardíos.

### 2.3 Decisiones Metodológicas Adicionales
1. **Acotamiento al Soporte Unitario:** Dado que la severidad es una proporción, se verifica que $\mathrm{LGD} \in [0, 1]$. En la red neuronal, la función de salida sigmoide mapea naturalmente las predicciones a $(0, 1)$, superando la limitación del modelo OLS donde los valores predichos pueden desbordar el rango teórico.
2. **Prevención de Fuga de Información (*Data Leakage*):** Las variables de recuperación (`total_rec_prncp`, `recoveries`) son estrictamente *ex-post*. Solo se utilizan para calcular la etiqueta objetivo observada; **ninguna variable posterior al otorgamiento entra como predictora del modelo**.
3. **Agrupamiento por Grado (`grade`):** Mediante pruebas ANOVA y Tukey HSD sobre la muestra LGD, se demostró que las categorías de mayor riesgo `F` y `G` no presentan diferencias estadísticamente significativas entre sí ($p = 0{,}1424$), sustentando su consolidación funcional.
