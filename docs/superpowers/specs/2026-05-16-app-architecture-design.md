# Diseño: App de simulación de línea de 3 estaciones

**Fecha:** 2026-05-16
**Tipo:** Spec de arquitectura (greenfield)
**Asignatura:** Simulación, Análisis y Diseño — Mini Reto III
**Fecha de entrega académica:** 16 de mayo

---

## 1. Contexto

Se va a construir una aplicación web interactiva que simula una línea de
ensamblaje de 3 estaciones en serie con buffers finitos, tiempos de proceso
exponenciales, bloqueo y hambre. El propósito es académico (entregable del
curso), pero el diseño se trata como un proyecto serio: modular, testeado y
desplegado.

El problema completo y los entregables del curso están en
`/Users/bryan-andagoya/Development/personal/3-station-line-project/assets/Mini Reto III-2026A.CL2.pdf`.
Resumen de las reglas del sistema:

- Tres estaciones E1, E2, E3 en serie. Buffers B1 (capacidad K1) entre
  E1-E2 y B2 (capacidad K2) entre E2-E3.
- Tiempos de proceso ~ Exponencial(λ=0.25) → media 4 min.
- **Bloqueo:** si Eᵢ termina y el buffer siguiente está lleno, Eᵢ se queda
  con la pieza hasta que se libere espacio.
- **Hambre:** si Eᵢ está libre y su buffer de entrada está vacío, espera.
- E1 nunca pasa hambre (input infinito), E3 nunca se bloquea (output infinito).
- El usuario debe elegir K1, K2 ∈ [1, 10] que optimicen throughput, flow time
  y WIP minimizando bloqueo y hambre.

## 2. Objetivos y no-objetivos

### Objetivos

- App Streamlit pública (Streamlit Community Cloud) con UI para configurar
  K1, K2, semilla, duración, número de réplicas y correr la simulación.
- Implementación propia de:
  - Generador Lineal Congruencial (parámetros Numerical Recipes).
  - Transformada inversa para tiempos exponenciales.
  - Simulación por eventos discretos (Future Event List + manejadores) con
    lógica explícita de bloqueo y hambre.
  - Estadística: intervalos de confianza, prueba t de hipótesis, pruebas de
    uniformidad χ² y K-S.
- Tests automatizados con pytest cubriendo invariantes críticos.
- Script CLI para generar el trace de Excel del entregable 2.2.
- CI con GitHub Actions.
- Funcionalidades extra: comparación de 2 configuraciones, exportar resultados
  a CSV, heatmap K1×K2.

### No-objetivos

- No usar SimPy ni librerías de DES (decisión explícita: lógica propia para
  defender en sustentación).
- No packaging como librería instalable (`pyproject.toml`).
- No autenticación ni persistencia entre sesiones (no es necesario).
- No tests del UI Streamlit ni del CLI (cubiertos por tests de los módulos
  subyacentes + verificación manual).
- No optimización extrema: legibilidad > microperformance, salvo el caso
  específico del heatmap.

## 3. Arquitectura

Enfoque "modular por capas". Streamlit es solo orquestación; toda la lógica
está en módulos puros y testeables.

```
3-station-line/
├── app.py                          # Entrypoint Streamlit (~150 líneas)
├── src/
│   ├── __init__.py
│   ├── rng.py                      # GLC + transformada inversa
│   ├── simulation.py               # DES core (FEL, manejadores, estado)
│   ├── statistics.py               # IC, t-test, χ², K-S
│   └── plotting.py                 # Figuras matplotlib
├── scripts/
│   └── excel_trace.py              # CLI para entregable 2.2 (.xlsx)
├── tests/
│   ├── test_rng.py
│   ├── test_simulation.py
│   └── test_statistics.py
├── docs/superpowers/specs/         # Specs (este documento)
├── .github/workflows/test.yml      # CI
├── .streamlit/config.toml          # Tema visual
├── requirements.txt                # Dependencias pinned
├── .gitignore
└── README.md
```

### Stack

| Categoría | Elección | Razón |
|---|---|---|
| Lenguaje | Python 3.11 | Compatible con Streamlit Cloud |
| UI | Streamlit 1.32 | Pedido por el PDF, deploy gratis |
| Gráficos | matplotlib | Estático, exportable a PDF para reporte |
| Stats helpers | scipy.stats | Solo para valores críticos (chi2, t) |
| Data | pandas | Tablas y export CSV |
| Excel | openpyxl | CLI del trace |
| Tests | pytest | Estándar |
| CI | GitHub Actions | Gratis, integrado con repo |
| Deploy | Streamlit Community Cloud | Gratis, 1 click |

## 4. Responsabilidades de los módulos

### `src/rng.py`

GLC sin estado global, expuesto como dataclass con método `next_uniform()`.

```python
@dataclass
class LCG:
    a: int = 1664525                # Numerical Recipes
    c: int = 1013904223
    m: int = 2**32
    state: int = 12345              # semilla

    def next_uniform(self) -> float:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

def exponential(rng: LCG, rate: float) -> float:
    """Transformada inversa. Excluye U=0 para evitar ln(0)."""
    u = rng.next_uniform()
    while u == 0.0:
        u = rng.next_uniform()
    return -math.log(u) / rate
```

**Justificación Hull-Dobell de los parámetros:**

1. `gcd(c, m) = gcd(1_013_904_223, 2³²) = 1` (c impar, m potencia de 2). ✓
2. Único factor primo de m: 2. `(a-1) = 1_664_524` es par. ✓
3. m divisible por 4. `(a-1)/4 = 416_131` (entero). ✓

→ Período = 2³² ≈ 4.29 × 10⁹.

### `src/simulation.py`

```python
@dataclass(frozen=True)
class SimulationConfig:
    k1: int
    k2: int
    rate: float = 0.25              # λ
    duration: float = 480.0
    seed: int = 12345

@dataclass
class SimulationResult:
    throughput: float                       # piezas/hora
    flow_times: list[float]                 # por pieza completada
    avg_wip: float
    pct_blocked_e1: float
    pct_blocked_e2: float
    pct_starved_e2: float
    pct_starved_e3: float
    busy_e1: float                          # para verificar invariante
    busy_e2: float
    busy_e3: float
    pieces_completed: int
    wip_trace: list[tuple[float, int]]      # (t, WIP) para evolución
    event_log: list[dict]                   # detalle evento por evento

def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Corre 1 réplica. Función pura: mismo config → mismo result."""
```

**Detalles internos:**
- FEL implementada con `heapq`. Tuplas `(t, prioridad, tipo, pieza_id)`.
- Prioridad para eventos simultáneos: FIN_E3 (0) > FIN_E2 (1) > FIN_E1 (2),
  procesando primero el de la salida para liberar bloqueos hacia atrás.
- Estados de estación: `Enum` con `FREE`, `BUSY`, `BLOCKED`.
- Tres manejadores privados `_handle_fin_e1`, `_handle_fin_e2`, `_handle_fin_e3`
  con la lógica de cascada descrita en la exploración previa (en el chat).
- Acumulación de estadísticas: antes de avanzar el reloj a `t_next`, se suma
  `(t_next - t_actual)` a los contadores de busy/blocked/starved según estado
  vigente, y se acumula `area_WIP += (B1+B2) * delta`.

### `src/statistics.py`

```python
def confidence_interval(values: list[float], confidence: float = 0.95
                        ) -> tuple[float, float, float]:
    """Retorna (media, lower, upper) usando t-Student."""

def t_test_greater(values: list[float], threshold: float
                   ) -> tuple[float, float]:
    """Retorna (t_stat, p_value) para H1: μ > threshold."""

def chi_square_uniformity(uniforms: list[float], k: int = 10
                          ) -> tuple[float, float, bool]:
    """Retorna (chi2_stat, critical_value_95, passes)."""

def ks_uniformity(uniforms: list[float]
                  ) -> tuple[float, float, bool]:
    """K-S test contra uniforme(0,1)."""
```

Cálculo manual donde sea pedagógicamente útil; scipy solo para valores
críticos (que requieren tablas).

### `src/plotting.py`

Cinco funciones que devuelven figuras matplotlib:

- `plot_wip_evolution(wip_trace)`
- `plot_flow_time_histogram(flow_times)`
- `plot_throughput_bar_with_ci(results)`
- `plot_heatmap(grid: dict[(int,int), float])`
- `plot_comparison(results_a, results_b)`

### `app.py`

Solo UI: lee inputs del sidebar, orquesta llamadas a `simulation` y
`statistics`, renderiza con widgets Streamlit y figuras de `plotting`.

### `scripts/excel_trace.py`

CLI:
```bash
python scripts/excel_trace.py --k1 3 --k2 3 --seed 12345 --duration 480 --out trace.xlsx
```

Genera xlsx con tres hojas: **Trace** (event log), **Métricas** (resumen),
**Inputs** (parámetros usados).

## 5. Flujo de datos

### Tab "Análisis"

```
SIDEBAR inputs → [Simular]
    ↓
Para i en range(n_replicas):
    cfg = SimulationConfig(k1, k2, ..., seed=base_seed + i*7919)
    results.append(run_simulation(cfg))
    ↓
AGREGACIÓN (statistics)
    throughputs = [r.throughput for r in results]
    ci_tp = confidence_interval(throughputs)
    t_stat, p = t_test_greater(throughputs, 12)
    ...
    ↓
RENDER
    st.metric(throughput, flow, WIP, %)
    st.pyplot(wip_evolution(seleccionable))
    st.pyplot(flow_time_histogram)
    st.dataframe(tabla por réplica) + download CSV
    expander "Validación GLC" con χ², K-S, tabla de Uᵢ
```

**Decorrelación de réplicas:** semilla por réplica `= base_seed + i * 7919`.
Multiplicar por un primo grande aleja las secuencias del GLC; usar solo
`base_seed + i` produciría primeros estados casi idénticos.

**Persistencia:** `st.session_state["results"]` para navegar entre tabs sin
re-simular. Botón "Simular" invalida.

### Tab "Comparación"

Dos sets de inputs (A vs B), se corre `n` réplicas de cada uno, se renderizan
métricas y gráficos lado a lado (overlay del WIP, bar chart comparativo de
throughput con IC).

### Tab "Heatmap K1×K2"

Para cada `(k1, k2)` en `{1..10} × {1..10}`, correr `n_replicas_heatmap` (default 3)
réplicas y promediar throughput. Mostrar heatmap con el máximo destacado.

**Optimizaciones:**
- `@st.cache_data` con clave `(seed, duration, n_replicas_per_cell)`.
- `event_log` se descarta para celdas del heatmap (solo se guarda throughput).
- Barra de progreso con `st.progress`.
- Confirmación previa con estimación de tiempo.

### CLI Excel

```
argparse → SimulationConfig
    ↓
run_simulation(cfg)
    ↓
result.event_log → openpyxl Workbook con hojas Trace/Métricas/Inputs
    ↓
save .xlsx
```

## 6. Layout de UI

- **Sidebar**: inputs K1, K2, semilla, duración, # réplicas, botón "Simular";
  expander "Sobre el modelo" con la teoría resumida.
- **Main**: 3 tabs — Análisis, Comparación, Heatmap K1×K2.

### Tab Análisis (vista principal)

1. Fila de `st.metric` con Throughput ± IC, Flow time ± IC, WIP, Bloqueo+Hambre.
2. Resultado de la prueba de hipótesis t (H1: μ > 12 piezas/h).
3. Gráfico de evolución del WIP en el tiempo (réplica seleccionable).
4. Histograma de tiempos de flujo (réplica 1 por defecto).
5. Tabla por réplica (descargable como CSV).
6. Expander colapsado "Validación del GLC" con tabla de Uᵢ, histograma,
   resultados de χ² y K-S, y media de Tᵢ exponenciales.

### Tab Comparación

Dos columnas (A y B) con KPIs. Debajo, gráficos overlay.

### Tab Heatmap

Confirmación de tiempo estimado → barra de progreso → heatmap con máximo
marcado.

## 7. Testing

Filosofía: tests de **invariantes y propiedades**, no de outputs exactos.

### `tests/test_rng.py`
- Reproducibilidad: misma semilla → misma secuencia.
- Smoke test contra valores X₁, X₂ calculados a mano con semilla 12345.
- Todos los U en `[0, 1)`.
- Media de 50k Tᵢ exponenciales dentro del 2% de 4.0 min.
- Tᵢ siempre `> 0`.

### `tests/test_simulation.py`
- Parametrizado sobre `(k1, k2, seed)`:
  - **Capacidad de buffer:** en todo evento, `B1 ≤ K1` y `B2 ≤ K2`.
  - **Conservación de piezas:** piezas iniciadas = completadas + en sistema.
  - **Suma de fracciones:** `busy + blocked + starved ≤ 1` por estación.
- E1 nunca pasa hambre; E3 nunca se bloquea.
- Reproducibilidad bit-a-bit: mismo config → mismo result.
- Throughput < 15.5 piezas/h (límite teórico ≈ 15).
- Promediando 20 réplicas: `throughput(K=10,10) > throughput(K=1,1)`.
- Caso degenerado `K1=K2=1`: no crashea, `throughput > 0`, `pct_blocked > 0`.

### `tests/test_statistics.py`
- IC sobre datos constantes → media exacta.
- IC del 95% sobre `normal(10, 1)` con N=100 → `[~9.8, ~10.2]`.
- `t_test_greater` rechaza claramente cuando valores son todos > threshold.
- χ² pasa sobre 10k uniformes del GLC con semilla fija.

### Cobertura objetivo
≥80% en `src/`, excluyendo `plotting.py` (verificación visual).

### Comandos
```bash
pytest tests/ -v
pytest tests/test_simulation.py -k "invariant or never or conservation" -v
pytest tests/ --cov=src --cov-report=term-missing
```

## 8. Manejo de errores

### Validación de inputs
Función `validate_inputs(...)` en `app.py` antes de simular. Errores se
muestran con `st.error()`. Rangos:
- `K1, K2 ∈ [1, 10]`
- `duration ≥ 500` (mínimo PDF), `duration ≤ 50_000` (evitar congelar navegador)
- `n_replicas ∈ [1, 20]`
- `seed ∈ [0, 2³¹]`

### Casos borde en simulación

| Caso | Manejo |
|---|---|
| Duración corta → 0 piezas completadas | `flow_times = []`. En statistics: `return None`. UI muestra "—". |
| `U = 0` en `exponential()` | Loop interno descarta y resamplea (en `rng.py`). |
| FEL vacía antes de `T_max` | `assert` que falla loud — indica bug, no caso normal. |

### Operaciones largas (heatmap)
- `st.progress` con texto del progreso `(i+1)/100`.
- `@st.cache_data` para no recomputar.
- Confirmación previa con estimación de tiempo.

### Filosofía
- **try/except solo en el borde con el usuario.** Internamente fail fast.
- No silenciar `ImportError`, `ModuleNotFoundError`, ni excepciones de
  matplotlib — son bugs reales que merecen visibilidad.

## 9. Deployment

### Stack de deploy
- **Repo público en GitHub** (requisito de Streamlit Community Cloud).
- **`requirements.txt` con versiones pinned** para evitar drift.
- **`.streamlit/config.toml`** con tema visual.
- **GitHub Actions** corriendo pytest en cada push.
- **Streamlit Community Cloud** apuntando al repo, rama `main`, archivo `app.py`.

### `requirements.txt`
```
streamlit==1.32.0
numpy==1.26.4
scipy==1.12.0
matplotlib==3.8.3
pandas==2.2.1
openpyxl==3.1.2
pytest==8.1.1
```

### `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 10
enableCORS = false

[browser]
gatherUsageStats = false
```

### Consideraciones de Streamlit Cloud
- **Cold start ~30-60s** tras 7 días sin uso. Documentar en README.
- **RAM 1GB**: el heatmap no debe guardar event_log por celda.
- **CPU compartida**: usar barras de progreso honestas.
- **No persistencia**: la reproducibilidad por semilla cubre esto.

### Checklist de release

- [ ] Repo en GitHub público con `.gitignore` correcto
- [ ] `pip install -r requirements.txt` limpio en venv nuevo
- [ ] `streamlit run app.py` sin errores
- [ ] `pytest tests/ -v` todo verde
- [ ] CI en verde (badge en README)
- [ ] `python scripts/excel_trace.py --k1 3 --k2 3 ...` genera xlsx válido
- [ ] Deploy en Streamlit Cloud verificado
- [ ] URL final probada en navegador limpio (sin cache, sin login)
- [ ] Cold-start probado: app responde dentro de ~60s tras inactividad
- [ ] README con badges, link a app, instrucciones, sección "Uso de IA"

## 10. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Bug en cascada de bloqueo/desbloqueo (E3 libera → E2 → E1) | Media | Alto (resultados incorrectos) | Tests parametrizados de invariantes (B≤K, conservación de piezas). |
| Heatmap demasiado lento en Streamlit Cloud | Media | Medio (mala UX) | Cache + reducir réplicas por celda + estimación previa. |
| Excel pierde precisión con números grandes del GLC | Alta | Bajo (solo afecta entregable 2.2) | Excel CLI corre el GLC en Python y exporta solo los U/T resultantes. |
| Cold-start del evaluador → mala primera impresión | Alta | Bajo | Documentado en README; visitar la app antes de la sustentación. |
| Equipo no usa git → conflictos al mergear | Depende | Alto | Brief inicial sobre branches y PRs (fuera del scope de este spec). |

## 11. Preguntas abiertas

- **¿Inicializar `git init` en el directorio del proyecto?** Actualmente no es
  un repo. Streamlit Cloud requiere GitHub, así que sí, pero confirmar con el
  usuario antes de hacerlo automáticamente.
- **¿Versiones de Python soportadas?** Por defecto 3.11 (la que Streamlit
  Cloud usa por default). Si el equipo tiene 3.10 local, igual sirve.
- **¿Usar `numpy` para vectorizar la generación de uniformes?** El PDF pide
  GLC propio pero no prohíbe vectorización. Decisión actual: mantener el loop
  escalar para legibilidad y mantener `state` correcto entre llamadas; usar
  numpy solo en estadísticas posteriores (medias, varianzas).

## 12. Siguientes pasos

Tras aprobación de este spec:

1. Invocar `superpowers:writing-plans` para generar plan de implementación
   con tareas concretas en orden.
2. Inicializar git repo (decisión del usuario).
3. Implementar en orden: `rng.py` → `simulation.py` → `statistics.py` →
   tests → CLI Excel → `plotting.py` → `app.py` → CI → deploy.
