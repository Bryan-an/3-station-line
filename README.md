# Línea de 3 Estaciones — Simulación

Mini Reto III · Simulación, Análisis y Diseño · EPN.

App web interactiva que simula una línea de ensamblaje de 3 estaciones en serie
con buffers finitos, tiempos exponenciales, bloqueo y hambre.

## App desplegada

→ **https://3-station-line.streamlit.app**

> Si la app duerme tras inactividad, el primer load toma ~30-60s.

## Cómo correr local

```bash
git clone <repo-url>
cd 3-station-line
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

## Generar el Excel del entregable 2.2

```bash
python scripts/excel_trace.py --k1 3 --k2 3 --seed 12345 --duration 480 --out trace.xlsx
```

## Estructura

```
app.py                 # Streamlit UI
src/rng.py             # Linear Congruential Generator + exponencial
src/simulation.py      # Discrete event simulation (FEL + manejadores)
src/statistics.py      # IC, t-test, χ², K-S
src/plotting.py        # Figuras matplotlib
scripts/excel_trace.py # CLI para entregable 2.2
tests/                 # pytest
```

## Decisiones técnicas

- **GLC**: parámetros Numerical Recipes (a=1_664_525, c=1_013_904_223, m=2³²),
  verificados con Hull-Dobell (período = 2³²).
- **Tiempos exponenciales**: transformada inversa T = -ln(U) / λ con λ = 0.25.
- **DES**: Future Event List con `heapq`. Prioridad downstream-first
  (FIN_E3 > FIN_E2 > FIN_E1) para eventos simultáneos.
- **Bloqueo/hambre**: programados a mano (sin SimPy) para defenderlos en
  sustentación.

## Uso de IA

### Herramienta utilizada

**Claude Code** (Anthropic) — modelo Opus 4.7 — como agente de desarrollo
asistido. Fue la única herramienta de IA empleada en el proyecto. No se usaron
ChatGPT, Copilot, ni otras.

### Metodología

El trabajo siguió un flujo estructurado en cuatro fases dirigidas por el
equipo, donde la IA actuó como par técnico:

1. **Exploración conceptual**: Claude explicó los conceptos teóricos del
   problema (GLC, transformada inversa, distribución exponencial, simulación
   por eventos discretos, intervalos de confianza, prueba de hipótesis,
   pruebas de uniformidad χ² y K-S). El equipo validó cada concepto antes de
   avanzar.
2. **Brainstorming de arquitectura**: a través de preguntas guiadas, el equipo
   definió el stack (Python + Streamlit), la modularidad (`src/` con
   responsabilidades separadas), el alcance (incluyendo features opcionales
   como heatmap y comparación de configuraciones) y la estrategia de testing.
3. **Diseño formal (spec)**: se produjo un documento de diseño revisable
   guardado en `docs/superpowers/specs/`.
4. **Implementación TDD con revisión por pares**: Claude generó un plan de 24
   tareas y ejecutó cada una en ciclos `test → fail → implement → pass →
   commit`, con revisiones automáticas de compliance al spec y de calidad de
   código entre tareas.

### Tareas concretas en las que se usó

- Implementación completa de `src/rng.py`, `src/simulation.py`,
  `src/statistics.py`, `src/plotting.py`, `app.py`, `scripts/excel_trace.py`.
- Diseño y redacción de los 35 tests automatizados (invariantes de buffers,
  conservación de piezas, convergencia de la media exponencial, etc.).
- Configuración de CI con GitHub Actions y del tema de Streamlit.
- Justificación matemática de los parámetros del GLC mediante el teorema de
  Hull-Dobell.
- Generación del spec arquitectónico y del plan de implementación.
- Redacción de este README.

### Reflexión: qué funcionó, qué no

**Funcionó bien:**
- El enfoque TDD generó código correcto desde la primera iteración en la
  mayoría de los módulos.
- La separación por responsabilidades hizo que los tests fueran rápidos y
  específicos.
- La generación de la lógica matemática (Hull-Dobell, transformada inversa,
  fórmulas de IC y t-test) fue exacta y se verificó contra cálculo manual.

**Requirió corrección humana / iteración:**
- **Bug en snapshots transitorios de la simulación**: la primera
  implementación del DES (commit `c15be57`) producía snapshots intermedios
  donde los buffers transitoriamente excedían su capacidad durante cascadas de
  desbloqueo. El revisor de código (también Claude, pero con contexto
  independiente) lo detectó y se corrigió reordenando las operaciones
  `popleft` antes de `append` (commit `2aa205a`).
- **Mismatch de versión de Python**: el plan inicial asumía Python 3.11, pero
  la máquina local solo tenía 3.12. Se ajustó el plan, el CI y la
  documentación para usar 3.12 consistentemente.
- **Filename del CSV obsoleto**: el botón de descarga inicialmente usaba los
  valores de los sliders en vivo en lugar de los valores con que efectivamente
  se simuló. El code review lo detectó y se corrigió usando
  `st.session_state.config_used` (commit `19e58cb`).
- **Texto engañoso del warning del heatmap**: anunciaba "tarda 2-4 min" pero
  en la práctica tarda <30 segundos. Se ajustó.

La IA fue eficaz como herramienta de **aceleración y verificación cruzada**,
pero las decisiones de diseño, scope y prioridades fueron del equipo, y cada
output fue revisado antes de integrarse.

## Autores

- Carolina Albiño
- Melany Andagoya
- Johana Ortiz
- Sebastián Pérez
- Cristina Rocha
