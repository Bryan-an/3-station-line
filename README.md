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

Esta sección documenta el uso de herramientas de IA durante el desarrollo:

- [ ] Listar herramientas (ChatGPT, Claude, Copilot, etc.)
- [ ] Listar tareas concretas en las que se usó
- [ ] Reflexión: qué funcionó, qué no, ejemplos de outputs corregidos

## Autores

- [Nombre 1]
- [Nombre 2]
- ...
