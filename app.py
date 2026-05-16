# app.py
"""Streamlit UI for the 3-station line simulation."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.simulation import SimulationConfig, run_simulation
from src.statistics import (
    confidence_interval,
    t_test_greater,
    chi_square_uniformity,
    ks_uniformity,
)
from src.plotting import (
    plot_wip_evolution,
    plot_flow_time_histogram,
    plot_throughput_bar_with_ci,
    plot_heatmap,
    plot_comparison_wip,
)
from src.rng import LCG, exponential


SEED_PRIME = 7919  # decorrelates replication seeds


st.set_page_config(
    page_title="Línea de 3 Estaciones",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def validate_inputs(k1: int, k2: int, duration: float, n_replicas: int, seed: int) -> list[str]:
    errors = []
    if not (1 <= k1 <= 10):
        errors.append("K1 debe estar entre 1 y 10")
    if not (1 <= k2 <= 10):
        errors.append("K2 debe estar entre 1 y 10")
    if duration < 500:
        errors.append("Duración mínima: 500 minutos")
    if duration > 50_000:
        errors.append("Duración > 50_000 min puede congelar el navegador")
    if not (1 <= n_replicas <= 20):
        errors.append("Réplicas entre 1 y 20")
    if seed < 0 or seed >= 2 ** 31:
        errors.append("Semilla fuera de rango [0, 2³¹)")
    return errors


def run_replications(k1: int, k2: int, duration: float, n_replicas: int, base_seed: int):
    results = []
    progress = st.progress(0.0, text="Simulando...")
    for i in range(n_replicas):
        cfg = SimulationConfig(
            k1=k1, k2=k2, duration=duration, seed=base_seed + i * SEED_PRIME
        )
        results.append(run_simulation(cfg))
        progress.progress((i + 1) / n_replicas, text=f"Réplica {i+1}/{n_replicas}")
    progress.empty()
    return results


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Parámetros")
    k1 = st.slider("K1 (buffer entre E1 y E2)", 1, 10, 3)
    k2 = st.slider("K2 (buffer entre E2 y E3)", 1, 10, 3)
    seed = st.number_input("Semilla GLC", min_value=0, max_value=2**31 - 1, value=12_345, step=1)
    duration = st.number_input("Duración (min)", min_value=500, max_value=50_000, value=960, step=60)
    n_replicas = st.slider("Número de réplicas", 1, 20, 10)
    run_button = st.button("▶ Simular", type="primary", use_container_width=True)

    with st.expander("ℹ️ Sobre el modelo"):
        st.markdown(
            """
            **Línea de 3 estaciones en serie** con tiempos exponenciales (media 4 min),
            buffers finitos K1 y K2, bloqueo y hambre.

            - **E1** nunca pasa hambre (input infinito).
            - **E3** nunca se bloquea (output infinito).
            - **Bloqueo:** estación termina pero buffer siguiente lleno → espera.
            - **Hambre:** estación libre pero buffer anterior vacío → espera.

            GLC: Numerical Recipes (a=1_664_525, c=1_013_904_223, m=2³²).
            """
        )

# --- Main area ---
st.title("🏭 Simulación: Línea de 3 Estaciones")

tab_analysis, tab_compare, tab_heatmap = st.tabs(
    ["📊 Análisis", "🔀 Comparación", "🔥 Heatmap K1×K2"]
)

# Persist results across reruns
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.config_used = None

if run_button:
    errors = validate_inputs(k1, k2, duration, n_replicas, seed)
    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Corriendo réplicas..."):
            st.session_state.results = run_replications(k1, k2, duration, n_replicas, seed)
            st.session_state.config_used = dict(
                k1=k1, k2=k2, duration=duration, n_replicas=n_replicas, seed=seed
            )

def render_metrics_and_hypothesis(results):
    throughputs = [r.throughput for r in results]
    flow_means = [
        sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0
        for r in results
    ]
    wips = [r.avg_wip for r in results]
    blocked_e1 = [r.pct_blocked_e1 for r in results]
    blocked_e2 = [r.pct_blocked_e2 for r in results]
    starved_e2 = [r.pct_starved_e2 for r in results]
    starved_e3 = [r.pct_starved_e3 for r in results]

    n = len(throughputs)
    if n >= 2:
        tp_mean, tp_lo, tp_hi = confidence_interval(throughputs)
        ft_mean, ft_lo, ft_hi = confidence_interval(flow_means)
    else:
        tp_mean, tp_lo, tp_hi = throughputs[0], throughputs[0], throughputs[0]
        ft_mean, ft_lo, ft_hi = flow_means[0], flow_means[0], flow_means[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Throughput (piezas/h)",
        f"{tp_mean:.2f}",
        delta=f"IC95 [{tp_lo:.2f}, {tp_hi:.2f}]",
        delta_color="off",
    )
    c2.metric(
        "Tiempo de flujo (min)",
        f"{ft_mean:.2f}",
        delta=f"IC95 [{ft_lo:.2f}, {ft_hi:.2f}]",
        delta_color="off",
    )
    c3.metric("WIP promedio", f"{sum(wips)/n:.2f}")
    total_block_starve = (
        sum(blocked_e1) + sum(blocked_e2) + sum(starved_e2) + sum(starved_e3)
    ) / n
    c4.metric("Bloqueo + Hambre", f"{total_block_starve * 100:.1f}%")

    st.subheader("Prueba de hipótesis  ·  H₁: μ throughput > 12 piezas/h")
    if n >= 2:
        t_stat, p_val = t_test_greater(throughputs, threshold=12.0)
        decision = "🟢 Rechazo H₀" if p_val < 0.05 else "🔴 No rechazo H₀"
        st.write(
            f"t = **{t_stat:.3f}**  ·  p-value = **{p_val:.4g}**  ·  {decision}  (α = 0.05)"
        )
    else:
        st.info("Se necesitan ≥ 2 réplicas para la prueba t.")

    return throughputs, flow_means, wips


with tab_analysis:
    if st.session_state.results is None:
        st.info("Configura los parámetros y pulsa **Simular** en la barra lateral.")
    else:
        results = st.session_state.results
        throughputs, flow_means, wips = render_metrics_and_hypothesis(results)
        st.session_state._cached_throughputs = throughputs
        st.session_state._cached_flow_means = flow_means

        # --- Plots ---
        st.subheader("📈 Evolución del WIP")
        rep_idx = st.selectbox(
            "Réplica a visualizar",
            options=list(range(1, len(results) + 1)),
            index=0,
            key="wip_rep_idx",
        )
        st.pyplot(plot_wip_evolution(results[rep_idx - 1].wip_trace))

        st.subheader("📊 Histograma de tiempos de flujo")
        st.pyplot(plot_flow_time_histogram(results[rep_idx - 1].flow_times))

        st.subheader("📐 Throughput por réplica")
        if len(throughputs) >= 2:
            ci = confidence_interval(throughputs)
        else:
            ci = (throughputs[0], throughputs[0], throughputs[0])
        st.pyplot(plot_throughput_bar_with_ci(throughputs, ci=ci))

        st.subheader("📋 Resultados por réplica")
        df = pd.DataFrame([
            {
                "Réplica": i + 1,
                "Throughput (piezas/h)": round(r.throughput, 3),
                "Flow time prom. (min)": round(
                    sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0, 3
                ),
                "WIP prom.": round(r.avg_wip, 3),
                "% Bloqueo E1": round(r.pct_blocked_e1 * 100, 2),
                "% Bloqueo E2": round(r.pct_blocked_e2 * 100, 2),
                "% Hambre E2": round(r.pct_starved_e2 * 100, 2),
                "% Hambre E3": round(r.pct_starved_e3 * 100, 2),
                "Piezas completadas": r.pieces_completed,
            }
            for i, r in enumerate(results)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Descargar CSV",
            data=csv_bytes,
            file_name=f"resultados_K1{k1}_K2{k2}_seed{seed}.csv",
            mime="text/csv",
        )

with tab_compare:
    st.write("Comparación a poblar en Task 5.6")

with tab_heatmap:
    st.write("Heatmap a poblar en Task 5.7")
