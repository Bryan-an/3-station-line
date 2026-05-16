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

        with st.expander("🎲 Validación del GLC"):
            n_u = 1_000
            val_seed = st.session_state.config_used["seed"]
            rng_val = LCG(state=val_seed)
            uniforms = [rng_val.next_uniform() for _ in range(n_u)]
            rng_exp = LCG(state=val_seed)
            ts = [exponential(rng_exp, 0.25) for _ in range(n_u)]

            chi2, crit_chi, pass_chi = chi_square_uniformity(uniforms, k=10)
            d, crit_ks, pass_ks = ks_uniformity(uniforms)
            mean_t = sum(ts) / n_u

            colu1, colu2, colu3 = st.columns(3)
            colu1.metric("χ²", f"{chi2:.2f}", delta=f"crítico {crit_chi:.2f}",
                         delta_color="normal" if pass_chi else "inverse")
            colu2.metric("K-S D", f"{d:.4f}", delta=f"crítico {crit_ks:.4f}",
                         delta_color="normal" if pass_ks else "inverse")
            colu3.metric("Media T exp.", f"{mean_t:.3f} min", delta="esperado 4.00", delta_color="off")

            st.write("Primeros 30 valores Uᵢ generados:")
            st.dataframe(
                pd.DataFrame({"i": range(1, 31), "Uᵢ": uniforms[:30]}),
                use_container_width=True,
                hide_index=True,
            )

            st.write("Histograma de 1000 Uᵢ (debe verse plano):")
            import matplotlib.pyplot as plt
            fig_u, ax_u = plt.subplots(figsize=(9, 2.5))
            ax_u.hist(uniforms, bins=20, edgecolor="black", alpha=0.7)
            ax_u.set_xlabel("U")
            ax_u.set_ylabel("Frecuencia")
            fig_u.tight_layout()
            st.pyplot(fig_u)

with tab_compare:
    st.subheader("🔀 Comparar dos configuraciones (K1, K2)")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Configuración A**")
        ka1 = st.slider("K1 (A)", 1, 10, 2, key="ka1")
        ka2 = st.slider("K2 (A)", 1, 10, 2, key="ka2")
    with col_b:
        st.markdown("**Configuración B**")
        kb1 = st.slider("K1 (B)", 1, 10, 5, key="kb1")
        kb2 = st.slider("K2 (B)", 1, 10, 5, key="kb2")

    n_cmp = st.slider("Réplicas por configuración", 1, 20, 5, key="n_cmp")
    cmp_seed = st.number_input(
        "Semilla base", min_value=0, max_value=2**31 - 1, value=42, key="cmp_seed"
    )
    run_compare = st.button("▶ Comparar", key="run_compare")

    if run_compare:
        with st.spinner("Simulando A..."):
            res_a = run_replications(ka1, ka2, 960.0, n_cmp, cmp_seed)
        with st.spinner("Simulando B..."):
            res_b = run_replications(kb1, kb2, 960.0, n_cmp, cmp_seed)
        st.session_state.cmp = (res_a, res_b, ka1, ka2, kb1, kb2)

    if "cmp" in st.session_state:
        res_a, res_b, ka1_, ka2_, kb1_, kb2_ = st.session_state.cmp
        tp_a = [r.throughput for r in res_a]
        tp_b = [r.throughput for r in res_b]
        wip_a = sum(r.avg_wip for r in res_a) / len(res_a)
        wip_b = sum(r.avg_wip for r in res_b) / len(res_b)
        flow_a = [sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0 for r in res_a]
        flow_b = [sum(r.flow_times) / len(r.flow_times) if r.flow_times else 0.0 for r in res_b]

        ca, cb = st.columns(2)
        with ca:
            st.markdown(f"**A: K1={ka1_}, K2={ka2_}**")
            st.metric("Throughput", f"{sum(tp_a)/len(tp_a):.2f}")
            st.metric("Flow time", f"{sum(flow_a)/len(flow_a):.2f} min")
            st.metric("WIP", f"{wip_a:.2f}")
        with cb:
            st.markdown(f"**B: K1={kb1_}, K2={kb2_}**")
            st.metric("Throughput", f"{sum(tp_b)/len(tp_b):.2f}")
            st.metric("Flow time", f"{sum(flow_b)/len(flow_b):.2f} min")
            st.metric("WIP", f"{wip_b:.2f}")

        st.subheader("Evolución del WIP — primera réplica de cada config")
        st.pyplot(plot_comparison_wip(
            res_a[0].wip_trace, res_b[0].wip_trace,
            label_a=f"A (K1={ka1_}, K2={ka2_})",
            label_b=f"B (K1={kb1_}, K2={kb2_})",
        ))

@st.cache_data(show_spinner=False)
def compute_heatmap(seed: int, duration: float, n_per_cell: int) -> dict[tuple[int, int], float]:
    grid: dict[tuple[int, int], float] = {}
    for k1_ in range(1, 11):
        for k2_ in range(1, 11):
            tps = []
            for i in range(n_per_cell):
                cfg = SimulationConfig(
                    k1=k1_, k2=k2_, duration=duration, seed=seed + i * SEED_PRIME
                )
                tps.append(run_simulation(cfg).throughput)
            grid[(k1_, k2_)] = sum(tps) / n_per_cell
    return grid


with tab_heatmap:
    st.subheader("🔥 Throughput promedio por (K1, K2)")
    st.warning(
        "Esto corre 100 combinaciones × N réplicas × duración. "
        "Con N=3 y duración=960 min tarda ~2-4 min en Streamlit Cloud."
    )

    n_per_cell = st.slider("Réplicas por celda", 1, 5, 3, key="hm_n")
    hm_duration = st.number_input(
        "Duración por simulación (min)", min_value=300, max_value=2000, value=960, step=60, key="hm_dur"
    )
    hm_seed = st.number_input(
        "Semilla base", min_value=0, max_value=2**31 - 1, value=42, key="hm_seed"
    )
    run_heatmap = st.button("▶ Calcular heatmap", key="run_heatmap")

    if run_heatmap:
        with st.spinner("Calculando 100 combinaciones..."):
            grid = compute_heatmap(hm_seed, float(hm_duration), n_per_cell)
        st.session_state.heatmap_grid = grid

    if "heatmap_grid" in st.session_state:
        grid = st.session_state.heatmap_grid
        st.pyplot(plot_heatmap(grid, metric_name="Throughput (piezas/h)"))
        best = max(grid.items(), key=lambda kv: kv[1])
        (k1_best, k2_best), val_best = best
        st.success(f"🏆 Óptimo: K1={k1_best}, K2={k2_best} → {val_best:.2f} piezas/h")
