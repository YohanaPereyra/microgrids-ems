# -*- coding: utf-8 -*-
"""MOC - Primer código de error de pronostico solar.

Guardar junto a 02_microrred_24_horas.py y ejecutar este archivo.

1. Optimizar usando el pronostico del caso original.
2. Mantener FIJAS la carga y descarga horarias de bateria.
3. Simular una generacion real menor entre las 9 y las 16 horas.
4. Ajustar compras y ventas de red para mantener el balance.
5. Comparar con una referencia ideal de informacion perfecta.

La generacion real es una realizacion sintetica elegida para el ejercicio,
no una medicion. Demanda, precios y parametros de bateria no cambian.
El ajuste de red usa los precios de compra/venta originales, sin tarifas
separadas de desbalance. La bateria no se reoptimiza durante la ejecucion.

La referencia ideal resuelve con toda la generacion real conocida antes
de decidir: es una cota de comparacion, no una estrategia implementable
cuando ese futuro no se conoce. Este archivo todavia no incorpora UCF.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo


FACTOR_SOLAR_REAL = 0.70
HORA_INICIO_ERROR = 9
HORA_FIN_ERROR = 16  # Intervalo [9, 16): incluye la fila 15.
MOSTRAR_GRAFICO = True
ARCHIVO_BASE = Path(__file__).resolve().with_name("02_microrred_24_horas.py")
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados_04"


def cargar_modelo_base():
    if not ARCHIVO_BASE.exists():
        raise FileNotFoundError("Guarda este archivo junto a 02_microrred_24_horas.py.")
    spec = importlib.util.spec_from_file_location("modelo_base_moc", ARCHIVO_BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def crear_realizacion(datos):
    """Generacion real del ejemplo; el pronostico original queda intacto."""
    if not 0 <= FACTOR_SOLAR_REAL <= 1:
        raise ValueError("En este ejercicio el factor solar debe estar entre 0 y 1.")
    if not 0 <= HORA_INICIO_ERROR < HORA_FIN_ERROR <= len(datos):
        raise ValueError("Revisa el inicio y fin del intervalo con error.")
    real = datos.copy(deep=True)
    afectadas = real["hora"].between(HORA_INICIO_ERROR, HORA_FIN_ERROR - 1)
    real.loc[afectadas, "solar_disponible_kw"] *= FACTOR_SOLAR_REAL
    return real


def simular_ejecucion(plan, datos_reales, p):
    """Ejecutar la misma bateria y usar la red para compensar la diferencia.

    Si sobra energia por encima del limite de exportacion, se recorta PV.
    Si la red no puede cubrir el deficit, se informa que este plan fijo
    no es factible. No se permite superar el limite de red silenciosamente.
    """
    real = datos_reales.copy(deep=True)
    for columna in ("carga_kw", "descarga_kw", "energia_inicial_kwh",
                    "energia_final_kwh", "soc_final_pct"):
        real[columna] = plan[columna].to_numpy(copy=True)
    neta = (real["demanda_kw"] + real["carga_kw"] - real["descarga_kw"]
            - real["solar_disponible_kw"])
    compra = neta.clip(lower=0)
    venta_requerida = (-neta).clip(lower=0)
    if (compra > p["limite_red_kw"] + 1e-6).any():
        raise RuntimeError("El plan fijo requiere compras mayores al limite de la red.")
    recorte = (venta_requerida - p["limite_red_kw"]).clip(lower=0)
    if (recorte > real["solar_disponible_kw"] + 1e-6).any():
        raise RuntimeError("El plan fijo genera un excedente que no puede compensarse.")
    real["compra_kw"] = compra
    real["venta_kw"] = venta_requerida - recorte
    real["solar_usada_kw"] = real["solar_disponible_kw"] - recorte
    real["solar_no_usada_kw"] = recorte
    real["costo_intervalo_um"] = p["paso_h"] * (
        real["precio_compra_um_kwh"] * real["compra_kw"]
        - real["precio_venta_um_kwh"] * real["venta_kw"]
    )
    return real


def construir_comparacion(plan, real):
    return pd.DataFrame({
        "hora": plan["hora"],
        "pv_prev_kw": plan["solar_disponible_kw"],
        "pv_real_kw": real["solar_disponible_kw"],
        "red_prev_kw": plan["compra_kw"] - plan["venta_kw"],
        "red_real_kw": real["compra_kw"] - real["venta_kw"],
        "carga_fija_kw": plan["carga_kw"],
        "descarga_fija_kw": plan["descarga_kw"],
        "costo_previsto_um": plan["costo_intervalo_um"],
        "costo_real_um": real["costo_intervalo_um"],
        "variacion_costo_um": real["costo_intervalo_um"] - plan["costo_intervalo_um"],
    })


def graficar(comparacion, p, costos, carpeta):
    bordes = np.arange(len(comparacion) + 1) * p["paso_h"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Pronostico y ejecucion real")
    fig.suptitle("Error de pronostico solar - realizacion sintetica de 24 horas", fontsize=14)

    axes[0].stairs(comparacion["pv_prev_kw"], bordes, label="Solar pronosticada",
                   color="#2455a4", linewidth=2)
    axes[0].stairs(comparacion["pv_real_kw"], bordes, label="Solar real simulada",
                   color="#dc8b15", linewidth=2)
    prevista = comparacion["pv_prev_kw"].to_numpy()
    real = comparacion["pv_real_kw"].to_numpy()
    axes[0].fill_between(bordes, np.r_[prevista, prevista[-1]], np.r_[real, real[-1]],
                         step="post", color="#dc8b15", alpha=0.18)
    axes[0].set_title("1. Con que generacion se programo y que generacion hubo", loc="left", fontsize=11)
    axes[0].set_ylabel("Potencia solar (kW)")

    axes[1].stairs(comparacion["red_prev_kw"], bordes, label="Red programada",
                   color="#7854a5", linewidth=2)
    axes[1].stairs(comparacion["red_real_kw"], bordes, label="Red en ejecucion real",
                   color="#b04c44", linestyle="--", linewidth=2)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_title("2. Misma bateria programada; la red compensa el desbalance", loc="left", fontsize=11)
    axes[1].set_ylabel("Red (kW)\n+ compra / - venta")
    axes[1].set_xlabel("Hora del dia (cada intervalo dura 1 hora)")
    for ax in axes[:2]:
        ax.set_xlim(0, bordes[-1])
        ax.set_xticks(np.arange(0, 25, 2))
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", ncol=2, fontsize=9)

    etiquetas = ["Previsto\ncon pronostico", "Real\ncon el mismo plan", "Referencia ideal\ncon informacion perfecta"]
    barras = axes[2].bar(etiquetas, costos, color=["#2455a4", "#dc8b15", "#18856d"], width=0.55)
    axes[2].bar_label(barras, labels=[f"{v:.2f}" for v in costos], padding=5)
    axes[2].set_ylim(0, max(costos) * 1.25)
    axes[2].set_ylabel("Costo operativo del dia (u.m.)")
    axes[2].set_title("3. Las columnas 2 y 3 usan la misma generacion real", loc="left", fontsize=11)
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].set_axisbelow(True)
    fig.savefig(carpeta / "error_pronostico_24h.png", dpi=160)
    return fig


def main():
    base = cargar_modelo_base()
    p = base.PARAMETROS.copy()
    pronostico = base.DATOS.copy(deep=True)

    # A. Decisiones tomadas antes de conocer el futuro.
    modelo, plan = base.resolver_modelo(pronostico, p)
    base.validar_resultados(plan, p)
    costo_previsto = float(pyo.value(modelo.costo))

    # B. Ejecucion: la bateria sigue ese plan; cambian las compras y ventas.
    datos_reales = crear_realizacion(pronostico)
    ejecucion = simular_ejecucion(plan, datos_reales, p)
    errores = base.validar_resultados(ejecucion, p)
    costo_real = float(ejecucion["costo_intervalo_um"].sum())

    # C. Referencia retrospectiva: conocer toda la generacion real al decidir.
    modelo_ideal, despacho_ideal = base.resolver_modelo(datos_reales, p)
    base.validar_resultados(despacho_ideal, p)
    costo_ideal = float(pyo.value(modelo_ideal.costo))
    if costo_real < costo_ideal - 1e-6:
        raise RuntimeError("La ejecucion factible no puede mejorar la referencia optima ideal.")

    comparacion = construir_comparacion(plan, ejecucion)
    energia_prevista = float(pronostico["solar_disponible_kw"].sum() * p["paso_h"])
    energia_real = float(datos_reales["solar_disponible_kw"].sum() * p["paso_h"])
    resumen = (
        "MOC - Error de pronostico solar. Caso sintetico.\n"
        f"Entre {HORA_INICIO_ERROR} y {HORA_FIN_ERROR} h, generacion real al {100 * FACTOR_SOLAR_REAL:.0f} % de la prevista.\n"
        f"Energia solar prevista: {energia_prevista:.2f} kWh\n"
        f"Energia solar real: {energia_real:.2f} kWh\n"
        f"Costo previsto: {costo_previsto:.2f} unidades monetarias\n"
        f"Costo real con el mismo plan de bateria: {costo_real:.2f} unidades monetarias\n"
        f"Variacion frente al costo previsto: {costo_real - costo_previsto:.2f} unidades monetarias\n"
        f"Referencia ideal con informacion perfecta: {costo_ideal:.2f} unidades monetarias\n"
        f"Brecha respecto de la referencia ideal: {costo_real - costo_ideal:.2f} unidades monetarias\n"
        f"Maximo error de balance en ejecucion real: {errores['balance_kw']:.2e} kW\n"
        "Carga, descarga y SOC de bateria se mantienen como fueron programados.\n"
        "La red compensa a los precios originales; no se incluyen tarifas de desbalance.\n"
        "Referencia ideal: el futuro se conoce antes de decidir; no es un controlador real.\n"
        "La variacion de costo de este ejemplo no es una UCF.\n"
    )
    print(resumen)
    print("Detalle del intervalo con error. Red positiva = compra; negativa = venta.")
    afectadas = comparacion["hora"].between(HORA_INICIO_ERROR, HORA_FIN_ERROR - 1)
    columnas = ["hora", "pv_prev_kw", "pv_real_kw", "red_prev_kw", "red_real_kw", "carga_fija_kw"]
    print(comparacion.loc[afectadas, columnas].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in (("plan_pronostico", plan), ("ejecucion_real", ejecucion),
                           ("referencia_info_perfecta", despacho_ideal),
                           ("comparacion_pronostico_real", comparacion)):
        tabla.to_csv(CARPETA_RESULTADOS / f"{nombre}.csv", index=False,
                     float_format="%.9f", encoding="utf-8-sig")
    (CARPETA_RESULTADOS / "resumen_error_pronostico.txt").write_text(resumen, encoding="utf-8")
    fig = graficar(comparacion, p, [costo_previsto, costo_real, costo_ideal], CARPETA_RESULTADOS)
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICO:
        print("Cerra el grafico para finalizar la ejecucion.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
