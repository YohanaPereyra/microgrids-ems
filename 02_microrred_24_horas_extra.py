# -*- coding: utf-8 -*-
"""MOC - Segundo código: EMS determinista PV + bateria + red, 24 horas.

DATOS SINTETICOS: perfiles y precios ficticios para verificar
el modelo; no representan mediciones ni tarifas reales.

El modelo decide compras, ventas, uso solar, carga y descarga. Supone un
unico nodo, parametros conocidos, sin autodescarga, degradacion ni UCF.
La potencia de bateria se mide en el bus: la carga almacena eta_carga
veces la energia recibida y la descarga extrae energia/eta_descarga.

Balance de almacenamiento de referencia:
https://docs.pypsa.org/latest/user-guide/optimization/storage/

"""

from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyomo.environ as pyo


# 1. DATOS Y PARAMETROS. Cada posicion corresponde a un intervalo:
#    indice 0 = 00:00-01:00; indice 23 = 23:00-24:00.
DATOS = pd.DataFrame({
    "hora": np.arange(24),
    "demanda_kw": [
        1.2, 1.1, 1.0, 1.0, 1.1, 1.4, 2.0, 2.5,
        2.2, 1.8, 1.7, 1.8, 2.0, 2.1, 2.0, 1.9,
        2.2, 2.8, 3.5, 4.0, 3.8, 3.0, 2.2, 1.6,
    ],
    "solar_disponible_kw": [
        0, 0, 0, 0, 0, 0, 0.3, 1.0,
        2.0, 3.2, 4.2, 4.8, 5.0, 4.8, 4.0, 3.0,
        2.0, 0.8, 0.2, 0, 0, 0, 0, 0,
    ],
    "precio_compra_um_kwh": [20.0] * 7 + [30.0] * 11 + [60.0] * 5 + [30.0],
    "precio_venta_um_kwh": [10.0] * 24,
})

PARAMETROS = {
    "paso_h": 1.0,
    "capacidad_kwh": 10.0,
    "potencia_carga_kw": 3.0,
    "potencia_descarga_kw": 3.0,
    "soc_min": 0.10,
    "soc_max": 0.90,
    "soc_inicial": 0.50,
    "soc_final": 0.50,
    "eta_carga": 0.95,
    "eta_descarga": 0.95,
    "limite_red_kw": 10.0,
}

MOSTRAR_GRAFICO = True
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados_02"


def crear_modelo(datos, p):
    """Escribir el problema matematico; todavia no resolverlo."""
    n = len(datos)
    m = pyo.ConcreteModel()
    m.horas = pyo.RangeSet(0, n - 1)
    m.instantes = pyo.RangeSet(0, n)
    e_min = p["soc_min"] * p["capacidad_kwh"]
    e_max = p["soc_max"] * p["capacidad_kwh"]

    # 2. VARIABLES DE DECISION.
    m.compra = pyo.Var(m.horas, bounds=(0, p["limite_red_kw"]))
    m.venta = pyo.Var(m.horas, bounds=(0, p["limite_red_kw"]))
    m.pv = pyo.Var(
        m.horas,
        bounds=lambda m, t: (0, float(datos.loc[t, "solar_disponible_kw"])),
    )
    m.carga = pyo.Var(m.horas, bounds=(0, p["potencia_carga_kw"]))
    m.descarga = pyo.Var(m.horas, bounds=(0, p["potencia_descarga_kw"]))
    m.energia = pyo.Var(m.instantes, bounds=(e_min, e_max))
    m.modo_carga = pyo.Var(m.horas, domain=pyo.Binary)
    m.modo_compra = pyo.Var(m.horas, domain=pyo.Binary)

    # 3. RESTRICCIONES. Todos los terminos del balance son potencias del bus.
    m.balance = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.compra[t] + m.pv[t] + m.descarga[t]
        == float(datos.loc[t, "demanda_kw"]) + m.carga[t] + m.venta[t],
    )
    m.evolucion = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.energia[t + 1] == m.energia[t] + p["paso_h"] * (
            p["eta_carga"] * m.carga[t] - m.descarga[t] / p["eta_descarga"]
        ),
    )
    # La variable binaria habilita un solo sentido en cada intervalo.
    m.limite_carga = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.carga[t] <= p["potencia_carga_kw"] * m.modo_carga[t],
    )
    m.limite_descarga = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.descarga[t]
        <= p["potencia_descarga_kw"] * (1 - m.modo_carga[t]),
    )
    m.limite_compra = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.compra[t] <= p["limite_red_kw"] * m.modo_compra[t],
    )
    m.limite_venta = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.venta[t] <= p["limite_red_kw"] * (1 - m.modo_compra[t]),
    )
    # El SOC final igual al inicial evita consumir gratis la reserva inicial.
    m.energia[0].fix(p["soc_inicial"] * p["capacidad_kwh"])
    m.energia[n].fix(p["soc_final"] * p["capacidad_kwh"])

    # 4. OBJETIVO: compras menos ingresos por ventas, en unidades monetarias.
    m.costo = pyo.Objective(
        expr=sum(p["paso_h"] * (
            float(datos.loc[t, "precio_compra_um_kwh"]) * m.compra[t]
            - float(datos.loc[t, "precio_venta_um_kwh"]) * m.venta[t]
        ) for t in m.horas),
        sense=pyo.minimize,
    )
    return m


def resolver_modelo(datos, p):
    """5. Resolver el MILP y extraer los valores numericos."""
    m = crear_modelo(datos, p)
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["mip_rel_gap"] = 1e-8
    resultado = solver.solve(m)
    if not pyo.check_optimal_termination(resultado):
        raise RuntimeError("El solver no encontro una solucion optima.")

    tabla = datos.copy()
    for nombre, variable in (
        ("compra_kw", m.compra), ("venta_kw", m.venta),
        ("solar_usada_kw", m.pv), ("carga_kw", m.carga),
        ("descarga_kw", m.descarga),
    ):
        tabla[nombre] = [pyo.value(variable[t]) for t in m.horas]
    tabla["energia_inicial_kwh"] = [pyo.value(m.energia[t]) for t in m.horas]
    tabla["energia_final_kwh"] = [pyo.value(m.energia[t + 1]) for t in m.horas]
    tabla["soc_final_pct"] = 100 * tabla["energia_final_kwh"] / p["capacidad_kwh"]
    tabla["solar_no_usada_kw"] = tabla["solar_disponible_kw"] - tabla["solar_usada_kw"]
    tabla["costo_intervalo_um"] = p["paso_h"] * (
        tabla["precio_compra_um_kwh"] * tabla["compra_kw"]
        - tabla["precio_venta_um_kwh"] * tabla["venta_kw"]
    )
    return m, tabla


def validar_resultados(tabla, p):
    """Comprobar el balance y la operacion fisica con una tolerancia numerica."""
    tol = 1e-6
    balance = (tabla["compra_kw"] + tabla["solar_usada_kw"] + tabla["descarga_kw"]
               - tabla["demanda_kw"] - tabla["carga_kw"] - tabla["venta_kw"])
    evolucion = (tabla["energia_final_kwh"] - tabla["energia_inicial_kwh"]
                 - p["paso_h"] * (p["eta_carga"] * tabla["carga_kw"]
                                    - tabla["descarga_kw"] / p["eta_descarga"]))
    errores = {"balance_kw": float(balance.abs().max()),
               "evolucion_kwh": float(evolucion.abs().max())}
    energias = np.r_[tabla["energia_inicial_kwh"].iloc[0], tabla["energia_final_kwh"]]
    correcto = (
        max(errores.values()) <= tol
        and energias.min() >= p["soc_min"] * p["capacidad_kwh"] - tol
        and energias.max() <= p["soc_max"] * p["capacidad_kwh"] + tol
        and abs(energias[0] - p["soc_inicial"] * p["capacidad_kwh"]) <= tol
        and abs(energias[-1] - p["soc_final"] * p["capacidad_kwh"]) <= tol
        and np.allclose(tabla["energia_inicial_kwh"].iloc[1:],
                        tabla["energia_final_kwh"].iloc[:-1], atol=tol, rtol=0)
    )
    for columna, limite in (
        ("compra_kw", p["limite_red_kw"]), ("venta_kw", p["limite_red_kw"]),
        ("carga_kw", p["potencia_carga_kw"]),
        ("descarga_kw", p["potencia_descarga_kw"]),
    ):
        correcto = correcto and tabla[columna].min() >= -tol and tabla[columna].max() <= limite + tol
    correcto = correcto and (
        tabla["solar_usada_kw"].min() >= -tol
        and tabla["solar_no_usada_kw"].min() >= -tol
        and not ((tabla["carga_kw"] > tol) & (tabla["descarga_kw"] > tol)).any()
        and not ((tabla["compra_kw"] > tol) & (tabla["venta_kw"] > tol)).any()
    )
    if not correcto:
        raise RuntimeError("La verificacion fisica del despacho fallo.")
    return errores


def referencia_sin_bateria(datos, p):
    """Misma demanda, PV y precios, sin almacenamiento; excedentes vendidos."""
    demanda_neta = datos["demanda_kw"] - datos["solar_disponible_kw"]
    compras = demanda_neta.clip(lower=0)
    ventas = (-demanda_neta).clip(lower=0)
    if max(compras.max(), ventas.max()) > p["limite_red_kw"] + 1e-6:
        raise ValueError("El caso sin bateria supera el limite de la red.")
    return float(p["paso_h"] * (
        datos["precio_compra_um_kwh"] * compras
        - datos["precio_venta_um_kwh"] * ventas
    ).sum())


def agregar_abastecimiento_demanda(tabla):
    """Reparto VISUAL: solar a demanda primero, luego bateria y red.

    El MILP balancea un unico nodo y no distingue recorridos fisicos de la
    energia. Esta convencion solo representa los aportes inmediatos al
    consumo: excluye la potencia destinada a cargar la bateria y las ventas.
    No modifica el despacho ni el costo de la optimizacion.
    """
    tabla = tabla.copy()
    demanda = tabla["demanda_kw"].to_numpy()
    solar = np.minimum(np.maximum(tabla["solar_usada_kw"].to_numpy(), 0), demanda)
    bateria = np.minimum(
        np.maximum(tabla["descarga_kw"].to_numpy(), 0),
        np.maximum(demanda - solar, 0),
    )
    red = np.maximum(demanda - solar - bateria, 0)
    if (not np.allclose(solar + bateria + red, demanda, atol=1e-6, rtol=0)
            or np.any(red > tabla["compra_kw"].to_numpy() + 1e-6)):
        raise RuntimeError("El reparto visual no coincide con el balance de demanda.")
    tabla["solar_a_demanda_kw"] = solar
    tabla["bateria_a_demanda_kw"] = bateria
    tabla["red_a_demanda_kw"] = red
    etiquetas = []
    for aportes in zip(solar, bateria, red):
        activas = [nombre for nombre, aporte in zip(("Solar", "Bateria", "Red"), aportes)
                   if aporte > 1e-6]
        etiquetas.append(activas[0] if len(activas) == 1 else
                         "Mixto" if activas else "Sin demanda")
    tabla["fuente_demanda"] = etiquetas
    return tabla


def graficar_abastecimiento(tabla, p, carpeta):
    """Demanda coloreada por fuente y aportes simultaneos apilados."""
    bordes = np.arange(len(tabla) + 1) * p["paso_h"]
    demanda = tabla["demanda_kw"].to_numpy()
    colores = {"Solar": "#dc8b15", "Bateria": "#18856d", "Red": "#7854a5",
               "Mixto": "#247ac0", "Sin demanda": "#888888"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Abastecimiento de la demanda")
    fig.suptitle(
        "Quien abastece la demanda - dia simulado de 24 horas\n"
        "Asignacion visual: solar primero; luego bateria y red",
        fontsize=14,
    )

    # Misma altura que la demanda; el color identifica cada intervalo.
    segmentos = [[(bordes[t], demanda[t]), (bordes[t + 1], demanda[t])]
                 for t in range(len(tabla))]
    axes[0].add_collection(LineCollection(
        segmentos, colors=[colores[f] for f in tabla["fuente_demanda"]],
        linewidths=7, zorder=3,
    ))
    axes[0].stairs(demanda, bordes, color="black", linestyle="--",
                   linewidth=1.4, zorder=4)
    leyenda = [Line2D([0], [0], color="black", linestyle="--", label="Demanda")]
    for fuente, nombre in (("Solar", "Solar"), ("Bateria", "Bateria"),
                           ("Red", "Red"), ("Mixto", "Varias fuentes a la vez")):
        leyenda.append(Line2D([0], [0], color=colores[fuente], linewidth=5, label=nombre))
    axes[0].legend(handles=leyenda, loc="upper left", ncol=3, fontsize=9)
    axes[0].set_title("La curva cambia de color segun quien abastece la demanda", loc="left", fontsize=11)

    # Las alturas de los tres colores suman exactamente la demanda.
    solar = tabla["solar_a_demanda_kw"].to_numpy()
    bateria = tabla["bateria_a_demanda_kw"].to_numpy()
    red = tabla["red_a_demanda_kw"].to_numpy()
    axes[1].stackplot(
        bordes, np.r_[solar, solar[-1]], np.r_[bateria, bateria[-1]],
        np.r_[red, red[-1]], labels=["Solar a demanda", "Bateria a demanda", "Red a demanda"],
        colors=[colores["Solar"], colores["Bateria"], colores["Red"]],
        alpha=0.8, step="post",
    )
    axes[1].stairs(demanda, bordes, color="black", linestyle="--",
                   linewidth=1.6, label="Demanda total", zorder=4)
    axes[1].legend(loc="upper left", ncol=2, fontsize=9)
    axes[1].set_title("Aporte de cada fuente: los colores apilados suman la demanda", loc="left", fontsize=11)
    axes[1].set_xlabel("Hora del dia (cada intervalo dura 1 hora)")
    axes[1].set_xticks(np.arange(0, len(tabla) + 1, 2) * p["paso_h"])
    for ax in axes:
        ax.set_ylabel("Potencia para demanda (kW)")
        ax.set_ylim(0, max(float(demanda.max()) * 1.35, 1))
        ax.set_xlim(bordes[0], bordes[-1])
        ax.grid(alpha=0.22)
    fig.savefig(carpeta / "abastecimiento_demanda_24h.png", dpi=160)
    return fig


def graficar(tabla, p, carpeta, costo, costo_referencia):
    """6. Mostrar perfiles, despacho, SOC y precios del mismo dia."""
    bordes = np.arange(len(tabla) + 1) * p["paso_h"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True, layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Despacho, SOC y precios")
    fig.suptitle(
        "EMS determinista de 24 horas - datos sinteticos\n"
        f"Costo con bateria: {costo:.2f} u.m. | Sin bateria: {costo_referencia:.2f} u.m.",
        fontsize=14,
    )
    axes[0].stairs(tabla["demanda_kw"], bordes, label="Demanda", color="#2455a4", linewidth=1.8)
    axes[0].stairs(tabla["solar_disponible_kw"], bordes, label="Solar disponible", color="#dc8b15", linewidth=1.8)
    axes[0].set_ylabel("Potencia (kW)")
    axes[0].set_title("Perfiles conocidos al programar el dia", loc="left", fontsize=11)

    axes[1].stairs(tabla["compra_kw"] - tabla["venta_kw"], bordes,
                   label="Red: + compra / - venta", color="#7854a5", linewidth=1.8)
    axes[1].stairs(tabla["descarga_kw"] - tabla["carga_kw"], bordes,
                   label="Bateria: + descarga / - carga", color="#18856d", linewidth=1.8)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Potencia (kW)")
    axes[1].set_title("Decisiones calculadas por el MILP", loc="left", fontsize=11)

    soc = 100 * np.r_[tabla["energia_inicial_kwh"].iloc[0],
                      tabla["energia_final_kwh"]] / p["capacidad_kwh"]
    axes[2].plot(bordes, soc, marker="o", markersize=3, color="#18856d", label="SOC")
    axes[2].axhline(100 * p["soc_min"], linestyle="--", color="gray", label="Limites de SOC")
    axes[2].axhline(100 * p["soc_max"], linestyle="--", color="gray")
    axes[2].set_ylabel("SOC (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Estado de carga al inicio y al final de cada intervalo", loc="left", fontsize=11)

    axes[3].stairs(tabla["precio_compra_um_kwh"], bordes, label="Precio de compra", color="#b04c44", linewidth=1.8)
    axes[3].stairs(tabla["precio_venta_um_kwh"], bordes, label="Precio de venta", color="#51758b", linewidth=1.8)
    axes[3].set_ylabel("Precio (u.m./kWh)")
    axes[3].set_xlabel("Hora del dia (cada intervalo dura 1 hora)")
    axes[3].set_xticks(np.arange(0, 25, 3))
    axes[3].set_xlim(0, 24)
    for ax in axes:
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.savefig(carpeta / "microrred_24h.png", dpi=160)
    return fig


def main():
    m, tabla = resolver_modelo(DATOS, PARAMETROS)
    errores = validar_resultados(tabla, PARAMETROS)
    tabla = agregar_abastecimiento_demanda(tabla)
    costo = pyo.value(m.costo)
    costo_referencia = referencia_sin_bateria(DATOS, PARAMETROS)
    diferencia = costo_referencia - costo
    resumen = (
        "Estado: solucion optima\n"
        "EMS determinista PV + bateria + red. Datos y precios sinteticos.\n"
        f"Costo con bateria: {costo:.2f} unidades monetarias\n"
        f"Costo sin bateria: {costo_referencia:.2f} unidades monetarias\n"
        f"Diferencia de costo: {diferencia:.2f} unidades monetarias\n"
        f"SOC inicial: {100 * PARAMETROS['soc_inicial']:.1f} %\n"
        f"SOC final: {tabla['soc_final_pct'].iloc[-1]:.1f} %\n"
        f"Maximo error de balance: {errores['balance_kw']:.2e} kW\n"
        f"Maximo error de evolucion de bateria: {errores['evolucion_kwh']:.2e} kWh\n"
    )
    print(resumen)
    columnas = ["hora", "demanda_kw", "compra_kw", "venta_kw",
                "carga_kw", "descarga_kw", "soc_final_pct"]
    print(tabla[columnas].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    DATOS.to_csv(CARPETA_RESULTADOS / "datos_24h.csv", index=False, encoding="utf-8-sig")
    tabla.to_csv(CARPETA_RESULTADOS / "despacho_24h.csv", index=False,
                 float_format="%.9f", encoding="utf-8-sig")
    parametros_texto = "\n".join(f"{k}: {v}" for k, v in PARAMETROS.items())
    versiones = "\n".join(f"{nombre}: {version(nombre)}"
                          for nombre in ("pyomo", "highspy", "numpy", "pandas", "matplotlib"))
    (CARPETA_RESULTADOS / "resumen_24h.txt").write_text(
        resumen + "\nParametros\n" + parametros_texto + "\n\nVersiones\n" + versiones + "\n",
        encoding="utf-8",
    )
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")
    figura_general = graficar(tabla, PARAMETROS, CARPETA_RESULTADOS, costo, costo_referencia)
    figura_demanda = graficar_abastecimiento(tabla, PARAMETROS, CARPETA_RESULTADOS)
    if MOSTRAR_GRAFICO:
        print("Se abriran dos figuras. Cerra ambas ventanas para finalizar la ejecucion.")
        plt.show()
    else:
        plt.close(figura_general)
        plt.close(figura_demanda)


if __name__ == "__main__":
    main()
