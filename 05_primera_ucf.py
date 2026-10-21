# -*- coding: utf-8 -*-
"""MOC - Primera UCF: programar potencia solar para un intervalo de 1 hora.

Ejemplo independiente y sintetico. Ejecutar con el mismo entorno .venv.
No modifica ni importa los archivos anteriores.

W es la potencia disponible, incierta; s es la potencia programada, elegida
ANTES de conocer W. Se minimiza la penalizacion esperada:

    UCF(s) = sum(prob_i * [c_falta * max(s-W_i, 0)
                          + c_exceso * max(W_i-s, 0)]) * duracion_h

Las probabilidades y penalizaciones son supuestos didacticos. El faltante
es respecto de la programacion, no demanda electrica sin abastecer. Los
coeficientes no son las tarifas de compra/venta del ejemplo de 24 horas.

Se usa la forma de penalizacion por desvio de la definicion 1 de:
Sanchez Reinoso y Rivera (2025), DOI: 10.37394/23206.2025.24.79.
https://www.wseas.com/journals/articles.php?id=11638
Aqui c_falta corresponde a sobreestimar la generacion; c_exceso, a
subestimarla. La distribucion se representa con tres escenarios discretos.

La formulacion con variables auxiliares es un LP exacto para estos
escenarios: no requiere variables enteras. Todavia no integra bateria,
demanda, red ni el MILP de 24 horas. Los costos son de un intervalo,
no una factura diaria ni el costo operativo total de una microrred.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo


# 1. DATOS CONOCIDOS ANTES DE DECIDIR. Probabilidades suman 1.
ESCENARIOS = pd.DataFrame({
    "escenario": ["Menor generacion", "Intermedio", "Mayor generacion"],
    "solar_kw": [3.5, 5.0, 6.5],
    "probabilidad": [0.25, 0.50, 0.25],
})

PARAMETROS = {
    "duracion_h": 1.0,
    "potencia_max_kw": 7.0,
    "costo_faltante_um_kwh": 50.0,
    "costo_excedente_um_kwh": 10.0,
    "programacion_referencia_kw": 5.0,
}

MOSTRAR_GRAFICO = True
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados_05"


def validar_datos(escenarios, p):
    valores = escenarios[["solar_kw", "probabilidad"]].to_numpy(dtype=float)
    if len(escenarios) == 0 or not np.isfinite(valores).all():
        raise ValueError("Revisa las potencias y probabilidades de los escenarios.")
    if not all(np.isfinite(float(v)) for v in p.values()):
        raise ValueError("Los parametros deben ser numeros finitos.")
    if p["duracion_h"] <= 0 or p["potencia_max_kw"] <= 0:
        raise ValueError("La duracion y potencia maxima deben ser positivas.")
    if min(p["costo_faltante_um_kwh"], p["costo_excedente_um_kwh"]) <= 0:
        raise ValueError("Este ejemplo requiere ambas penalizaciones positivas.")
    if (escenarios["probabilidad"] < 0).any() or not np.isclose(
        escenarios["probabilidad"].sum(), 1.0, atol=1e-10, rtol=0
    ):
        raise ValueError("Las probabilidades deben ser no negativas y sumar 1.")
    if (escenarios["solar_kw"] < 0).any() or (
        escenarios["solar_kw"] > p["potencia_max_kw"]
    ).any():
        raise ValueError("La generacion de cada escenario debe respetar la potencia maxima.")
    if not 0 <= p["programacion_referencia_kw"] <= p["potencia_max_kw"]:
        raise ValueError("La programacion de referencia debe respetar la potencia maxima.")


def evaluar_ucf(programacion_kw, escenarios, p):
    """Evaluacion directa de la formula matematica para una decision dada."""
    disponible = escenarios["solar_kw"].to_numpy(dtype=float)
    probabilidades = escenarios["probabilidad"].to_numpy(dtype=float)
    faltante = np.maximum(programacion_kw - disponible, 0.0)
    excedente = np.maximum(disponible - programacion_kw, 0.0)
    penalizaciones = p["duracion_h"] * (
        p["costo_faltante_um_kwh"] * faltante
        + p["costo_excedente_um_kwh"] * excedente
    )
    return float(probabilidades @ penalizaciones)


def resolver_ucf(escenarios, p):
    """2. LINEALIZAR los maximos y resolver con Pyomo + HiGHS.

    Hay UNA programacion comun a todos los escenarios: se decide antes de
    saber cual ocurre. Las variables de desvio cambian por escenario.
    Al minimizar con coeficientes positivos, las desigualdades hacen que
    cada auxiliar sea exactamente el maximo correspondiente.
    """
    validar_datos(escenarios, p)
    datos = escenarios.reset_index(drop=True)
    m = pyo.ConcreteModel()
    m.escenarios = pyo.RangeSet(0, len(datos) - 1)

    # Decision previa a conocer la generacion real; no lleva indice i.
    m.programacion = pyo.Var(bounds=(0, p["potencia_max_kw"]))
    m.faltante = pyo.Var(m.escenarios, domain=pyo.NonNegativeReals)
    m.excedente = pyo.Var(m.escenarios, domain=pyo.NonNegativeReals)

    m.desvio_faltante = pyo.Constraint(
        m.escenarios,
        rule=lambda m, i: m.faltante[i] >= m.programacion - float(datos.loc[i, "solar_kw"]),
    )
    m.desvio_excedente = pyo.Constraint(
        m.escenarios,
        rule=lambda m, i: m.excedente[i] >= float(datos.loc[i, "solar_kw"]) - m.programacion,
    )
    m.costo_esperado = pyo.Objective(
        expr=sum(float(datos.loc[i, "probabilidad"]) * p["duracion_h"] * (
            p["costo_faltante_um_kwh"] * m.faltante[i]
            + p["costo_excedente_um_kwh"] * m.excedente[i]
        ) for i in m.escenarios),
        sense=pyo.minimize,
    )
    solver = pyo.SolverFactory("appsi_highs")
    resultado = solver.solve(m)
    if not pyo.check_optimal_termination(resultado):
        raise RuntimeError("El solver no encontro una solucion optima de la UCF.")

    programacion = float(pyo.value(m.programacion))
    costo = float(pyo.value(m.costo_esperado))
    directo = evaluar_ucf(programacion, datos, p)
    if abs(costo - directo) > 1e-6:
        raise RuntimeError("La formulacion lineal no coincide con la formula de la UCF.")
    return m, programacion, costo


def detallar_decision(programacion_kw, escenarios, p):
    tabla = escenarios.copy(deep=True)
    tabla["programacion_kw"] = programacion_kw
    tabla["faltante_kw"] = np.maximum(programacion_kw - tabla["solar_kw"], 0.0)
    tabla["excedente_kw"] = np.maximum(tabla["solar_kw"] - programacion_kw, 0.0)
    tabla["penalizacion_si_ocurre_um"] = p["duracion_h"] * (
        p["costo_faltante_um_kwh"] * tabla["faltante_kw"]
        + p["costo_excedente_um_kwh"] * tabla["excedente_kw"]
    )
    tabla["aporte_costo_esperado_um"] = (
        tabla["probabilidad"] * tabla["penalizacion_si_ocurre_um"]
    )
    return tabla


def graficar(escenarios, p, curva, optimo_kw, costo_optimo, costo_ref, carpeta):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Primera UCF")
    fig.suptitle("Primera UCF - un intervalo de 1 hora, datos sinteticos", fontsize=14)

    etiquetas = [f"{fila.escenario}\n{fila.solar_kw:g} kW"
                 for fila in escenarios.itertuples()]
    barras = axes[0].bar(etiquetas, 100 * escenarios["probabilidad"],
                         color=["#b04c44", "#2455a4", "#dc8b15"], width=0.55)
    axes[0].bar_label(barras, labels=[f"{100 * v:g} %" for v in escenarios["probabilidad"]],
                      padding=5)
    axes[0].set_ylim(0, 65)
    axes[0].set_ylabel("Probabilidad (%)")
    axes[0].set_title("1. Tres generaciones posibles antes de decidir; ocurre una sola", loc="left")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].set_axisbelow(True)

    axes[1].plot(curva["programacion_kw"], curva["ucf_um"], color="#2455a4", linewidth=2,
                 label="UCF: penalizacion esperada")
    ref_kw = p["programacion_referencia_kw"]
    axes[1].scatter(ref_kw, costo_ref, s=85, color="#dc8b15", zorder=4,
                    label=f"Referencia: {ref_kw:g} kW; UCF {costo_ref:.2f} u.m.")
    axes[1].scatter(optimo_kw, costo_optimo, s=90, color="#18856d", zorder=5,
                    label=f"Optimo: {optimo_kw:g} kW; UCF {costo_optimo:.2f} u.m.")
    axes[1].vlines([optimo_kw, ref_kw], 0, [costo_optimo, costo_ref],
                   colors=["#18856d", "#dc8b15"], linestyles="--", alpha=0.7)
    axes[1].set_xlim(0, p["potencia_max_kw"])
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Potencia solar que decidimos programar (kW)")
    axes[1].set_ylabel("Penalizacion esperada del intervalo (u.m.)")
    axes[1].set_title("2. El solver elige la programacion con menor UCF; el eje horizontal no es tiempo",
                      loc="left", fontsize=10.5)
    axes[1].legend(loc="upper center", fontsize=9)
    axes[1].grid(alpha=0.22)
    fig.savefig(carpeta / "primera_ucf.png", dpi=160)
    return fig


def main():
    p = PARAMETROS.copy()
    escenarios = ESCENARIOS.copy(deep=True)
    _, optimo_kw, costo_optimo = resolver_ucf(escenarios, p)
    ref_kw = p["programacion_referencia_kw"]
    costo_ref = evaluar_ucf(ref_kw, escenarios, p)
    media_kw = float((escenarios["probabilidad"] * escenarios["solar_kw"]).sum())
    referencia = detallar_decision(ref_kw, escenarios, p)
    optimo = detallar_decision(optimo_kw, escenarios, p)

    # Incluir exactamente los quiebres y decisiones al dibujar la funcion.
    potencias = np.unique(np.r_[np.linspace(0, p["potencia_max_kw"], 351),
                                escenarios["solar_kw"], ref_kw, optimo_kw])
    curva = pd.DataFrame({"programacion_kw": potencias,
                          "ucf_um": [evaluar_ucf(v, escenarios, p) for v in potencias]})
    resumen = (
        "MOC - Primera UCF con tres escenarios sinteticos.\n"
        "Modelo: programacion lineal (LP) con auxiliares para los desvios.\n"
        f"Duracion del intervalo: {p['duracion_h']:.2f} h\n"
        f"Penalizacion por faltante: {p['costo_faltante_um_kwh']:.2f} u.m./kWh\n"
        f"Penalizacion por excedente: {p['costo_excedente_um_kwh']:.2f} u.m./kWh\n"
        f"Potencia solar media de los escenarios: {media_kw:.2f} kW\n"
        f"Programacion de referencia: {ref_kw:.2f} kW\n"
        f"UCF con esa referencia: {costo_ref:.2f} u.m.\n"
        f"Programacion elegida al minimizar UCF: {optimo_kw:.2f} kW\n"
        f"UCF minima: {costo_optimo:.2f} u.m.\n"
        f"Reduccion de penalizacion esperada: {costo_ref - costo_optimo:.2f} u.m.\n"
        "La generacion disponible no cambia al cambiar la programacion.\n"
        "La UCF es un promedio probabilistico: no es el costo observado de un escenario.\n"
        "Este archivo todavia no integra la UCF al MILP de la microrred.\n"
        "Las penalizaciones de este intervalo no se comparan con la factura de 24 horas.\n"
    )
    print(resumen)
    for titulo, tabla in (("DETALLE CON LA PROGRAMACION DE REFERENCIA", referencia),
                           ("DETALLE CON LA PROGRAMACION OPTIMA", optimo)):
        print(titulo)
        print(tabla.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print()

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in (("escenarios", escenarios), ("detalle_referencia", referencia),
                           ("detalle_optimo", optimo), ("curva_ucf", curva)):
        tabla.to_csv(CARPETA_RESULTADOS / f"{nombre}.csv", index=False,
                     float_format="%.9f", encoding="utf-8-sig")
    (CARPETA_RESULTADOS / "resumen_ucf.txt").write_text(resumen, encoding="utf-8")
    fig = graficar(escenarios, p, curva, optimo_kw, costo_optimo, costo_ref, CARPETA_RESULTADOS)
    print(f"Resultados guardados en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICO:
        print("Cerra el grafico para finalizar la ejecucion.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
