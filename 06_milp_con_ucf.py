# -*- coding: utf-8 -*-
"""Primera integracion didactica de UCF al MILP de 24 horas.

Datos, probabilidades y penalizaciones sinteticos.
La suma costo programado + UCF es un criterio de planificacion;
no representa una factura real medida.

Se conserva UN plan de bateria para todos los escenarios.
La red compensa la generacion disponible de cada escenario.
Se supone aprovechamiento completo de la generacion real y una
planta capaz de producir el mayor perfil solar considerado.

La solar programada sigue limitada por el pronostico nominal.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo


FACTORES = np.array([0.70, 1.00, 1.30])
PROBABILIDADES = np.array([0.25, 0.50, 0.25])

# Penalizaciones ficticias de este nuevo ejercicio.
COSTO_FALTANTE = 100.0  # u.m./kWh
COSTO_EXCEDENTE = 10.0  # u.m./kWh

CARPETA = Path(__file__).resolve().parent
RESULTADOS = CARPETA / "resultados_06"


def cargar_base():
    archivo = CARPETA / "02_microrred_24_horas.py"
    if not archivo.exists():
        raise FileNotFoundError(
            "Guarda este archivo junto a 02_microrred_24_horas.py."
        )

    spec = importlib.util.spec_from_file_location("moc_base_06", archivo)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def resolver(base, datos, p, solar_escenarios, incluir_ucf):
    # Reutilizamos las variables binarias, balance y bateria del MILP.
    m = base.crear_modelo(datos, p)
    m.escenarios = pyo.RangeSet(0, len(FACTORES) - 1)

    # La misma bateria debe poder ejecutarse en los tres escenarios.
    # Red positiva = compra; negativa = venta.
    # Esta condicion se agrega a AMBOS modelos.
    m.factibilidad_escenarios = pyo.ConstraintList()

    for k in m.escenarios:
        for t in m.horas:
            red_real = (
                float(datos.loc[t, "demanda_kw"])
                + m.carga[t]
                - m.descarga[t]
                - float(solar_escenarios[k, t])
            )
            m.factibilidad_escenarios.add(
                pyo.inequality(
                    -p["limite_red_kw"],
                    red_real,
                    p["limite_red_kw"],
                )
            )

    if incluir_ucf:
        # Auxiliares para representar los maximos mediante restricciones
        # lineales. m.pv[t] representa ahora la solar PROGRAMADA.
        m.faltante = pyo.Var(
            m.escenarios, m.horas, domain=pyo.NonNegativeReals
        )
        m.excedente = pyo.Var(
            m.escenarios, m.horas, domain=pyo.NonNegativeReals
        )
        m.desvios = pyo.ConstraintList()

        for k in m.escenarios:
            for t in m.horas:
                disponible = float(solar_escenarios[k, t])
                m.desvios.add(
                    m.faltante[k, t] >= m.pv[t] - disponible
                )
                m.desvios.add(
                    m.excedente[k, t] >= disponible - m.pv[t]
                )

        m.ucf = pyo.Expression(
            expr=sum(
                float(PROBABILIDADES[k]) * p["paso_h"] * (
                    COSTO_FALTANTE * m.faltante[k, t]
                    + COSTO_EXCEDENTE * m.excedente[k, t]
                )
                for k in m.escenarios
                for t in m.horas
            )
        )

        costo_programado = m.costo.expr
        m.costo.deactivate()
        m.objetivo_con_ucf = pyo.Objective(
            expr=costo_programado + m.ucf,
            sense=pyo.minimize,
        )

    solver = pyo.SolverFactory("appsi_highs")
    solver.options["mip_rel_gap"] = 1e-8
    resultado = solver.solve(m)

    if not pyo.check_optimal_termination(resultado):
        raise RuntimeError("El solver no encontro una solucion optima.")

    # Extraer el plan y comprobar el balance y la bateria.
    tabla = datos.copy()

    for nombre, variable in (
        ("compra_kw", m.compra),
        ("venta_kw", m.venta),
        ("solar_usada_kw", m.pv),
        ("carga_kw", m.carga),
        ("descarga_kw", m.descarga),
    ):
        tabla[nombre] = [pyo.value(variable[t]) for t in m.horas]

    tabla["energia_inicial_kwh"] = [
        pyo.value(m.energia[t]) for t in m.horas
    ]
    tabla["energia_final_kwh"] = [
        pyo.value(m.energia[t + 1]) for t in m.horas
    ]
    tabla["soc_final_pct"] = (
        100 * tabla["energia_final_kwh"] / p["capacidad_kwh"]
    )
    tabla["solar_no_usada_kw"] = (
        tabla["solar_disponible_kw"] - tabla["solar_usada_kw"]
    )
    tabla["costo_intervalo_um"] = p["paso_h"] * (
        tabla["precio_compra_um_kwh"] * tabla["compra_kw"]
        - tabla["precio_venta_um_kwh"] * tabla["venta_kw"]
    )
    base.validar_resultados(tabla, p)

    # Evaluamos la MISMA UCF en ambos planes, incluso cuando no fue
    # incluida en el objetivo. Esto permite una comparacion consistente.
    programada = tabla["solar_usada_kw"].to_numpy()
    faltante = np.maximum(programada - solar_escenarios, 0)
    excedente = np.maximum(solar_escenarios - programada, 0)

    tabla["ucf_intervalo_um"] = (
        PROBABILIDADES[:, None]
        * (COSTO_FALTANTE * faltante + COSTO_EXCEDENTE * excedente)
    ).sum(axis=0) * p["paso_h"]

    # Verificar y guardar el intercambio real de cada escenario.
    for k in range(len(FACTORES)):
        red_real = (
            tabla["demanda_kw"].to_numpy()
            + tabla["carga_kw"].to_numpy()
            - tabla["descarga_kw"].to_numpy()
            - solar_escenarios[k]
        )
        if np.max(np.abs(red_real)) > p["limite_red_kw"] + 1e-6:
            raise RuntimeError("Un escenario supera el limite de red.")

        tabla[f"red_real_escenario_{k}_kw"] = red_real

    costo = float(tabla["costo_intervalo_um"].sum())
    ucf = float(tabla["ucf_intervalo_um"].sum())

    if incluir_ucf and not np.isclose(
        pyo.value(m.objetivo_con_ucf), costo + ucf,
        atol=1e-4, rtol=0,
    ):
        raise RuntimeError("La UCF del modelo no coincide con su formula.")

    return tabla, {
        "Costo programado": costo,
        "UCF": ucf,
        "Criterio total": costo + ucf,
    }


def main():
    if (
        len(FACTORES) != len(PROBABILIDADES)
        or np.any(FACTORES < 0)
        or np.any(PROBABILIDADES < 0)
        or not np.isclose(PROBABILIDADES.sum(), 1)
        or min(COSTO_FALTANTE, COSTO_EXCEDENTE) <= 0
    ):
        raise ValueError("Revisa escenarios, probabilidades y penalizaciones.")

    base = cargar_base()
    datos = base.DATOS.copy(deep=True).reset_index(drop=True)
    p = base.PARAMETROS.copy()

    # Cada fila representa un perfil diario completo posible.
    solar_escenarios = (
        FACTORES[:, None]
        * datos["solar_disponible_kw"].to_numpy()[None, :]
    )

    sin_ucf, resumen_sin = resolver(
        base, datos, p, solar_escenarios, False
    )
    con_ucf, resumen_con = resolver(
        base, datos, p, solar_escenarios, True
    )

    resumen = pd.DataFrame(
        [resumen_sin, resumen_con],
        index=["Sin UCF en objetivo", "Con UCF en objetivo"],
    )

    # El plan optimizado con UCF debe mejorar o igualar el mismo
    # criterio evaluado sobre el plan que la ignoro.
    if resumen_con["Criterio total"] > resumen_sin["Criterio total"] + 1e-4:
        raise RuntimeError("La comparacion de objetivos no es consistente.")

    print("\nCOMPARACION DE LOS PLANES � unidades monetarias ficticias")
    print(resumen.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nCriterio total = costo programado + penalizacion esperada.")
    print("La UCF de ambos planes usa los mismos escenarios y coeficientes.")
    print("La bateria mantiene un plan comun; la red compensa cada escenario.")
    print("Esta suma es un criterio de planificacion, no una factura medida.")

    RESULTADOS.mkdir(exist_ok=True)
    resumen.to_csv(
        RESULTADOS / "comparacion.csv", encoding="utf-8-sig"
    )

    for nombre, tabla in (("sin_ucf", sin_ucf), ("con_ucf", con_ucf)):
        # En los archivos aclaramos que se trata de solar programada.
        tabla.rename(columns={
            "solar_usada_kw": "solar_programada_kw",
            "solar_no_usada_kw": "margen_no_programado_kw",
        }).to_csv(
            RESULTADOS / f"plan_{nombre}.csv",
            index=False, encoding="utf-8-sig",
        )

    bordes = np.arange(len(datos) + 1) * p["paso_h"]
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 10), layout="constrained"
    )
    fig.suptitle("MILP de 24 horas: primera integracion de UCF")

    axes[0].stairs(
        datos["solar_disponible_kw"], bordes,
        color="gray", linestyle="--", label="Pronostico nominal",
    )

    for tabla, etiqueta, color in (
        (sin_ucf, "Sin UCF en objetivo", "#a42424"),
        (con_ucf, "Con UCF en objetivo", "#18856d"),
    ):
        axes[0].stairs(
            tabla["solar_usada_kw"], bordes,
            label=etiqueta, color=color,
        )
        energia = np.r_[
            tabla["energia_inicial_kwh"].iloc[0],
            tabla["energia_final_kwh"].to_numpy(),
        ]
        axes[1].plot(
            bordes, 100 * energia / p["capacidad_kwh"],
            label=etiqueta, color=color,
        )

    axes[0].set_title("1. Potencia solar que decide programar cada modelo")
    axes[0].set_ylabel("Solar programada (kW)")
    axes[1].set_title("2. Estado de carga de la bateria")
    axes[1].set_ylabel("SOC (%)")
    axes[1].set_xlabel("Hora del dia")

    for ax in axes[:2]:
        ax.set_xlim(bordes[0], bordes[-1])
        ax.legend()
        ax.grid(alpha=0.25)

    x = np.arange(2)
    costos = resumen["Costo programado"].to_numpy()
    penalizaciones = resumen["UCF"].to_numpy()

    axes[2].bar(x, costos, label="Costo programado", color="#a42424")
    barras = axes[2].bar(
        x, penalizaciones, bottom=costos,
        label="UCF esperada", color="#dc8b15",
    )
    axes[2].bar_label(
        barras,
        labels=[f"{v:.2f}" for v in resumen["Criterio total"]],
        padding=4,
    )
    axes[2].set_xticks(x, ["Sin UCF\nen objetivo", "Con UCF\nen objetivo"])
    axes[2].set_title("3. Ambos planes evaluados con el mismo criterio")
    axes[2].set_ylabel("Costo programado + UCF (u.m.)")
    axes[2].legend()

    fig.savefig(RESULTADOS / "comparacion_milp_ucf.png", dpi=160)
    print(f"\nResultados guardados en: {RESULTADOS}")
    plt.show()


if __name__ == "__main__":
    main()