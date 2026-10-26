# -*- coding: utf-8 -*-
"""MOC - Analisis de sensibilidad del EMS determinista.


ESTUDIO A: capacidad nominal de bateria = 5, 10, 15 y 20 kWh.
Se mantienen la potencia maxima de 3 kW, eficiencias, fracciones de SOC,
demanda, perfil solar y precios del caso de referencia.

ESTUDIO B: produccion solar = 50 %, 75 % y 100 % del perfil de referencia.
Se mantienen la bateria de 10 kWh y todos los demas parametros.

Cada caso se vuelve a optimizar con su perfil solar conocido de antemano.
Son escenarios deterministas; este código no evalua errores de pronostico
ni incorpora UCF. Los datos y precios siguen siendo sinteticos.

Los costos son de operacion. No incluyen compra ni degradacion de bateria;
este analisis no determina la capacidad economicamente optima de inversion.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo


CAPACIDADES_KWH = [5.0, 10.0, 15.0, 20.0]
FACTORES_SOLARES = [0.50, 0.75, 1.00]
MOSTRAR_GRAFICO = True
ARCHIVO_BASE = Path(__file__).resolve().with_name("02_microrred_24_horas.py")
CARPETA_RESULTADOS = Path(__file__).resolve().parent / "resultados_03"


def cargar_modelo_base():
    """Importar el archivo 02 sin ejecutar su funcion main."""
    if not ARCHIVO_BASE.exists():
        raise FileNotFoundError(
            "Guarda 03_analisis_sensibilidad.py y 02_microrred_24_horas.py "
            "en la misma carpeta. No se encontro: " + str(ARCHIVO_BASE)
        )
    spec = importlib.util.spec_from_file_location("modelo_base_moc", ARCHIVO_BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def evaluar_caso(base, datos, parametros, estudio, factor_solar, nombre, carpeta):
    """Resolver, verificar y registrar un escenario completo."""
    modelo, despacho = base.resolver_modelo(datos, parametros)
    errores = base.validar_resultados(despacho, parametros)
    costo = float(pyo.value(modelo.costo))
    referencia = base.referencia_sin_bateria(datos, parametros)
    if costo > referencia + 1e-6:
        raise RuntimeError("El costo supera la alternativa factible sin usar la bateria.")
    paso = parametros["paso_h"]
    energia = np.r_[despacho["energia_inicial_kwh"].iloc[0], despacho["energia_final_kwh"]]
    resultado = {
        "estudio": estudio,
        "capacidad_kwh": parametros["capacidad_kwh"],
        "potencia_carga_max_kw": parametros["potencia_carga_kw"],
        "potencia_descarga_max_kw": parametros["potencia_descarga_kw"],
        "solar_pct": 100 * factor_solar,
        "costo_con_bateria_um": costo,
        "costo_sin_bateria_um": referencia,
        "reduccion_costo_um": referencia - costo,
        "reduccion_costo_pct": 100 * (referencia - costo) / referencia,
        "compra_kwh": float(despacho["compra_kw"].sum() * paso),
        "venta_kwh": float(despacho["venta_kw"].sum() * paso),
        "carga_bateria_kwh": float(despacho["carga_kw"].sum() * paso),
        "descarga_bateria_kwh": float(despacho["descarga_kw"].sum() * paso),
        "soc_min_observado_pct": 100 * float(energia.min()) / parametros["capacidad_kwh"],
        "soc_max_observado_pct": 100 * float(energia.max()) / parametros["capacidad_kwh"],
        "soc_final_pct": float(despacho["soc_final_pct"].iloc[-1]),
        "max_error_balance_kw": errores["balance_kw"],
        "max_error_bateria_kwh": errores["evolucion_kwh"],
    }
    despacho.to_csv(carpeta / f"despacho_{nombre}.csv", index=False,
                    float_format="%.9f", encoding="utf-8-sig")
    return resultado


def ejecutar_estudios(base, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    capacidad = []
    solar = []
    for valor in CAPACIDADES_KWH:
        parametros = base.PARAMETROS.copy()
        parametros["capacidad_kwh"] = valor
        datos = base.DATOS.copy(deep=True)
        print(f"Resolviendo bateria de {valor:g} kWh...", flush=True)
        capacidad.append(evaluar_caso(
            base, datos, parametros, "capacidad", 1.0,
            f"capacidad_{valor:g}kwh", carpeta,
        ))
    for factor in FACTORES_SOLARES:
        parametros = base.PARAMETROS.copy()
        datos = base.DATOS.copy(deep=True)
        datos["solar_disponible_kw"] *= factor
        print(f"Resolviendo produccion solar del {100 * factor:g} %...", flush=True)
        solar.append(evaluar_caso(
            base, datos, parametros, "solar", factor,
            f"solar_{100 * factor:g}pct", carpeta,
        ))
    return pd.DataFrame(capacidad), pd.DataFrame(solar)


def graficar(capacidad, solar, carpeta):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Sensibilidad del costo operativo")
    fig.suptitle("Sensibilidad del EMS determinista - dia sintetico de 24 horas", fontsize=14)
    axes[0].plot(capacidad["capacidad_kwh"], capacidad["costo_con_bateria_um"],
                 "o-", color="#18856d", linewidth=2, label="Con bateria")
    axes[0].plot(capacidad["capacidad_kwh"], capacidad["costo_sin_bateria_um"],
                 "--", color="#666666", linewidth=1.8, label="Sin bateria")
    axes[0].set_title("A. Cambia solo la capacidad de bateria\nPotencia fija: 3 kW; solar: 100 %", fontsize=11)
    axes[0].set_xlabel("Capacidad nominal de bateria (kWh)")
    axes[0].set_xticks(capacidad["capacidad_kwh"])

    axes[1].plot(solar["solar_pct"], solar["costo_con_bateria_um"],
                 "o-", color="#18856d", linewidth=2, label="Con bateria")
    axes[1].plot(solar["solar_pct"], solar["costo_sin_bateria_um"],
                 "o--", color="#666666", linewidth=1.8, label="Sin bateria")
    axes[1].set_title("B. Cambia solo el perfil de generacion solar\nBateria fija: 10 kWh y 3 kW", fontsize=11)
    axes[1].set_xlabel("Produccion solar respecto del perfil de referencia (%)")
    axes[1].set_xticks(solar["solar_pct"])
    for ax in axes:
        ax.set_ylabel("Costo operativo del dia (u.m.)")
        ax.grid(alpha=0.25)
        ax.legend()
        ax.set_ylim(bottom=0)
    fig.savefig(carpeta / "sensibilidad_costos.png", dpi=160)
    return fig


def main():
    base = cargar_modelo_base()
    capacidad, solar = ejecutar_estudios(base, CARPETA_RESULTADOS)
    columnas = ["capacidad_kwh", "solar_pct", "costo_con_bateria_um",
                "costo_sin_bateria_um", "reduccion_costo_um"]
    print("\nESTUDIO A: CAMBIA SOLO LA CAPACIDAD DE BATERIA")
    print(capacidad[columnas].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nESTUDIO B: CAMBIA SOLO LA GENERACION SOLAR")
    print(solar[columnas].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    capacidad.to_csv(CARPETA_RESULTADOS / "sensibilidad_capacidad.csv", index=False,
                     float_format="%.9f", encoding="utf-8-sig")
    solar.to_csv(CARPETA_RESULTADOS / "sensibilidad_solar.csv", index=False,
                 float_format="%.9f", encoding="utf-8-sig")
    resumen = (
        "MOC - Sensibilidad del EMS determinista\n"
        "Datos y precios sinteticos; cada escenario se conoce antes de optimizar.\n"
        "Cada costo con bateria se compara con su propio caso sin bateria.\n"
        "A: cambia solo la capacidad; potencia fija de 3 kW y solar al 100 %.\n"
        "B: cambia solo la produccion solar; bateria fija de 10 kWh y 3 kW.\n"
        "SOC inicial y final: 50 %; limites: 10-90 %; eficiencias: 95 % cada una.\n"
        "Costos operativos: no incluyen inversion ni degradacion.\n"
        "Este estudio aun no incorpora UCF ni errores de pronostico.\n\n"
        "Estudio A\n" + capacidad[columnas].to_string(index=False)
        + "\n\nEstudio B\n" + solar[columnas].to_string(index=False) + "\n"
    )
    (CARPETA_RESULTADOS / "resumen_sensibilidad.txt").write_text(resumen, encoding="utf-8")
    fig = graficar(capacidad, solar, CARPETA_RESULTADOS)
    print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICO:
        print("Cerra el grafico para finalizar la ejecucion.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
