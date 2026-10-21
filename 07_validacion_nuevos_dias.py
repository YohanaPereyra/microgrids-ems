# -*- coding: utf-8 -*-
"""MOC - Evaluar los planes del archivo 06 en nuevos dias solares sinteticos.

Guardar junto a 02_microrred_24_horas.py y 06_milp_con_ucf.py. Ejecutar 06
primero: se leen sus CSV de resultados_06. Aqui NO se resuelve ningun MILP
ni se cambia la bateria despues de conocer la generacion de prueba.

Se aplica la misma muestra a ambos planes. La generacion real simulada
se aprovecha por completo y la red compensa, como en el archivo 06. Si un
plan requiere superar el limite de red, ese dia se marca no factible.
Los promedios comparativos usan solo dias factibles para AMBOS planes.

DOS EVALUACIONES DIFERENTES (no se suman entre si):
1. Criterio del 06 fuera de sus escenarios: costo PROGRAMADO fijo mas
   penalizacion realizada por desvio. La media estima ese criterio bajo
   la distribucion de prueba; no esta garantizado que mejore.
2. Gasto de red simulado: compras REALES SIMULADAS menos ventas, a las
   tarifas originales del 02. Sustituye el costo programado en esta otra
   evaluacion; no se le agregan el costo programado ni la penalizacion.

La distribucion de prueba es un supuesto nuevo, no mediciones ni una
validacion empirica: factor diario uniforme entre 0.7 y 1.3, ruido horario
correlacionado de desviacion 0.08 y factores finales entre 0.6 y 1.4.
Se supone que la planta puede producir hasta 1.4 veces el pico previsto.
La semilla, parametros y perfiles se guardan para reproducir el ensayo.
"""

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NUM_DIAS = 200
SEMILLA = 2026
DESVIACION_HORARIA = 0.08
CORRELACION_HORARIA = 0.70
MOSTRAR_GRAFICO = True
CARPETA = Path(__file__).resolve().parent
CARPETA_RESULTADOS = CARPETA / "resultados_07"


def cargar_contexto():
    archivo = CARPETA / "06_milp_con_ucf.py"
    if not archivo.exists():
        raise FileNotFoundError("Guarda el archivo 07 junto a 06_milp_con_ucf.py y al 02.")
    spec = importlib.util.spec_from_file_location("moc_06_para_validacion", archivo)
    modelo06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modelo06)
    base = modelo06.cargar_base()
    datos = base.DATOS.copy(deep=True).reset_index(drop=True)
    p = base.PARAMETROS.copy()
    factores = np.asarray(modelo06.FACTORES, dtype=float)
    probabilidades = np.asarray(modelo06.PROBABILIDADES, dtype=float)
    cf = float(modelo06.COSTO_FALTANTE)
    ce = float(modelo06.COSTO_EXCEDENTE)
    if (factores.shape != probabilidades.shape or factores.ndim != 1
            or len(factores) == 0 or not np.isfinite(factores).all()
            or not np.isfinite(probabilidades).all()
            or np.any(factores < 0) or np.any(probabilidades < 0)
            or not np.isclose(probabilidades.sum(), 1)
            or not np.isfinite([cf, ce]).all() or min(cf, ce) <= 0):
        raise ValueError("Revisa los escenarios y coeficientes del archivo 06.")
    entrenamiento = factores[:, None] * datos["solar_disponible_kw"].to_numpy()[None, :]
    planes = {}
    columnas_datos = ["hora", "demanda_kw", "solar_disponible_kw",
                     "precio_compra_um_kwh", "precio_venta_um_kwh"]
    for nombre in ("sin_ucf", "con_ucf"):
        ruta = CARPETA / "resultados_06" / f"plan_{nombre}.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta.name}. Ejecuta primero el archivo 06.")
        tabla = pd.read_csv(ruta).rename(columns={
            "solar_programada_kw": "solar_usada_kw",
            "margen_no_programado_kw": "solar_no_usada_kw",
        })
        requeridas = columnas_datos + ["compra_kw", "venta_kw", "solar_usada_kw",
            "solar_no_usada_kw", "carga_kw", "descarga_kw", "energia_inicial_kwh",
            "energia_final_kwh", "soc_final_pct", "costo_intervalo_um", "ucf_intervalo_um"]
        if (not set(requeridas).issubset(tabla.columns) or len(tabla) != len(datos)
                or not np.isfinite(tabla[requeridas].to_numpy(dtype=float)).all()):
            raise ValueError(f"Revisa {ruta.name}; vuelve a ejecutar el 06.")
        if not np.allclose(tabla[columnas_datos], datos[columnas_datos], atol=1e-7, rtol=0):
            raise ValueError("Los datos del 02 cambiaron desde el 06. Ejecuta nuevamente el 06.")
        base.validar_resultados(tabla, p)
        costo = p["paso_h"] * (tabla["precio_compra_um_kwh"] * tabla["compra_kw"]
                               - tabla["precio_venta_um_kwh"] * tabla["venta_kw"])
        s = tabla["solar_usada_kw"].to_numpy()
        ucf = (probabilidades[:, None] * (cf * np.maximum(s - entrenamiento, 0)
               + ce * np.maximum(entrenamiento - s, 0))).sum(axis=0) * p["paso_h"]
        if (not np.allclose(costo, tabla["costo_intervalo_um"], atol=1e-6, rtol=0)
                or not np.allclose(ucf, tabla["ucf_intervalo_um"], atol=1e-6, rtol=0)):
            raise ValueError("Los coeficientes o resultados del 06 cambiaron. Ejecuta nuevamente el 06.")
        planes[nombre] = tabla
    return modelo06, datos, p, planes


def generar_dias(pronostico, n_dias=NUM_DIAS, semilla=SEMILLA):
    """Muestra nueva, compartida por ambos planes; no se usa para optimizar."""
    if n_dias < 1 or not 0 <= CORRELACION_HORARIA < 1 or DESVIACION_HORARIA < 0:
        raise ValueError("Revisa cantidad de dias, correlacion y desviacion horaria.")
    rng = np.random.default_rng(semilla)
    factor_diario = rng.uniform(0.7, 1.3, size=(n_dias, 1))
    ruido = rng.normal(0, DESVIACION_HORARIA, size=(n_dias, len(pronostico)))
    rho = CORRELACION_HORARIA
    for t in range(1, len(pronostico)):
        ruido[:, t] = rho * ruido[:, t - 1] + np.sqrt(1 - rho ** 2) * ruido[:, t]
    factores = np.clip(factor_diario + ruido, 0.6, 1.4)
    return factores * np.asarray(pronostico, dtype=float)[None, :]


def evaluar_plan(tabla, solares, p, cf, ce):
    """Misma bateria todos los dias. Sin recorte PV ni carga no abastecida."""
    solares = np.asarray(solares, dtype=float)
    if (solares.ndim != 2 or solares.shape[1] != len(tabla)
            or not np.isfinite(solares).all() or np.any(solares < 0)):
        raise ValueError("Los perfiles solares deben ser una matriz de dias por horas, no negativa.")
    demanda = tabla["demanda_kw"].to_numpy()
    carga = tabla["carga_kw"].to_numpy()
    descarga = tabla["descarga_kw"].to_numpy()
    programada = tabla["solar_usada_kw"].to_numpy()
    red = demanda + carga - descarga - solares
    factible = (np.abs(red) <= p["limite_red_kw"] + 1e-6).all(axis=1)
    compras = np.maximum(red, 0)
    ventas = np.maximum(-red, 0)
    balance = compras + solares + descarga - demanda - carga - ventas
    if np.max(np.abs(balance)) > 1e-6:
        raise RuntimeError("El balance de ejecucion no cuadra.")
    costo_red = p["paso_h"] * (
        compras * tabla["precio_compra_um_kwh"].to_numpy()
        - ventas * tabla["precio_venta_um_kwh"].to_numpy()
    ).sum(axis=1)
    penalizacion = p["paso_h"] * (
        cf * np.maximum(programada - solares, 0)
        + ce * np.maximum(solares - programada, 0)
    ).sum(axis=1)
    costo_programado = float(tabla["costo_intervalo_um"].sum())
    return pd.DataFrame({
        "dia_prueba": np.arange(1, len(solares) + 1),
        "factible": factible,
        "max_red_requerida_kw": np.max(np.abs(red), axis=1),
        "solar_simulada_kwh": solares.sum(axis=1) * p["paso_h"],
        "costo_programado_um": costo_programado,
        "penalizacion_realizada_um": penalizacion,
        "criterio_06_realizado_um": costo_programado + penalizacion,
        # No presentar compras imposibles como un gasto factible.
        "costo_red_simulado_um": np.where(factible, costo_red, np.nan),
    })


def resumir(evaluaciones):
    comunes = evaluaciones["sin_ucf"]["factible"] & evaluaciones["con_ucf"]["factible"]
    if not comunes.any():
        raise RuntimeError("Ningun dia es factible para ambos planes; revisa el limite de red y los planes.")
    filas = []
    for nombre, tabla in evaluaciones.items():
        muestra = tabla.loc[comunes]
        filas.append({
            "plan": nombre,
            "dias_factibles": int(tabla["factible"].sum()),
            "dias_comparados": int(comunes.sum()),
            "costo_programado_um": float(muestra["costo_programado_um"].iloc[0]),
            "penalizacion_media_um": float(muestra["penalizacion_realizada_um"].mean()),
            "criterio_06_medio_um": float(muestra["criterio_06_realizado_um"].mean()),
            "criterio_06_p95_um": float(muestra["criterio_06_realizado_um"].quantile(0.95)),
            "costo_red_medio_um": float(muestra["costo_red_simulado_um"].mean()),
        })
    return pd.DataFrame(filas).set_index("plan"), comunes


def graficar(datos, solares, resumen, p, carpeta):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Validacion con nuevos dias")
    fig.suptitle(f"Prueba con {len(solares)} dias sinteticos nuevos - planes fijos", fontsize=14)
    bordes = np.arange(len(datos) + 1) * p["paso_h"]
    axes[0].stairs(datos["solar_disponible_kw"], bordes, label="Pronostico nominal",
                   color="#777777", linestyle="--", linewidth=2)
    axes[0].stairs(solares[0], bordes, label="Primer dia de prueba", color="#dc8b15", linewidth=2)
    axes[0].set_title("1. Un ejemplo de los nuevos perfiles (se muestra el primero)", loc="left", fontsize=11)
    axes[0].set_xlabel("Hora del dia")
    axes[0].set_ylabel("Potencia solar (kW)")
    axes[0].set_xlim(bordes[0], bordes[-1])
    axes[0].legend()
    axes[0].grid(alpha=0.22)
    etiquetas = ["Plan sin UCF", "Plan con UCF"]
    for ax, columna, titulo in (
        (axes[1], "criterio_06_medio_um", "2. Criterio del 06 en los nuevos dias: costo programado + penalizacion"),
        (axes[2], "costo_red_medio_um", "3. Gasto neto de red simulado: compras menos ventas"),
    ):
        valores = resumen.loc[["sin_ucf", "con_ucf"], columna].to_numpy()
        barras = ax.bar(etiquetas, valores, color=["#2455a4", "#18856d"], width=0.5)
        ax.bar_label(barras, labels=[f"{v:.2f}" for v in valores], padding=5)
        ax.set_ylim(min(0, float(valores.min()) * 1.15), max(1, float(valores.max()) * 1.18))
        ax.set_title(titulo, loc="left", fontsize=10.5)
        ax.set_ylabel("Media por dia (u.m.)")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.savefig(carpeta / "validacion_nuevos_dias.png", dpi=160)
    return fig


def main():
    modelo06, datos, p, planes = cargar_contexto()
    cf = float(modelo06.COSTO_FALTANTE)
    ce = float(modelo06.COSTO_EXCEDENTE)
    solares = generar_dias(datos["solar_disponible_kw"].to_numpy())
    evaluaciones = {nombre: evaluar_plan(tabla, solares, p, cf, ce)
                    for nombre, tabla in planes.items()}
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in evaluaciones.items():
        tabla.to_csv(CARPETA_RESULTADOS / f"dias_{nombre}.csv", index=False, encoding="utf-8-sig")
    resumen, comunes = resumir(evaluaciones)
    sin = evaluaciones["sin_ucf"].loc[comunes]
    con = evaluaciones["con_ucf"].loc[comunes]
    mejora_criterio = sin["criterio_06_realizado_um"] - con["criterio_06_realizado_um"]
    mejora_red = sin["costo_red_simulado_um"] - con["costo_red_simulado_um"]
    texto = (
        f"MOC - Prueba con {NUM_DIAS} dias sinteticos nuevos; semilla {SEMILLA}.\n"
        "Ambos planes se leen del 06 y se ejecutan sin reoptimizar.\n"
        "La muestra es identica para ambos; no son mediciones.\n"
        f"Dias factibles: sin UCF {int(evaluaciones['sin_ucf']['factible'].sum())}/{NUM_DIAS}; "
        f"con UCF {int(evaluaciones['con_ucf']['factible'].sum())}/{NUM_DIAS}.\n"
        f"Comparacion sobre {int(comunes.sum())} dias factibles para ambos.\n\n"
        + resumen.to_string(float_format=lambda x: f"{x:.2f}") + "\n\n"
        f"Diferencia media del criterio (sin - con): {mejora_criterio.mean():.2f} u.m.\n"
        f"Diferencia media del gasto de red (sin - con): {mejora_red.mean():.2f} u.m.\n"
        "Diferencia positiva: mejora con UCF; negativa: aumento con UCF.\n"
        f"Dias con menor criterio con UCF: {int((mejora_criterio > 1e-6).sum())}/{int(comunes.sum())}.\n"
        "El criterio del 06 y el gasto de red son evaluaciones distintas: no se suman.\n"
        "Una UCF no garantiza mejorar todos los dias ni bajo otra distribucion.\n"
        "Estos resultados corresponden a la distribucion sintetica de prueba, no a ahorro real demostrado.\n"
    )
    print(texto)
    resumen.to_csv(CARPETA_RESULTADOS / "resumen_validacion.csv", encoding="utf-8-sig")
    (CARPETA_RESULTADOS / "resumen_validacion.txt").write_text(texto, encoding="utf-8")
    perfiles = pd.DataFrame(solares, columns=[f"solar_intervalo_{t}_kw" for t in range(len(datos))])
    perfiles.insert(0, "dia_prueba", np.arange(1, NUM_DIAS + 1))
    perfiles.to_csv(CARPETA_RESULTADOS / "perfiles_solares_prueba.csv", index=False, encoding="utf-8-sig")
    configuracion = {"dias": NUM_DIAS, "semilla": SEMILLA, "numpy": np.__version__,
                     "factor_diario": [0.7, 1.3], "factor_horario_limites": [0.6, 1.4],
                     "desviacion_horaria": DESVIACION_HORARIA,
                     "correlacion_horaria": CORRELACION_HORARIA,
                     "costo_faltante_um_kwh": cf, "costo_excedente_um_kwh": ce,
                     "parametros_microrred": p}
    (CARPETA_RESULTADOS / "configuracion_prueba.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8")
    fig = graficar(datos, solares, resumen, p, CARPETA_RESULTADOS)
    print(f"Resultados guardados en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICO:
        print("Cerra el grafico para finalizar la ejecucion.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
