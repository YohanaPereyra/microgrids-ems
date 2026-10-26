# -*- coding: utf-8 -*-
"""MOC - Evaluacion fuera de muestra del EMS con UCF historica.

Se ejecuta la formulacion del 10 en TODOS los dias de prueba del 08.
El banco de errores se fija SOLO con calibracion, sin ajustes con prueba.
Cada dia usa su propio pronostico y los mismos escenarios para ambos planes.
La referencia FV estimada entra solo al evaluar despues de resolver.

Ensayo de DIAS INDEPENDIENTES: cada dia comienza y termina con el SOC
configurado en el 02. No es una simulacion continua de 109 dias ni un MPC.
Dentro de cada dia, la bateria mantiene su programa; la red compensa la FV.
Demanda y tarifas ficticias del 02, repetidas diariamente. cf/ce del 09
siguen siendo supuestos; no se calibran con meteorologia ni con prueba.

Se distinguen C_programado, UCF_esperada, C_programado + penalizacion del
dia y gasto neto de red simulado. Este ultimo se calcula APARTE, sin sumar
otra vez penalizaciones. La FV de referencia no es una medicion de planta.

Los resumenes economicos usan los MISMOS dias con ambos planes optimos y
red factible en la referencia. Se informan por separado todos los fallos.
P95 usa la inversa de la distribucion empirica (no intervalo de confianza):
https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
"""

import gc
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CARPETA = Path(__file__).resolve().parent
SALIDA = CARPETA / "resultados_11"
MOSTRAR_GRAFICOS = True
TOL = 1e-6
PLANES = ("sin_ucf", "con_ucf")
METRICAS = ("costo_programado_um", "ucf_esperada_um", "criterio_esperado_um",
            "penalizacion_realizada_um", "criterio_realizado_um", "gasto_red_simulado_um")


def cargar_10():
    ruta = CARPETA / "10_milp_ucf_historica.py"
    if not ruta.exists():
        raise FileNotFoundError("Falta 10_milp_ucf_historica.py. Guardalo junto al codigo 11.")
    spec = importlib.util.spec_from_file_location("moc10_para_11", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def preparar_periodo(base, historico):
    entrada = CARPETA / "resultados_08"
    ruta = entrada / "metadatos_datos_historicos.json"
    if not ruta.exists():
        raise FileNotFoundError("Falta resultados_08. Ejecuta primero el codigo 08.")
    meta = json.loads(ruta.read_text(encoding="utf-8"))
    if meta["zona_local"] != "UTC-3" or meta["signo_error"] != "referencia menos pronostico":
        raise ValueError("Se requieren el horario y el signo de error del codigo 08.")
    cal = historico.leer_pronosticos(entrada / "pronosticos_calibracion.csv", "calibracion")
    banco = historico.calibrar_errores(cal)
    # El banco queda congelado ANTES de acceder al periodo de prueba.
    prueba = historico.leer_pronosticos(entrada / "pronosticos_prueba.csv", "prueba")
    if (cal["inicio_local"].max() >= prueba["inicio_local"].min()
            or len(banco) != meta["dias_calibracion"]
            or prueba["fecha"].nunique() != meta["dias_prueba"]
            or prueba["fecha"].iloc[0] != meta["primera_fecha_prueba"]):
        raise ValueError("La particion temporal no coincide con el registro del 08.")
    datos = base.DATOS.copy().reset_index(drop=True)
    p = dict(base.PARAMETROS)
    if len(datos) != 24 or not np.array_equal(datos["hora"], np.arange(24)):
        raise ValueError("El 02 debe contener los 24 intervalos de 0 a 23.")
    if not np.isclose(p["paso_h"], 1) or not np.isclose(p["paso_h"], historico.PASO_H):
        raise ValueError("Se requieren intervalos de una hora en los archivos 02 y 09.")
    return datos, p, banco, prueba, meta


def resolver_dia(m10, base, historico, datos, p, escenarios, fecha):
    """Resolver los dos planes sin recibir la referencia solar futura."""
    planes, estados = {}, {}
    for nombre, incluir in (("sin_ucf", False), ("con_ucf", True)):
        try:
            planes[nombre], estados[nombre] = m10.resolver_ems(
                base, historico, datos, p, escenarios, incluir)
        except RuntimeError as error:
            mensaje = str(error)
            # Solo se registra como resultado un rechazo explicito por infactibilidad.
            # Otros errores detienen el ensayo para no ocultar fallas del programa.
            if mensaje.startswith("El MILP no termino con solucion optima:") and "infeasible" in mensaje.lower():
                estados[nombre] = {"terminacion_solver": "infeasible", "detalle": mensaje}
            else:
                raise RuntimeError(f"Fallo en {fecha}, plan {nombre}: {mensaje}") from error
    if len(planes) == 2:
        c0 = planes["sin_ucf"]["costo_programado_intervalo_um"].sum()
        c1 = planes["con_ucf"]["costo_programado_intervalo_um"].sum()
        j0 = c0 + planes["sin_ucf"]["ucf_esperada_intervalo_um"].sum()
        j1 = c1 + planes["con_ucf"]["ucf_esperada_intervalo_um"].sum()
        if c0 > c1 + TOL or j1 > j0 + TOL:
            raise RuntimeError(f"La relacion entre objetivos optimos fallo en {fecha}.")
    return planes, estados


def resumir(diarios):
    """Comparacion pareada: mismos dias, fallos visibles y ninguna imputacion."""
    if diarios.duplicated(["fecha", "plan"]).any():
        raise ValueError("Hay resultados diarios duplicados.")
    pares = diarios.loc[diarios["plan"] == "sin_ucf"].drop(columns="plan").merge(
        diarios.loc[diarios["plan"] == "con_ucf"].drop(columns="plan"),
        on="fecha", how="outer", suffixes=("_sin_ucf", "_con_ucf"), validate="one_to_one")
    pares = pares.sort_values("fecha").reset_index(drop=True)
    if pares[["optimo_sin_ucf", "optimo_con_ucf"]].isna().any().any():
        raise ValueError("Cada fecha debe tener un registro para cada plan.")
    comun = (pares["optimo_sin_ucf"] & pares["optimo_con_ucf"]
             & pares["red_factible_en_referencia_sin_ucf"] & pares["red_factible_en_referencia_con_ucf"])
    pares["dia_comun_valido"] = comun
    for metrica in METRICAS:
        a, b = pares[f"{metrica}_sin_ucf"], pares[f"{metrica}_con_ucf"]
        if not np.isfinite(pares.loc[comun, [f"{metrica}_sin_ucf", f"{metrica}_con_ucf"]].to_numpy(dtype=float)).all():
            raise ValueError("Un dia declarado valido tiene indicadores no finitos.")
        pares[f"delta_{metrica}"] = (b - a).where(comun)
    filas, frecuencias = [], []
    for metrica in METRICAS:
        for nombre in PLANES:
            v = pares.loc[comun, f"{metrica}_{nombre}"].to_numpy(dtype=float)
            filas.append({"metrica": metrica, "plan": nombre, "dias_comunes": len(v),
                "media": float(v.mean()) if len(v) else np.nan,
                "mediana": float(np.median(v)) if len(v) else np.nan,
                "p95": float(np.quantile(v, .95, method="inverted_cdf")) if len(v) else np.nan,
                "maximo": float(v.max()) if len(v) else np.nan,
                "suma_dias_independientes": float(v.sum()) if len(v) else np.nan})
        delta = pares.loc[comun, f"delta_{metrica}"].to_numpy(dtype=float)
        frecuencias.append({"metrica": metrica, "dias_comunes": len(delta),
            "dias_menor_con_ucf": int(np.sum(delta < -TOL)),
            "dias_mayor_con_ucf": int(np.sum(delta > TOL)),
            "dias_iguales": int(np.sum(np.abs(delta) <= TOL)),
            "delta_media_con_menos_sin": float(delta.mean()) if len(delta) else np.nan})
    factibilidad = []
    for nombre in PLANES:
        d = diarios.loc[diarios["plan"] == nombre]
        factibilidad.append({"plan": nombre, "dias_intentados": len(d),
            "dias_con_plan_optimo": int(d["optimo"].sum()),
            "dias_sin_plan_optimo": int((~d["optimo"]).sum()),
            "dias_optimos_red_factible": int((d["optimo"] & d["red_factible_en_referencia"]).sum()),
            "dias_optimos_red_no_factible": int((d["optimo"] & ~d["red_factible_en_referencia"]).sum())})
    return pares, pd.DataFrame(filas), pd.DataFrame(frecuencias), pd.DataFrame(factibilidad)


def crear_resumen(pares, estadisticas, frecuencias, factibilidad, meta):
    n = int(pares["dia_comun_valido"].sum())
    medias = estadisticas.pivot(index="plan", columns="metrica", values="media").reindex(PLANES)
    cola = estadisticas.loc[estadisticas["metrica"].isin(["criterio_realizado_um", "gasto_red_simulado_um"]),
                              ["metrica", "plan", "p95", "maximo"]]
    texto = (f"MOC - Codigo 11: evaluacion de {len(pares)} dias de prueba.\n"
        f"Periodo: {pares['fecha'].iloc[0]} a {pares['fecha'].iloc[-1]}.\n"
        f"Calibracion fija: {meta['dias_calibracion']} dias; sin ajuste con prueba.\n"
        "Demanda, tarifas y penalizaciones ficticias; FV de referencia estimada.\n"
        "Cada dia reinicia el SOC configurado y exige su SOC final; dias independientes.\n\n"
        "FACTIBILIDAD - se informan todos los dias intentados\n"
        + factibilidad.to_string(index=False)
        + f"\n\nDias comunes para comparar costos: {n}/{len(pares)}.\n"
        "MEDIAS DE PLANIFICACION - unidades monetarias ficticias por dia\n"
        + medias[["costo_programado_um", "ucf_esperada_um", "criterio_esperado_um"]].to_string(float_format=lambda v: f"{v:.2f}")
        + "\n\nMEDIAS DE EVALUACION - mismos dias comunes\n"
        + medias[["penalizacion_realizada_um", "criterio_realizado_um", "gasto_red_simulado_um"]].to_string(float_format=lambda v: f"{v:.2f}")
        + "\n\nCOLA DE COSTOS - mismos dias comunes\n"
        + cola.to_string(index=False, float_format=lambda v: f"{v:.2f}")
        + "\n\nCOMPARACION DIARIA - menor, mayor o igual con UCF\n"
        + frecuencias.loc[frecuencias["metrica"].isin(["criterio_realizado_um", "gasto_red_simulado_um"])].to_string(index=False, float_format=lambda v: f"{v:.2f}")
        + "\n\nDelta = con UCF menos sin UCF; negativo favorece al plan con UCF para ese indicador.\n"
        "Criterio realizado = costo programado + penalizacion del dia; no es factura.\n"
        "Gasto neto de red: compras simuladas menos ventas; calculo separado. No sumarlos.\n"
        "P95 es un percentil de esta muestra, no un maximo ni una garantia futura.\n"
        "Las medias excluyen dias sin ambos planes optimos o con red no factible; ver los recuentos.\n"
        "Las sumas corresponden a dias independientes, no a operacion continua de la bateria.\n"
        "Un menor costo medio y ganar mas dias son resultados distintos.\n"
        "La prueba corresponde a un unico lugar y tramo del anio; no demuestra superioridad general.\n")
    return texto


def graficar(pares, estadisticas, carpeta):
    if not pares["dia_comun_valido"].any():
        print("No hay dias comunes validos para graficar una comparacion economica.")
        return []
    colores = {"sin_ucf": "#dc8b15", "con_ucf": "#18856d"}
    etiquetas = {"sin_ucf": "Sin UCF en objetivo", "con_ucf": "Con UCF en objetivo"}
    fechas = pd.to_datetime(pares["fecha"])
    comun = pares["dia_comun_valido"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, layout="constrained")
    fig.canvas.manager.set_window_title("MOC 11 - Evaluacion diaria historica")
    fig.suptitle("Dias de prueba: misma calibracion, planificacion nueva cada dia\n"
                 "Costos ficticios; referencia FV estimada; solo dias comunes validos", fontsize=12)
    for nombre in PLANES:
        axes[0].plot(fechas, pares[f"criterio_realizado_um_{nombre}"].where(comun),
                     color=colores[nombre], label=etiquetas[nombre], linewidth=1.3)
        axes[2].plot(fechas, pares[f"gasto_red_simulado_um_{nombre}"].where(comun),
                     color=colores[nombre], label=etiquetas[nombre], linewidth=1.3)
    delta = pares["delta_criterio_realizado_um"].to_numpy()
    axes[1].bar(fechas, delta, color=np.where(delta <= 0, colores["con_ucf"], colores["sin_ucf"]), width=.85)
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[0].set(title="Costo programado + penalizacion del dia", ylabel="Criterio realizado (u.m.)")
    axes[1].set(title="Diferencia: con UCF menos sin UCF; negativo favorece con UCF", ylabel="Diferencia (u.m.)")
    axes[2].set(title="Gasto neto de red simulado, calculado por separado", ylabel="Gasto de red (u.m.)", xlabel="Fecha del dia evaluado")
    axes[0].legend(fontsize=9);axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.savefig(carpeta / "validacion_historica_diaria.png", dpi=160)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    fig2.canvas.manager.set_window_title("MOC 11 - Media y P95 de los costos")
    fig2.suptitle(f"Comparacion sobre los mismos {int(comun.sum())} dias factibles para ambos planes", fontsize=12)
    for ax, metrica, titulo in zip(axes2, ["criterio_realizado_um", "gasto_red_simulado_um"],
            ["Costo programado + penalizacion", "Gasto neto de red simulado"]):
        x = np.arange(2)
        for j, nombre in enumerate(PLANES):
            row = estadisticas.loc[(estadisticas["plan"] == nombre) & (estadisticas["metrica"] == metrica)].iloc[0]
            bars = ax.bar(x + (j - .5) * .34, [row["media"], row["p95"]], width=.34,
                          color=colores[nombre], label=etiquetas[nombre])
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(x, ["Media diaria", "P95 de la muestra"])
        ax.set(title=titulo, ylabel="Unidades monetarias ficticias")
        ax.margins(y=.2);ax.legend(fontsize=8);ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
    fig2.savefig(carpeta / "resumen_validacion_historica.png", dpi=160)
    return [fig, fig2]


def main():
    inicio = perf_counter()
    m10 = cargar_10()
    base = m10.cargar_modulo("moc02_para_11", "02_microrred_24_horas.py")
    historico = m10.cargar_modulo("moc09_para_11", "09_calibracion_ucf_historica.py")
    plantilla, p, banco, prueba, meta = preparar_periodo(base, historico)
    SALIDA.mkdir(parents=True, exist_ok=True)
    fechas = sorted(prueba["fecha"].unique())
    banco.to_csv(SALIDA / "errores_calibracion_utilizados.csv", encoding="utf-8-sig")
    archivos = ("02_microrred_24_horas.py", "09_calibracion_ucf_historica.py", "10_milp_ucf_historica.py",
        "11_validacion_historica_ems.py", "resultados_08/pronosticos_calibracion.csv",
        "resultados_08/pronosticos_prueba.csv", "resultados_08/metadatos_datos_historicos.json")
    config = {"dias_calibracion": len(banco), "inicio_calibracion": str(banco.index[0]),
        "fin_calibracion": str(banco.index[-1]), "dias_prueba": len(fechas),
        "inicio_prueba": fechas[0], "fin_prueba": fechas[-1], "banco_fijo_sin_recalibracion": True,
        "escenarios_por_dia": len(banco), "parametros_ems": p,
        "coeficientes_asumidos": {"faltante_um_kwh": historico.COSTO_FALTANTE,
                                 "excedente_um_kwh": historico.COSTO_EXCEDENTE},
        "metodo_p95": "inverted_cdf; inversa de distribucion empirica",
        "comparacion_economica": "Mismos dias con ambos planes optimos y red factible en referencia",
        "delta": "con UCF menos sin UCF; negativo indica menor valor con UCF",
        "operacion_bateria": "Dias independientes con SOC inicial y final del 02; programa fijo dentro del dia",
        "pronostico": meta["pronostico"], "disponibilidad_datos": meta["disponibilidad_datos"],
        "referencia_fv": meta["referencia"], "demanda_y_tarifas": "Ficticias del 02, repetidas diariamente",
        "sha256_archivos": {nombre: m10.huella(CARPETA / nombre) for nombre in archivos},
        "sha256_pvgis": meta["sha256_respuesta"],
        "versiones": {"numpy": np.__version__, "pandas": pd.__version__, "pyomo": version("pyomo"), "highspy": version("highspy")},
        "limitaciones": ["Un lugar y un anio; sin condicionamiento estacional de errores",
            "FV estimada; demanda, tarifas y penalizaciones ficticias",
            "No representa operacion continua ni control adaptativo de bateria",
            "No incluye degradacion, potencia reactiva, vertido ni limite AC de inversor",
            "Pueden existir despachos distintos con igual objetivo",
            "No se ajustan parametros mirando los resultados de prueba"]}
    (SALIDA / "configuracion_validacion.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    def guardar_estado(completados, terminado):
        estado = {"dias_objetivo": len(fechas), "dias_completados": completados,
                  "evaluacion_completa": terminado, "segundos": perf_counter() - inicio}
        (SALIDA / "estado_ejecucion.json").write_text(json.dumps(estado, indent=2), encoding="utf-8")
    guardar_estado(0, False)
    print(f"MOC - Codigo 11: {len(fechas)} dias, dos MILP por dia; {len(banco)} escenarios.", flush=True)
    print(f"Prueba: {fechas[0]} a {fechas[-1]}. Calibracion fija hasta {banco.index[-1]}.", flush=True)
    print("Dias independientes; demanda y tarifas ficticias; FV estimada. Puede demorar.", flush=True)
    filas, horarios, verificaciones = [], [], []
    for i, fecha in enumerate(fechas, start=1):
        dia = prueba.loc[prueba["fecha"] == fecha].copy().reset_index(drop=True)
        altura = historico.leer_geometria_ejemplo(dia, meta)
        datos = plantilla.copy()
        datos["solar_disponible_kw"] = dia["solar_pronostico_kw"].to_numpy()
        escenarios = historico.construir_escenarios(datos["solar_disponible_kw"].to_numpy(), banco.to_numpy(), altura)
        planes, estados = resolver_dia(m10, base, historico, datos, p, escenarios, fecha)
        # Solo despues de elegir AMBOS planes se evalua la referencia del dia.
        for nombre in PLANES:
            verificaciones.append({"fecha": fecha, "plan": nombre, **estados[nombre]})
            fila = {"fecha": fecha, "plan": nombre, "optimo": nombre in planes,
                    "estado_solver": estados[nombre]["terminacion_solver"],
                    "red_factible_en_referencia": False, **{k: np.nan for k in METRICAS}}
            if nombre in planes:
                tabla, valores = m10.evaluar_referencia(planes[nombre], dia["solar_referencia_kw"].to_numpy(),
                    p, historico.COSTO_FALTANTE, historico.COSTO_EXCEDENTE)
                tabla.insert(0, "plan", nombre);tabla.insert(0, "fecha", fecha)
                horarios.append(tabla)
                fila.update(valores)
            filas.append(fila)
        pd.DataFrame(filas).to_csv(SALIDA / "resultados_diarios.csv", index=False, encoding="utf-8-sig")
        guardar_estado(i, False)
        if i == 1 or i % 10 == 0 or i == len(fechas):
            print(f"Dias completados: {i}/{len(fechas)}; ultimo {fecha}; {perf_counter() - inicio:.1f} s.", flush=True)
            gc.collect()
    diarios = pd.DataFrame(filas)
    pares, estadisticas, frecuencias, factibilidad = resumir(diarios)
    for nombre, tabla in (("comparacion_por_dia.csv", pares), ("resumen_metricas.csv", estadisticas),
            ("frecuencia_mejoras.csv", frecuencias), ("factibilidad.csv", factibilidad),
            ("verificaciones_planes.csv", pd.DataFrame(verificaciones))):
        tabla.to_csv(SALIDA / nombre, index=False, encoding="utf-8-sig")
    if horarios:
        pd.concat(horarios, ignore_index=True).to_csv(SALIDA / "planes_horarios.csv", index=False, encoding="utf-8-sig")
    texto = crear_resumen(pares, estadisticas, frecuencias, factibilidad, meta)
    (SALIDA / "resumen_validacion.txt").write_text(texto, encoding="utf-8")
    figuras = graficar(pares, estadisticas, SALIDA)
    guardar_estado(len(fechas), True)
    print("\n" + texto)
    print(f"Resultados guardados en: {SALIDA}")
    if MOSTRAR_GRAFICOS and figuras:
        print("Cerra las dos ventanas de graficos para finalizar.")
        plt.show()
    else:
        for fig in figuras:plt.close(fig)


if __name__ == "__main__":
    main()
