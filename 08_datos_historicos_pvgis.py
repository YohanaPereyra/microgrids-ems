# -*- coding: utf-8 -*-
"""MOC - Serie solar historica estimada y pronostico diario de persistencia.

Requiere internet la primera vez; guarda y reutiliza la respuesta original.

FUENTE: PVGIS 5.3 / PVGIS-ERA5, Joint Research Centre, European Commission.
https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en
https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-user-manual_en

La referencia es PRODUCCION FV ESTIMADA con meteorologia historica de
reanalisis, no mediciones de una planta ni pronosticos originales de PVGIS.
La planta y ubicacion son un caso de estudio elegido para La Rioja.

Pronostico: al iniciar cada dia, repetir las 24 potencias del dia anterior.
Se supone que cada dato esta disponible al finalizar su intervalo horario.
Cada pronostico se construye solo con datos disponibles en su emision.

La primera parte cronologica se reserva para calibrar errores en el futuro;
la ultima parte se guarda como prueba. Aqui no se ajusta una distribucion,
no se calibra una UCF y no se ejecuta el EMS. Se prepara su entrada solar.
"""

import calendar
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANIO = 2023
LATITUD = -29.413
LONGITUD = -66.856
POTENCIA_PICO_KWP = 5.0
PERDIDAS_PORCENTAJE = 14.0
INCLINACION_GRADOS = 30.0
AZIMUT_PVGIS = -179.0  # Casi norte; en PVGIS 0=sur, +/-180=norte.
UTC_OFFSET_HORAS = -3
FRACCION_CALIBRACION = 0.70
UMBRAL_SOLAR_KW = 0.01
MOSTRAR_GRAFICO = True
CARPETA = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA / "datos_08"
CARPETA_RESULTADOS = CARPETA / "resultados_08"
API = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
TZ_LOCAL = timezone(timedelta(hours=UTC_OFFSET_HORAS))


def parametros_consulta():
    return {
        "lat": LATITUD, "lon": LONGITUD,
        "startyear": ANIO, "endyear": ANIO,
        "pvcalculation": 1, "peakpower": POTENCIA_PICO_KWP,
        "loss": PERDIDAS_PORCENTAJE, "angle": INCLINACION_GRADOS,
        "aspect": AZIMUT_PVGIS, "raddatabase": "PVGIS-ERA5",
        "outputformat": "json", "pvtechchoice": "crystSi",
        "mountingplace": "free", "usehorizon": 1,
    }


def convertir_respuesta(respuesta):
    """Verificar la consulta, unidades, continuidad y conversion UTC -> local."""
    try:
        entradas = respuesta["inputs"]
        unidades = respuesta["meta"]["outputs"]["hourly"]["variables"]["P"]["units"]
        tabla = pd.DataFrame(respuesta["outputs"]["hourly"])
        meteo = entradas["meteo_data"]
        montaje = entradas["mounting_system"]["fixed"]
        esperados = [LATITUD, LONGITUD, POTENCIA_PICO_KWP,
                     PERDIDAS_PORCENTAJE, INCLINACION_GRADOS, AZIMUT_PVGIS]
        recibidos = [entradas["location"]["latitude"], entradas["location"]["longitude"],
                     entradas["pv_module"]["peak_power"], entradas["pv_module"]["system_loss"],
                     montaje["slope"]["value"], montaje["azimuth"]["value"]]
        if (unidades != "W" or meteo["radiation_db"] != "PVGIS-ERA5"
                or meteo["year_min"] != ANIO or meteo["year_max"] != ANIO
                or not np.allclose(recibidos, esperados, atol=1e-7, rtol=0)):
            raise ValueError("La respuesta no coincide con la consulta o con las unidades esperadas.")
        centro_utc = pd.DatetimeIndex(pd.to_datetime(tabla["time"], format="%Y%m%d:%H%M", utc=True))
        cantidad = (366 if calendar.isleap(ANIO) else 365) * 24
        esperado = pd.date_range(f"{ANIO}-01-01 00:30", periods=cantidad, freq="h", tz="UTC")
        if len(centro_utc) != cantidad or not (centro_utc == esperado).all():
            raise ValueError("La serie debe contener todas las horas del anio, ordenadas y sin duplicados.")
        potencia_w = pd.to_numeric(tabla["P"], errors="raise").to_numpy(dtype=float)
        reconstruida = pd.to_numeric(tabla["Int"], errors="raise").to_numpy(dtype=float)
        if (not np.isfinite(potencia_w).all() or np.any(potencia_w < 0)
                or not np.isin(reconstruida, [0, 1]).all()):
            raise ValueError("La serie contiene potencias o indicadores no validos.")
    except (KeyError, TypeError) as exc:
        raise ValueError("La respuesta no tiene el formato PVGIS esperado.") from exc

    # ERA5 identifica la media horaria en HH:30. La guardamos como intervalo
    # [HH:00, HH+1:00), despues de convertir a nuestro horario local fijo.
    inicio_local = centro_utc.tz_convert(TZ_LOCAL) - pd.Timedelta(minutes=30)
    serie = pd.DataFrame({
        "centro_utc": centro_utc,
        "solar_referencia_kw": potencia_w / 1000.0,
        "radiacion_reconstruida": reconstruida.astype(int),
    }, index=inicio_local)
    serie.index.name = "inicio_local"
    return serie


def obtener_serie():
    parametros = parametros_consulta()
    url = API + "?" + urlencode(parametros)
    clave = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    archivo = CARPETA_DATOS / f"pvgis_era5_{ANIO}_{clave}.json"
    registro = archivo.with_name(archivo.stem + "_origen.json")
    descargado = None
    if archivo.exists():
        print("Leyendo la respuesta PVGIS guardada...")
        bruto = archivo.read_bytes()
        if registro.exists():
            descargado = json.loads(registro.read_text(encoding="utf-8")).get("descargado_utc")
    else:
        print("Consultando PVGIS: serie horaria de un anio. Puede tardar unos segundos...")
        try:
            solicitud = Request(url, headers={"User-Agent": "MOC-EMS-investigacion/1.0"})
            with urlopen(solicitud, timeout=45) as respuesta:
                bruto = respuesta.read(10_000_001)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"No se pudo consultar PVGIS. Reintenta con internet disponible.\nConsulta: {url}") from exc
        descargado = datetime.now(timezone.utc).isoformat()
    if len(bruto) > 10_000_000:
        raise ValueError("La respuesta supera el tamanio esperado.")
    contenido = json.loads(bruto)
    serie = convertir_respuesta(contenido)
    origen = {"fuente": "JRC / PVGIS 5.3 / PVGIS-ERA5", "url": url,
              "parametros_solicitados": parametros, "entradas_devuelta_api": contenido["inputs"],
              "descargado_utc": descargado,
              "sha256_respuesta": hashlib.sha256(bruto).hexdigest(),
              "archivo_respuesta": archivo.name}
    if not archivo.exists():
        temporal = archivo.with_suffix(".tmp")
        temporal.write_bytes(bruto)
        temporal.replace(archivo)
        registro.write_text(json.dumps(origen, ensure_ascii=False, indent=2), encoding="utf-8")
    return serie, origen


def preparar_pronosticos(serie):
    """Objetivos de dias completos; origen = mismo intervalo del dia previo."""
    if not 0 < FRACCION_CALIBRACION < 1:
        raise ValueError("La fraccion de calibracion debe estar entre 0 y 1.")
    conteos = serie.groupby(serie.index.normalize()).size()
    completos = pd.DatetimeIndex(conteos.index[conteos == 24]).sort_values()
    dias_validos = completos[(completos - pd.Timedelta(days=1)).isin(completos)]
    if len(dias_validos) < 3:
        raise ValueError("No hay suficientes dias completos con un dia anterior disponible.")
    tabla = serie.loc[serie.index.normalize().isin(dias_validos)].copy()
    origen = tabla.index - pd.Timedelta(days=1)
    tabla["origen_pronostico_local"] = origen
    tabla["emision_pronostico_local"] = tabla.index.normalize()
    tabla["solar_pronostico_kw"] = serie["solar_referencia_kw"].reindex(origen).to_numpy()
    if (not np.isfinite(tabla["solar_pronostico_kw"]).all()
            or not ((origen + pd.Timedelta(hours=1)) <= tabla.index.normalize()).all()):
        raise RuntimeError("El pronostico usa datos que no estaban disponibles al iniciar el dia.")
    tabla["error_kw"] = tabla["solar_referencia_kw"] - tabla["solar_pronostico_kw"]
    n_calibracion = int(np.floor(FRACCION_CALIBRACION * len(dias_validos)))
    n_calibracion = min(max(1, n_calibracion), len(dias_validos) - 1)
    primer_dia_prueba = dias_validos[n_calibracion]
    tabla["periodo"] = np.where(tabla.index < primer_dia_prueba, "calibracion", "prueba")
    tabla["fecha"] = tabla.index.strftime("%Y-%m-%d")
    tabla["hora"] = tabla.index.hour
    info = {"horas_originales": len(serie), "dias_locales_completos": len(completos),
            "dias_locales_incompletos_excluidos": int((conteos != 24).sum()),
            "dias_con_pronostico": len(dias_validos), "dias_calibracion": n_calibracion,
            "dias_prueba": len(dias_validos) - n_calibracion,
            "primera_fecha_prueba": primer_dia_prueba.date().isoformat()}
    return tabla, info


def calcular_metricas(tabla):
    filas = []
    for periodo in ("calibracion", "prueba"):
        parte = tabla.loc[tabla["periodo"] == periodo]
        error = parte["error_kw"].to_numpy()
        solares = ((parte["solar_referencia_kw"] > UMBRAL_SOLAR_KW)
                   | (parte["solar_pronostico_kw"] > UMBRAL_SOLAR_KW))
        e_solar = parte.loc[solares, "error_kw"].to_numpy()
        if len(e_solar) == 0:
            raise ValueError("Un periodo no contiene horas con actividad solar.")
        filas.append({"periodo": periodo, "dias": parte["fecha"].nunique(),
                      "horas": len(parte), "horas_solares": len(e_solar),
                      "mae_24h_kw": float(np.abs(error).mean()),
                      "rmse_24h_kw": float(np.sqrt(np.mean(error ** 2))),
                      "mae_solar_kw": float(np.abs(e_solar).mean()),
                      "rmse_solar_kw": float(np.sqrt(np.mean(e_solar ** 2))),
                      "sesgo_solar_kw": float(e_solar.mean())})
    return pd.DataFrame(filas).set_index("periodo")


def graficar(tabla, carpeta):
    prueba = tabla.loc[tabla["periodo"] == "prueba"]
    fecha = prueba["fecha"].iloc[0]
    ejemplo = prueba.loc[prueba["fecha"] == fecha]
    bordes = np.arange(25)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Historico solar y persistencia")
    fig.suptitle(f"La Rioja - primer dia de prueba: {fecha} (UTC{UTC_OFFSET_HORAS:+d})\n"
                 "Referencia PVGIS estimada; pronostico = perfil del dia anterior", fontsize=13)
    axes[0].stairs(ejemplo["solar_referencia_kw"], bordes, color="#2455a4", linewidth=2,
                   label="Referencia historica estimada (PVGIS)")
    axes[0].stairs(ejemplo["solar_pronostico_kw"], bordes, color="#dc8b15", linewidth=2,
                   linestyle="--", label="Pronostico de persistencia")
    axes[0].set_ylabel("Potencia solar (kW)")
    axes[0].set_title("1. Lo pronosticado al iniciar el dia y la referencia de ese dia", loc="left", fontsize=11)
    axes[0].legend()
    errores = ejemplo["error_kw"].to_numpy()
    axes[1].bar(np.arange(24) + 0.5, errores, width=0.8,
                color=np.where(errores >= 0, "#18856d", "#b04c44"))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("2. Error = referencia - pronostico; negativo significa menos solar de lo previsto",
                      loc="left", fontsize=10.5)
    axes[1].set_ylabel("Error del pronostico (kW)")
    axes[1].set_xlabel("Hora local (cada intervalo dura 1 hora)")
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 2))
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    fig.savefig(carpeta / "pronostico_y_error_pvgis.png", dpi=160)
    return fig


def main():
    serie, origen = obtener_serie()
    tabla, info = preparar_pronosticos(serie)
    metricas = calcular_metricas(tabla)
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    serie.to_csv(CARPETA_RESULTADOS / "serie_horaria_local.csv", encoding="utf-8-sig")
    for periodo in ("calibracion", "prueba"):
        tabla.loc[tabla["periodo"] == periodo].to_csv(
            CARPETA_RESULTADOS / f"pronosticos_{periodo}.csv", encoding="utf-8-sig")
    metricas.to_csv(CARPETA_RESULTADOS / "metricas_pronostico.csv", encoding="utf-8-sig")
    metadatos = {**origen, **info, "zona_local": f"UTC{UTC_OFFSET_HORAS:+d}",
                 "referencia": "Produccion FV estimada, no mediciones de una planta",
                 "convencion_temporal": "ERA5 HH:30 es la media del intervalo HH:00 a HH+1:00",
                 "pronostico": "Al iniciar el dia, repetir el perfil del dia anterior",
                 "disponibilidad_datos": "Se supone cada dato disponible al terminar su hora",
                 "fraccion_calibracion": FRACCION_CALIBRACION,
                 "umbral_hora_solar_kw": UMBRAL_SOLAR_KW,
                 "signo_error": "referencia menos pronostico",
                 "numpy": np.__version__, "pandas": pd.__version__}
    (CARPETA_RESULTADOS / "metadatos_datos_historicos.json").write_text(
        json.dumps(metadatos, ensure_ascii=False, indent=2), encoding="utf-8")
    resumen = (
        "MOC - Preparacion de datos solares historicos y pronostico.\n"
        f"Fuente: PVGIS 5.3 / ERA5; anio {ANIO}; planta de referencia {POTENCIA_PICO_KWP:g} kWp.\n"
        "La referencia es produccion FV estimada con meteorologia historica, no mediciones.\n"
        f"Ubicacion elegida: latitud {LATITUD}, longitud {LONGITUD}; horario UTC{UTC_OFFSET_HORAS:+d}.\n"
        f"Horas originales: {len(serie)}; dias completos con pronostico: {info['dias_con_pronostico']}.\n"
        f"Calibracion: {info['dias_calibracion']} dias; prueba: {info['dias_prueba']} dias.\n"
        f"La prueba comienza el {info['primera_fecha_prueba']}.\n"
        "Se excluyen dias locales incompletos y el primer dia completo sin antecedente completo.\n"
        "Pronostico: repetir el perfil del dia anterior, usando solo datos ya disponibles.\n\n"
        + metricas[["dias", "mae_solar_kw", "rmse_solar_kw", "sesgo_solar_kw"]].to_string(
            float_format=lambda x: f"{x:.3f}") + "\n\n"
        "MAE: error absoluto medio. RMSE: raiz del error cuadratico medio.\n"
        "Las columnas solares incluyen horas con mas de 0.01 kW en referencia o pronostico.\n"
        "Sesgo positivo: la referencia supera al pronostico, en promedio.\n"
        "Estos errores preparan la futura calibracion de escenarios/UCF. Aqui no se ejecuta el EMS.\n"
    )
    print(resumen)
    (CARPETA_RESULTADOS / "resumen_datos_historicos.txt").write_text(resumen, encoding="utf-8")
    fig = graficar(tabla, CARPETA_RESULTADOS)
    print(f"Resultados guardados en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICO:
        print("Cerra el grafico para finalizar la ejecucion.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
