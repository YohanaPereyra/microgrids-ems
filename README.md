# Gestión energética de microrredes con MILP y UCF

**Proyecto de investigación en desarrollo | Python · Pyomo · HiGHS**

Este repositorio reúne los modelos, experimentos y resultados del proyecto MOC sobre gestión energética de una microrred con generación solar fotovoltaica, batería y conexión a la red eléctrica.

El desarrollo avanza desde ejercicios de despacho determinista hasta un sistema de gestión de energía (EMS) basado en programación lineal entera mixta (MILP), con funciones de costo de incertidumbre (UCF, *Uncertainty Cost Functions*) construidas a partir de errores solares históricos. Incluye una evaluación de 109 días y un análisis exploratorio de sensibilidad.

Los resultados corresponden a una microrred simplificada y a los supuestos del caso de estudio. La referencia fotovoltaica de PVGIS es estimada mediante modelos; la demanda, las tarifas y las penalizaciones económicas son ficticias.

## Contenido

| Archivo | Etapa del desarrollo |
| --- | --- |
| [01_microrred_3_horas.py](01_microrred_3_horas.py) | Primer MILP de despacho de una microrred durante tres horas. |
| [02_microrred_24_horas.py](02_microrred_24_horas.py) | Planificación de 24 horas con eficiencias, límites de batería y compra y venta de energía. |
| [02_microrred_24_horas_extra.py](02_microrred_24_horas_extra.py) | Ampliación gráfica del código 02 con el abastecimiento de la demanda por fuente. |
| [03_analisis_sensibilidad.py](03_analisis_sensibilidad.py) | Sensibilidad del despacho determinista a la capacidad de batería y la generación solar. |
| [04_error_pronostico.py](04_error_pronostico.py) | Efecto de un error solar simulado al ejecutar un plan de batería fijo. |
| [05_primera_ucf.py](05_primera_ucf.py) | UCF aislada de un intervalo, formulada como un programa lineal continuo. |
| [06_milp_con_ucf.py](06_milp_con_ucf.py) | Integración de UCF al EMS MILP con escenarios solares sintéticos. |
| [07_validacion_nuevos_dias.py](07_validacion_nuevos_dias.py) | Evaluación de los planes del código 06 sobre 200 nuevos días solares sintéticos. |
| [08_datos_historicos_pvgis.py](08_datos_historicos_pvgis.py) | Preparación de referencias solares de PVGIS y pronósticos de persistencia. |
| [09_calibracion_ucf_historica.py](09_calibracion_ucf_historica.py) | Construcción de escenarios con errores históricos y optimización de una UCF aislada de 24 horas, sin EMS. |
| [10_milp_ucf_historica.py](10_milp_ucf_historica.py) | Integración de los escenarios históricos y la UCF al EMS MILP para un día ilustrativo. |
| [11_validacion_historica_ems.py](11_validacion_historica_ems.py) | Evaluación diaria de los planes sin y con UCF durante los 109 días de prueba. |
| [12_sensibilidad_ems_ucf.py](12_sensibilidad_ems_ucf.py) | Sensibilidad exploratoria a las penalizaciones de faltante y excedente y a la capacidad de batería. |

## Memoria técnica, datos y resultados

La [memoria técnica de los códigos 01 al 12](Explicacion_Codigos_01a12_Proyecto_Microgrids.pdf) explica los conceptos, modelos matemáticos, supuestos, resultados, gráficos y verificaciones de cada etapa. Incluye también la ampliación gráfica del código 02.

- `datos_08/`: respuesta de PVGIS guardada que permite conservar la entrada utilizada.
- `resultados_02/` a `resultados_12/`: salidas de los experimentos, según cada código, con tablas CSV, gráficos, resúmenes y registros de configuración o verificación.
- El código 01 muestra su resultado en consola y no genera una carpeta de resultados.

## Datos y diseño de la evaluación histórica

- **Referencia solar:** PVGIS 5.3 / ERA5, año 2023, ubicación en La Rioja, Argentina, planta de referencia de 5 kWp y horario local UTC-3.
- **Pronóstico:** persistencia diaria; se repite el perfil del día anterior usando los datos anteriores al día que se programa.
- **Calibración:** 254 días, del 02/01/2023 al 12/09/2023.
- **Prueba:** 109 días, del 13/09/2023 al 30/12/2023.
- **Escenarios:** 254 vectores diarios de errores de 24 horas, con igual peso. El banco de calibración permanece fijo durante la prueba.

Los códigos 10 a 12 comparan planes sin y con UCF en el objetivo, bajo las mismas restricciones de factibilidad para todos los escenarios históricos. Esta comparación permite estudiar el efecto de incorporar la penalización esperada al objetivo. El plan sin UCF de estas etapas también tiene restricciones por escenarios; no es un comparador exclusivamente nominal.

Cada plan mantiene una única programación de batería y la red compensa las distintas realizaciones solares. Los días se evalúan de forma independiente: la batería reinicia el SOC configurado y debe alcanzar el SOC final exigido en cada fecha. No se simula una operación continua de la batería entre días ni una reoptimización intradiaria.

## Indicadores económicos

| Indicador | Definición |
| --- | --- |
| Costo programado | Compras menos ventas de energía del plan, valoradas con los precios del modelo. |
| UCF esperada | Penalización media por desviaciones de la programación solar sobre los escenarios de calibración. |
| Criterio esperado | Costo programado + UCF esperada. |
| Criterio realizado | Costo programado + penalización calculada con la referencia solar del día evaluado. |
| Gasto neto de red simulado | Compras menos ventas de red al utilizar toda la referencia solar del día y mantener fijo el plan de batería. |

El criterio realizado y el gasto neto de red son evaluaciones distintas y **no deben sumarse entre sí**. Los criterios con penalización no representan una factura eléctrica medida. Un menor criterio esperado tampoco garantiza mejorar cada día ni reducir el gasto de red.

### Resultados del caso base del código 11

En los 109 días se obtuvieron planes óptimos y factibles al evaluar la red con la referencia solar para ambas alternativas.

| Indicador diario, en unidades monetarias ficticias | Sin UCF en el objetivo | Con UCF en el objetivo |
| --- | ---: | ---: |
| Media del criterio realizado | 1254,91 | 1235,32 |
| P95 del criterio realizado | 2098,57 | 1950,22 |
| Media del gasto neto de red simulado | 1042,83 | 1041,34 |

La reducción media del criterio realizado es aproximadamente del **1,56 %**, mientras que la del gasto de red es del **0,14 %**. El criterio realizado mejora en 53 días y empeora en 56: mejorar el promedio y mejorar en más días son resultados diferentes. El P95 es un percentil de esta muestra, no una garantía para días futuros.

El código 12 estudia siete casos, cambiando un factor por vez. En todos disminuye el criterio realizado medio con UCF; el gasto de red disminuye en cinco y aumenta en dos. Los coeficientes económicos modifican la regla del criterio, por lo que las comparaciones sin/con UCF se realizan dentro de cada caso. El barrido es exploratorio sobre días ya observados; no selecciona parámetros óptimos ni constituye una validación nueva de parámetros elegidos con esos días.

## Requisitos y ejecución

El entorno utilizado durante el desarrollo es Windows con Python 3.14.4, NumPy, SciPy, pandas, Matplotlib, Pyomo y HiGHS mediante `appsi_highs`. Las versiones de las dependencias todavía no están fijadas en un archivo `requirements.txt` validado.

Para obtener el repositorio:

```powershell
git clone https://github.com/YohanaPereyra/microgrids-ems.git
cd microgrids-ems
```

Para crear un entorno e instalar las bibliotecas en Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy pandas matplotlib pyomo highspy
```

Por ejemplo, para ejecutar el primer ejercicio:

```powershell
.\.venv\Scripts\python.exe .\01_microrred_3_horas.py
```

En VS Code también se puede seleccionar `.venv\Scripts\python.exe` como intérprete y ejecutar cada archivo desde el editor.

Para recorrer la etapa histórica, ejecutar los códigos 08 a 12 en orden:

```powershell
.\.venv\Scripts\python.exe .\08_datos_historicos_pvgis.py
.\.venv\Scripts\python.exe .\09_calibracion_ucf_historica.py
.\.venv\Scripts\python.exe .\10_milp_ucf_historica.py
.\.venv\Scripts\python.exe .\11_validacion_historica_ems.py
.\.venv\Scripts\python.exe .\12_sensibilidad_ems_ucf.py
```

Los archivos Python deben conservarse juntos, porque varios códigos importan funciones de etapas anteriores. Los códigos 09 a 12 requieren `datos_08/` y `resultados_08/`; no necesitan consultar internet. El código 08 reutiliza la respuesta PVGIS guardada o consulta la fuente si esa respuesta no está disponible. Las salidas de los códigos 09 y 10 no son necesarias para ejecutar el 11, y el 12 no requiere las salidas del 11.

Con la configuración completa, el código 11 resuelve 218 MILP y el 12 resuelve 1090 MILP, por lo que pueden demorar. Cerrar las ventanas de gráficos cuando el programa lo indique para finalizar la ejecución y continuar con el siguiente archivo.

## Estado y próximos pasos

Está implementado el EMS con UCF histórica, con evaluación de 109 días y sensibilidad a penalizaciones y capacidad de batería. La demanda y los precios siguen siendo ficticios; la referencia solar es estimada y los ensayos corresponden a una ubicación y parte de un año. No se incluyen costos de inversión ni degradación de batería.

Los próximos pasos son:

- Consolidar la reproducibilidad y comprobar la correspondencia entre configuraciones, tablas y resúmenes de los experimentos.
- Completar la sensibilidad prevista sobre precios y supuestos del error de pronóstico.
- Fijar las versiones de las dependencias utilizadas.
- Redactar el informe integrado con discusión de resultados, interpretación económica y limitaciones.
- Revisar con el tutor el alcance final y la contribución de investigación.

## Autora y tutor

**Autora:** Yohana Del Valle Pereyra Flores.  
**Tutor:** Carlos Sánchez Reinoso.  
**Institución:** Universidad Tecnológica Nacional, Facultad Regional La Rioja.  
**Carrera:** Ingeniería Electrónica.
