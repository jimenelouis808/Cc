# scalpbot — micro-scalping en Binance con machine learning

Sistema completo y ejecutable: adquisición de datos, features de análisis
técnico + microestructura + contexto macro, etiquetado por triple barrera,
modelo con validación temporal purgada, backtest con costes reales y ejecución
en papel / testnet / mainnet.

---

## Lee esto antes que nada

El micro-scalping no es difícil por la parte de predicción. Es difícil por la
aritmética de los costes, y conviene verla con números antes de escribir una
línea de código.

En Binance USDT-M futures, nivel VIP0:

| Concepto | Coste |
|---|---|
| Comisión taker de entrada | 5.0 bps |
| Comisión taker de salida | 5.0 bps |
| Deslizamiento (2 lados, optimista) | 2.0 bps |
| **Ida y vuelta** | **12.0 bps = 0.12%** |

Con 50 operaciones al día son **6% del nocional pagado en costes cada día**.
Para no perder dinero, tu modelo tiene que acertar la dirección con una ventaja
bruta superior a 12 bps por operación, de forma sostenida, después de todos los
efectos de sobreajuste.

Para calibrar la dificultad: en el pipeline de este repositorio, con datos de
prueba que contienen una señal predecible **plantada a propósito**, el modelo
alcanza 52.9% de acierto y **+9.12 bps brutos** por operación. Es una habilidad
predictiva real y aun así **pierde dinero**, porque 9.12 < 12.

```
  Tasa de acierto          52.9%
  Media por trade          -2.88 bps      <- resultado neto
  Edge bruto               +9.12 bps      <- lo que predice el modelo
  Coste ida y vuelta       12.00 bps      <- lo que cuesta operar
```

Esa es la barrera real. Este repositorio está construido para medirla con
honestidad, no para esquivarla. Las tres palancas que de verdad la mueven son:

1. **Ser mucho más selectivo.** Operar solo cuando el valor esperado es alto.
   El comando `sweep` mide exactamente esto.
2. **Ser maker en vez de taker.** Con órdenes límite pasas de 12 a 6 bps de
   ida y vuelta. Es la mejora individual más grande disponible, y a cambio
   asumes riesgo de no ejecución. En el pipeline, sobre exactamente las mismas
   predicciones, `--maker` mueve el resultado de **−3.31% a +5.53%**
   (`t = 3.47`): no cambia el modelo, cambia la estructura de costes.
3. **Bajar de nivel VIP / usar rebates.** Estructural, no algorítmico.

```bash
python -m scalpbot backtest            #  12 bps ida y vuelta  ->  -1.05 bps/trade
python -m scalpbot backtest --maker    #   6 bps ida y vuelta  ->  +2.00 bps/trade
```

El modo `--maker` es deliberadamente optimista: asume que **todas** tus órdenes
límite se ejecutan. En la realidad una parte no se llena, y suelen ser
justamente las de los movimientos que querías capturar. Trátalo como cota
superior, no como previsión.

Si un tutorial de bots de scalping no te enseña esta tabla en la primera
página, no está describiendo el problema real.

---

## Instalación

```bash
git clone <tu-repo> && cd scalpbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# opcional pero recomendado: pip install pyarrow matplotlib pytest
```

Requiere Python 3.10 o superior. No necesitas TA-Lib: todos los indicadores
están implementados en pandas puro.

## Arranque rápido (5 minutos, sin cuenta de Binance)

```bash
python -m scalpbot all --bars 50000
```

Esto genera un mercado sintético, construye las features, entrena con
walk-forward purgado, backtestea sobre predicciones out-of-sample e imprime el
informe. Sirve para verificar que todo funciona en tu máquina.

> Los datos sintéticos validan el **código**, nunca la **rentabilidad**.
> Cualquier número que salga de aquí no dice nada sobre el mercado real.

## Con datos reales de Binance

```bash
python -m scalpbot download --symbol BTCUSDT --interval 1m --days 90
python -m scalpbot features
python -m scalpbot train
python -m scalpbot backtest
python -m scalpbot sweep
```

Los endpoints de mercado son públicos: no hace falta API key para descargar ni
para backtestear. Si Binance está bloqueado en tu región, descarga los CSV
oficiales de <https://data.binance.vision> y colócalos en `data/`.

## Comandos

| Comando | Qué hace |
|---|---|
| `synth` | Genera un dataset sintético para probar offline |
| `download` | Descarga klines + funding + OI + long/short + fear&greed |
| `features` | Construye features y etiquetas de triple barrera |
| `train` | Entrena walk-forward purgado, guarda modelo y predicciones OOS |
| `backtest` | Backtestea sobre las predicciones OOS con costes reales |
| `sweep` | Barre el umbral de EV mínimo (el diagnóstico más informativo) |
| `paper` | Trading en papel contra datos reales en vivo, sin riesgo |
| `live` | Trading real. Por defecto testnet + dry-run |
| `config` | Muestra la configuración efectiva |
| `all` | Pipeline completo de principio a fin |

---

## Cómo funciona

```
Binance REST ─┬─ klines OHLCV + taker_buy_base
              ├─ funding rate          ┐
              ├─ open interest         ├─ contexto, remuestreado y retrasado 1 barra
              ├─ top long/short ratio  │
              └─ fear & greed          ┘
                      │
                      ▼
        ┌──── features (75 columnas) ────┐
        │  ta_*   análisis técnico       │
        │  ms_*   microestructura        │
        │  ctx_*  contexto macro         │
        └────────────────────────────────┘
                      │
                      ▼
        triple barrera (TP / SL / tiempo)  ──►  y ∈ {-1, 0, +1}
                      │
                      ▼
        walk-forward purgado con embargo   ──►  probabilidades OOS
                      │
                      ▼
        EV neto de costes  ──►  Kelly fraccional  ──►  gestión de riesgo
                      │
                      ▼
              backtest  /  papel  /  live
```

### 1. Fuentes de datos

**Análisis técnico** (bloque `ta_`, ~40 columnas). Momento multi-escala,
distancias a medias móviles, osciladores, volatilidad, ADX, ratio de varianza
tipo Hurst, estructura de vela y estacionalidad intradía. Todo está normalizado
—z-scores, ratios, distancias en unidades de ATR— porque los niveles absolutos
no son estacionarios y destruyen la generalización.

**Microestructura** (bloque `ms_`, ~25 columnas). Es el bloque con más
contenido predictivo en horizontes de minutos. Las klines de Binance incluyen
`taker_buy_base`, el volumen ejecutado por compradores agresores, y con eso se
reconstruye:

- **OFI** (order flow imbalance): `(compras - ventas) / total`, en varias ventanas.
- **CVD**: delta de volumen acumulado y su pendiente.
- **Divergencia flujo/precio**: el flujo empuja pero el precio no cede.
- **Kyle lambda** y **Amihud**: cuánto mueve el precio una unidad de volumen.
- **VPIN** simplificado: fracción de volumen desequilibrado, proxy de toxicidad.
- **Desviación del VWAP** y **eficiencia del movimiento**.

**Contexto macro** (bloque `ctx_`, ~12 columnas). Funding rate (posicionamiento
apalancado), open interest y su concordancia con el precio, ratio long/short de
cuentas top, flujo taker agregado y Fear & Greed. Llegan cada 5 min u 8 h, así
que se reindexan con forward-fill y **se retrasan una barra** para que en el
instante de decisión solo se conozca lo ya publicado.

`book_features()` en `features/microstructure.py` calcula desequilibrio del
libro, microprecio y pendiente de liquidez. Solo es utilizable en vivo: Binance
no publica histórico del libro, así que no puede entrar en el backtest.

### 2. Etiquetado: triple barrera

No se etiqueta con «el retorno de la próxima barra». Un scalper no mantiene la
posición un tiempo fijo: sale por take-profit, por stop o por tiempo. La
etiqueta debe reflejar esa regla de salida.

Para cada barra se colocan tres barreras y gana la primera que se toque:

- **Superior**: `tp_sigma · σ · √horizonte` → etiqueta `+1`
- **Inferior**: `sl_sigma · σ · √horizonte` → etiqueta `-1`
- **Temporal**: `horizonte` barras → etiqueta `0`

El escalado por `√horizonte` es imprescindible y es un error clásico omitirlo.
Con la sigma de una sola barra las barreras quedan en ~7 bps, el ruido las toca
en 2-3 barras y ninguna estrategia sobrevive a 12 bps de costes. Con el
escalado quedan en ~25-30 bps y el trade tiene espacio para respirar. Además
`min_barrier_bps` impone un suelo que **debe superar con holgura el coste de
ida y vuelta**.

Si TP y SL se tocan en la misma vela, no sabemos el orden intrabar: se asume el
stop. Es la hipótesis conservadora y evita inflar el backtest.

Los pesos de muestra corrigen dos problemas: las etiquetas solapadas comparten
barras futuras y violan la independencia (se penalizan por concurrencia), y los
eventos de mayor magnitud importan más (se ponderan por `|retorno|`).

### 3. Validación: walk-forward purgado con embargo

Un K-Fold normal es catastrófico en series financieras: mezcla futuro y pasado
y produce Sharpes de fantasía. Aquí:

- **Walk-forward**: el test siempre está después del train.
- **Purga**: se eliminan del train las muestras cuya barrera invade el test.
- **Embargo**: se descartan barras extra tras el test para cortar la
  autocorrelación serial residual.

El comando `train` guarda las predicciones **out-of-sample** de cada fold, y el
backtest solo puede consumir esas. Backtestear con probabilidades in-sample da
curvas de equity preciosas y pérdidas reales. `backtest --in-sample` existe
únicamente para depurar y avisa por pantalla.

### 4. Modelo

LightGBM multiclase sobre las 3 clases de la triple barrera, con fallback
automático a `HistGradientBoostingClassifier` de scikit-learn si LightGBM no
está instalado. Gradient boosting sobre tabular es la elección correcta aquí:
las redes profundas necesitan mucho más dato del que hay en unos meses de velas
de 1 minuto, y en tabular ni siquiera ganan.

La preselección de features por importancia se hace **una sola vez sobre la
ventana de entrenamiento inicial**, nunca sobre el dataset completo: hacerlo
sobre todo el histórico es una fuga sutil que infla los resultados.

### 5. De probabilidad a operación

El paso que casi todos los tutoriales se saltan. Predecir la dirección no basta:
hay que comprobar que el valor esperado **neto de costes** es positivo.

```
EV_long  = p_up · TP − p_dn · SL − costes
EV_short = p_dn · TP − p_up · SL − costes
```

Solo se abre si `EV ≥ min_edge_bps` y la probabilidad direccional supera
`min_prob`. El tamaño sale de **Kelly fraccional** (por defecto un cuarto de
Kelly, limitado a `max_position_pct`). Kelly completo maximiza el crecimiento
logarítmico asintótico y en la práctica te arruina antes de llegar al
asintótico: cualquier error en la estimación de probabilidad se amplifica.

### 6. Gestión de riesgo

Cortacircuitos idénticos en backtest y en vivo: límite de pérdida diaria,
máximo de operaciones por día, racha máxima de pérdidas y verificación de
equity. Todos se reinician al cambiar de día UTC.

> Detalle que costó un bug real durante el desarrollo: la racha de pérdidas
> **tiene** que reiniciarse al cambiar de día. Si no, al alcanzar el límite el
> bot no puede abrir; como no abre, nunca cierra en ganancia; como nunca gana,
> el contador jamás baja y el bot queda bloqueado para siempre. En el primer
> backtest dejó de operar el día 3 de 16 y el informe parecía normal. Hay un
> test de regresión para eso (`test_consecutive_loss_breaker_resets_daily`).

### 7. Backtest

Motor barra a barra con supuestos explícitos y conservadores:

- La señal se calcula al **cierre** de la barra `t` y se ejecuta a la
  **apertura** de `t+1`. Nunca se opera al precio que generó la señal.
- El deslizamiento se aplica siempre en contra.
- Si TP y SL caen en la misma vela, se asume el stop.
- Comisiones cobradas en ambos lados sobre el nocional.
- Una sola posición abierta a la vez.

El informe incluye Sharpe, Sortino, max drawdown, profit factor, bps medios por
operación, comisiones totales, **t-stat del edge** y una **tabla de
sensibilidad a costes** que muestra qué queda si tus costes reales son 1, 2 o 5
bps peores de lo que supusiste. Esa tabla es la prueba más informativa de un
backtest de scalping: si el PnL se evapora con 2 bps extra, no tienes una
estrategia, tienes una estimación de costes optimista.

---

## Interpretar los resultados

Ejecuta `sweep` y mira la forma de la curva, no el mejor número:

```
 min_edge_bps  n_trades  avg_trade_bps  win_rate_pct   sharpe   t_stat
         0.00       942          -4.53         51.06   -22.79    -4.29
         2.00       916          -2.88         52.95   -14.85    -2.46
        10.00       472          -2.69         51.69    -6.58    -1.29
        20.00       236          +0.14         54.66    -1.52     0.04
        25.00       186          -2.46         52.69    -4.37    -0.54
        30.00       150          -1.84         52.00    -2.51    -0.33
```

Un edge real está **concentrado en las señales más fuertes** y mejora de forma
monótona al ser más selectivo. En la tabla de arriba mejora hasta 20 bps y
luego empeora: ese pico no es un óptimo, es ruido. Y aunque fuera monótono, el
valor concreto está sesgado al alza por haber probado muchos umbrales sobre los
mismos datos.

Qué mirar, por orden de importancia:

1. **`avg_trade_bps` neto** por encima de cero. Si no, nada más importa.
2. **`t_stat`** con `|t| > 2`. Con menos, tu edge es indistinguible del ruido.
3. **Monotonía del sweep**. Sin ella, es sobreajuste.
4. **Sensibilidad a costes**. Debe sobrevivir a +2 bps.
5. Sharpe y drawdown, al final. Un Sharpe alto con 40 operaciones no significa
   nada.

Desconfía de cualquier backtest de scalping con Sharpe superior a 3. Casi
siempre es fuga de información, costes irreales o sobreajuste al periodo.

---

## Trading en vivo

La progresión no es negociable. Cada etapa filtra una clase distinta de error.

```bash
# 1. Papel contra datos reales en vivo. Semanas, no días.
python -m scalpbot paper

# 2. Testnet en dry-run: valida la lógica de órdenes sin enviarlas.
python -m scalpbot live

# 3. Testnet enviando órdenes de verdad.
export BINANCE_API_KEY=...      # claves de testnet.binancefuture.com
export BINANCE_API_SECRET=...
python -m scalpbot live --real

# 4. Mainnet con dinero que puedas perder entero. Pide confirmación escrita.
python -m scalpbot live --real --mainnet
```

En modo futuros, al abrir posición el bot coloca TP y SL como órdenes
`reduce-only` en el propio exchange. Es deliberado: si el proceso muere, la
posición sigue protegida. Un bot de scalping sin stops del lado del exchange es
una posición sin supervisión esperando un movimiento adverso.

Otras salvaguardas: nunca se opera sobre la vela en curso (sus valores cambian
hasta el último segundo), `SIGINT`/`SIGTERM` cierran la posición de forma
ordenada, y una excepción en un ciclo no mata el bucle.

**Sobre las claves API**: crea claves específicas para el bot, sin permiso de
retirada, y restringidas por IP. Nunca las metas en el repositorio ni en
`config.yaml` — se leen del entorno.

---

## Configuración

Todo vive en `config.yaml`. Los parámetros que más mueven el resultado:

| Parámetro | Por defecto | Comentario |
|---|---|---|
| `costs.taker_fee_bps` | 5.0 | Ajústalo a **tu** nivel VIP real |
| `costs.slippage_bps` | 1.0 | Optimista para 1m; mídelo con tus fills |
| `labels.horizon` | 20 | Barras hasta la barrera temporal |
| `labels.min_barrier_bps` | 20.0 | Debe superar con holgura los costes |
| `strategy.min_edge_bps` | 2.0 | Súbelo bastante; usa `sweep` |
| `strategy.kelly_fraction` | 0.25 | Nunca 1.0 |
| `risk.max_daily_loss_pct` | 2.0 | Cortacircuito diario |
| `risk.leverage` | 1.0 | Súbelo solo tras meses de papel |

Cambiar de `spot` a `futures` reduce las comisiones a la mitad y es
probablemente la decisión de configuración más impactante.

---

## Tests

```bash
python -m pytest tests/ -v
```

17 tests. Los importantes no comprueban que el código corra, sino que **no hay
fuga de información**:

- `test_features_are_causal` — modifica el futuro del dataset y verifica que
  ninguna feature del pasado cambia. Una fuga aquí invalida absolutamente todo
  lo demás.
- `test_no_signal_data_yields_no_skill` — entrena sobre ruido puro y exige que
  la precisión direccional se quede entre 0.40 y 0.60. Es el control negativo.
- `test_purged_walk_forward_is_temporal` — ningún índice de train cae dentro o
  después del test, y la purga elimina las etiquetas solapadas.
- `test_backtest_executes_at_next_open` — la señal de `t` se llena a la
  apertura de `t+1`, no en `t`.
- `test_backtest_charges_costs_on_flat_market` — en mercado plano, cada
  operación pierde exactamente los costes.
- `test_consecutive_loss_breaker_resets_daily` — regresión del bloqueo
  permanente descrito arriba.

Ambos guardas están verificados: al inyectar una feature con `shift(-5)`, el
test de causalidad falla y el control negativo salta a 0.79 de precisión sobre
ruido puro. Si tocas el pipeline y estos tests siguen en verde, no has
introducido lookahead.

---

## Cómo mejorarlo, por orden de rentabilidad esperada

1. **Ejecución maker.** Pasar de 12 a 6 bps de ida y vuelta duplica el margen
   disponible. Requiere lógica de órdenes límite, gestión de no-ejecución y
   reposicionamiento. Es más trabajo de ingeniería que de modelado, y es donde
   más retorno hay.
2. **Datos de tick y libro.** Los websockets de Binance dan trades individuales
   y actualizaciones del libro. Ahí está la información que las velas de 1m
   promedian y destruyen. `book_features()` ya está implementado; falta
   almacenarlo para poder entrenar con ello.
3. **Meta-labeling.** Un segundo modelo que decide *si operar* la señal del
   primero. En la práctica sube bastante la precisión sobre las operaciones que
   sí se ejecutan.
4. **Calibración de probabilidades.** `CalibratedClassifierCV` con
   `method="isotonic"`. Los `logloss` por fold del pipeline actual están por
   encima de `ln(3)`: las probabilidades no están bien calibradas, y el cálculo
   de EV depende directamente de ellas.
5. **Modelos por régimen.** Entrenar por separado tendencia y rango, usando ADX
   y el ratio de varianza como clasificador de régimen.
6. **Reentrenamiento periódico.** Los mercados cambian. Un modelo de hace tres
   meses está midiendo otro mercado.

Lo que **no** recomiendo perseguir primero: arquitecturas más complejas
(LSTM, transformers, RL). Con este volumen de datos rinden peor que gradient
boosting, y el cuello de botella no está en el modelo — está en los costes y en
la calidad de las features.

---

## Advertencia

Esto es software educativo y de investigación. No es asesoramiento financiero.
El trading algorítmico de criptomonedas con apalancamiento puede hacerte perder
todo tu capital, y la mayoría de los bots de scalping minoristas pierden dinero
de forma consistente por exactamente la razón de la primera sección: los costes
superan al edge.

El uso realista de este repositorio es **medir** si tienes edge, con
metodología que no se engaña a sí misma. Un backtest negativo bien hecho vale
más que uno positivo mal hecho: te ha ahorrado dinero real.

Opera solo con dinero que puedas perder entero.

## Licencia

MIT.
