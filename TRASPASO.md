# Traspaso

Para retomar esto desde cero: en otra cuenta, en otro ordenador, o en otra
sesión que no recuerde nada de las anteriores.

---

## 1. Llevar el repositorio a la cuenta nueva

El repositorio no depende de ninguna cuenta: los 44 commits van a nombre de
`Claude <noreply@anthropic.com>`, no hay secretos, ni tokens, ni rutas
absolutas de nadie. Sólo hay que empujarlo al sitio nuevo.

Con el ZIP, sin necesitar acceso al repositorio viejo:

```bash
unzip nanocarbon.zip -d nanocarbon && cd nanocarbon
git init && git add -A
git commit -m "nanocarbon: importar el espacio de trabajo"
git branch -M main
git remote add origin https://github.com/TU-CUENTA-NUEVA/nanocarbon.git
git push -u origin main
```

Eso pierde el historial. Si lo quieres conservar —y merece la pena, porque
cada mensaje de commit explica **por qué** se hizo el cambio y con qué medida
se comprobó— clona el viejo mientras aún tengas acceso y cámbiale el remoto:

```bash
git clone https://github.com/jimenelouis808/Cc.git nanocarbon
cd nanocarbon
git remote set-url origin https://github.com/TU-CUENTA-NUEVA/nanocarbon.git
git push -u origin main
git push origin --all        # si quieres también las ramas viejas
```

La rama de trabajo es **`main`**. Contiene todo. Las ramas `claude/...` que
hay en el remoto viejo son de tareas ajenas a este proyecto y se pueden
ignorar o borrar.

La CI (`.github/workflows/ci.yml`) se dispara en `branches: ["**"]`, así que
funciona en la cuenta nueva sin tocar nada.

---

## 2. Poner en marcha

```bash
uv sync --all-packages --extra dev
uv run ramancarbon-gui
```

Si no hay `uv`: `pip install -e "packages/ramancarbon[dev]"`. Para las
interfaces hace falta Tkinter, que en Linux se instala aparte
(`sudo apt install python3-tk`). Los detalles están en `INSTALACION.md`.

Comprobar que todo pasa antes de tocar nada:

```bash
cd packages/ramancarbon
OMP_NUM_THREADS=1 uv run python -m pytest ramancarbon/tests -q -n 4 --dist loadscope
```

Deben salir **1337 pruebas** en verde (una se salta si no hay Tkinter).
`carbonforge` da 647 y `nanocarbon_lab` 1148 con `-m "not slow"`.

---

## 3. Qué leer antes de cambiar nada

Los `CLAUDE.md` **son el proyecto**, no documentación de cortesía. Cada
párrafo registra una decisión que se tomó porque su alternativa produjo un
resultado equivocado, con el número que lo demostró. Por ejemplo: por qué la
línea base es asLS y no arPLS, por qué el umbral de detección de DRX está en
18 y no en 12, por qué fijar un parámetro no es gratis, por qué `np.interp`
recortando en silencio informaba tres sobrepotenciales idénticos.

| Archivo | Líneas | Qué protege |
|---|---|---|
| `CLAUDE.md` | 57 | La regla del espacio de trabajo y cómo correr las pruebas |
| `packages/ramancarbon/CLAUDE.md` | ~1600 | Raman, DRX, XPS, electroquímica, figuras, GUI |
| `packages/nanocarbon_lab/CLAUDE.md` | ~1170 | Geometría, periodicidad, relajación |
| `packages/carbonforge/CLAUDE.md` | 119 | Preparación de cálculos |

**Si una sesión nueva de Claude Code abre este repositorio, los lee sola.**
Están donde están precisamente para eso.

---

## 4. Estado actual

Funciona y está probado contra datos sintéticos:

- **Raman** (carbono y TMD): preprocesado, deconvolución, índices, diámetros,
  multiláser, identificación de fases con base de datos ampliable por el
  usuario, polímeros como grupo opcional.
- **DRX**: identificación contra estructuras cristalinas (no tablas de picos),
  Rietveld por etapas con control por parámetro, microestructura, Le Bail y
  Pawley. Detecta la escala de cuentas cuando el archivo viene reescalado.
- **XPS**: survey, ajuste por regiones con control por componente, composición.
  Lee `.spe` de PHI y VAMAS.
- **Electroquímica**: CV, series de velocidad, Dunn, carga-descarga con tres
  convenios de capacitancia, impedancia con 20 circuitos editables, DRT,
  C(ω), Ragone, HER/OER. Lee EC-Lab `.mpr` y `.mps`.
- **Exportación**: cualquier figura de cualquier pestaña escribe sus números
  con «Guardar datos…».

### Lo que hay que verificar con datos reales

Nada está contrastado contra un patrón certificado. En concreto:

1. **El lector de `.mpr`.** La disposición binaria es ingeniería inversa de la
   comunidad. La aritmética cierra contra la longitud del archivo, lo que
   descarta que una columna tenga la ANCHURA equivocada, pero no que tenga el
   NOMBRE equivocado. **Contrasta una medida contra la exportación `.mpt` de
   EC-Lab la primera vez.** Está dicho en `echem/biologic.py` y en el aviso
   que acompaña a cada lectura.
2. **La identificación de fases Raman y DRX** sobre muestras conocidas.
3. **Las anchuras y posiciones XPS** contra un ajuste hecho en CasaXPS.

### Lo que quedó pendiente o a medias

- **La reconciliación de `carbonforge` y `nanocarbon_lab`.** Ambos construyen
  estructuras, las dopan y las validan, desde un ancestro común y sin historia
  git compartida. Han divergido. Es trabajo conocido y aplazado a propósito:
  no lo hagas por accidente mientras arreglas otra cosa.
- **Un pico en 580 cm⁻¹** de la muestra de carbono+FeSe sigue sin explicar.
  El programa ahora lista las tres líneas catalogadas más cercanas con su
  distancia; hace falta decidir cuál es y, si procede, añadir la fase.
- **El demo de Rietveld** funciona; sobre patrones reales muy ruidosos aún no
  se ha probado a fondo.
- **`ruff format` no se pasa nunca** en este repositorio: 301 de 395 archivos
  cambiarían. Se usa `ruff check`.

---

## 5. Cómo se trabaja aquí

Esto es lo que ha hecho que el proyecto no se rompa:

1. **Medir antes de cambiar.** Cuando algo falla, se reproduce primero con un
   caso sintético que lo demuestre, y sólo entonces se toca el código. Varios
   de los arreglos de las últimas rondas empezaron con un diagnóstico que
   resultó ser falso y se descartó por una medida.
2. **Un umbral no se baja para que un caso pase.** Si un caso real no se
   detecta, el problema suele estar en una MEDIDA mal hecha (una anchura, un
   ruido, una escala), no en el umbral. Bajarlo no encuentra más fases:
   encuentra más rizos y los llama fases.
3. **Cada arreglo lleva su prueba**, y la prueba dice en su docstring qué
   resultado equivocado producía su ausencia.
4. **Degradar, no reventar.** Si falta un dato, se devuelve `None` con el
   motivo y el informe lo dice.
5. **Los avisos son parte del resultado.** Si un número puede estar sesgado,
   el objeto que lo lleva tiene que decirlo.

---

## 6. Para seguir con Claude Code

Abre el repositorio y describe lo que quieres. Los `CLAUDE.md` se cargan
solos. Lo que más ayuda:

- **Capturas de pantalla.** Varios de los bugs de estas rondas salieron de
  una captura: el eje que llegaba a 2·10⁵ mA/cm², las intensidades de 1 a 3.5
  «cuentas», los tres sobrepotenciales idénticos.
- **Un archivo real** cuando algo falla al leerlo. El mensaje de error nombra
  ahora el formato que encontró, y con eso más el archivo se arregla rápido.
- **Decir qué esperabas ver.** «No identifica nada» y «identifica cementita
  cuando mi muestra es FeSe» llevan a sitios distintos.
