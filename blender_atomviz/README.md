# AtomViz Studio

Add-on de Blender para convertir estructuras importadas con **Atomic Blender**
(el importador `.xyz` / `.pdb` que viene con Blender) en imágenes de portada
de revista: texturas y sombreados por elemento, fondos procedurales, y efectos
de **electricidad, luz volumétrica y láseres**.

El add-on **nunca toca las coordenadas atómicas**. Solo asigna materiales,
escala las esferas instanciadas y añade luces, mundo, arcos, haces y nodos de
composición alrededor. La estructura importada sigue siendo científicamente
correcta.

```
blender_atomviz/
└── atomviz_studio/
    ├── core/        # datos de elementos, paletas, geometría, detección (mitad sin bpy)
    ├── materials/   # 12 estilos de sombreado + asignación por elemento
    ├── effects/     # fondos, rigs de luz, niebla, arcos eléctricos, láseres, post
    ├── scene/       # encuadre de cámara y ajustes de render
    ├── looks/       # recetas completas de portada en un clic
    ├── ui/          # propiedades, operadores y panel lateral (el "applet")
    ├── cli/         # render sin interfaz: blender -b -P ...
    └── tests/       # pytest (no necesita Blender)
```

---

## 1. Instalación

**Blender 3.3 – 4.1** (add-on clásico)

1. Comprime la carpeta `atomviz_studio/` en un `.zip`
   (`python tools/make_addon_zip.py` lo hace por ti).
2. `Edit ▸ Preferences ▸ Add-ons ▸ Install…` y selecciona el zip.
3. Activa **“AtomViz Studio (Atomic Blender covers & VFX)”**.

**Blender 4.2+** (extensión): el paquete incluye `blender_manifest.toml`, así
que también se puede instalar con `Install from Disk…` desde la pestaña de
extensiones.

Además hay que tener activo el importador de estructuras:
`Preferences ▸ Add-ons ▸ “Import-Export: Atomic Blender PDB/XYZ”`
(el botón *Import XYZ* del panel lo activa automáticamente si puede).

El panel aparece en la vista 3D, barra lateral (tecla **N**), pestaña
**AtomViz**.

---

## 2. Flujo de trabajo típico

1. **Importa** el `.xyz` (Atomic Blender, o el botón del panel).
2. Pulsa la lupa: AtomViz detecta la estructura y lista los elementos.
3. Elige un **Cover look** y pulsa **Apply cover look**. Eso deja materiales,
   fondo, luces, efectos, cámara, formato de render y post-proceso listos.
4. Ajusta a mano lo que quieras en los subpaneles (todo son los mismos
   operadores que usa el look).
5. **Render cover**.

Con `Seed` fijas el aleatorio: el mismo número reconstruye exactamente los
mismos arcos y la misma dispersión de haces.

---

## 3. Qué detecta (y por qué funciona con estructuras raras)

Atomic Blender crea, por cada elemento, una malla con un vértice por átomo que
instancia una esfera hija, más un objeto de enlaces:

```
Empty "molecula.xyz"
 ├── "Carbon_mesh"  (instance_type = VERTS)  → "Carbon_ball"  ← lleva el material
 ├── "Nitrogen_mesh"                          → "Nitrogen_ball"
 └── "Sticks"
```

La detección es heurística y tolerante: busca nombres de elemento (nombre
completo o símbolo) en el objeto, la malla y los materiales, sigue los
instanciadores hasta el hijo que lleva el material, y descarta la geometría de
enlaces. Nombres como `Carbon_ball.001`, `Iron_F2+`, `N.003` o `Au_sphere` se
resuelven bien; `Carbon` nunca se confunde con calcio ni `Nitrogen` con níquel.
Si no reconoce nada, agrupa la geometría como una estructura genérica para que
los operadores sigan siendo útiles.

Ámbito configurable: escena completa, selección o colección activa.

---

## 4. Sombreado y paleta

**12 estilos** (`materials/styles.py`):

| Estilo | Uso típico |
|---|---|
| `glossy_ceramic` | Cerámica esmaltada. El seguro por defecto. |
| `polished_metal` | Metal pulido, reflejos duros. |
| `matte_clay` | Difuso suave, ideal para impresión. |
| `glass_crystal` | Vidrio transmisivo coloreado. |
| `frosted_glass` | Vidrio esmerilado con brillo interior. |
| `subsurface_jelly` | Translúcido, la luz atraviesa los átomos. |
| `emissive_neon` | Núcleo emisivo dentro de una cáscara brillante. |
| `iridescent` | Nacarado; el tono cambia con el ángulo. |
| `toon_cel` | Cel-shading plano (EEVEE) para divulgación. |
| `xray_fresnel` | Cuerpo transparente con silueta luminosa. |
| `hologram` | Proyección con líneas de barrido (EEVEE). |
| `graphite` | Carbono oscuro con grano procedural. |

**8 paletas** (`core/palettes.py`): `cpk` (estándar), `pastel_lab`,
`midnight_neon`, `ice_fire`, `gold_lab`, `mono_accent` (marco neutro + un solo
color de acento), `spectral` (color según Z) y `blueprint`.

Extras útiles:

- **Glowing dopants**: emisión extra automática para todo lo que no sea C o H.
  Es el truco clásico de portada: marco sobrio, heteroátomos que guían el ojo.
- **Atom radii**: reescala las esferas por radio de van der Waals, covalente o
  uniforme, sin tocar las posiciones.

---

## 5. Fondo, luz y atmósfera

- **Fondos**: `solid`, `gradient`, `studio_white`, `nebula`, `starfield`,
  `plasma`, `transparent` (film alpha para componer fuera). Todos
  procedurales, controlados por tres colores.
- **Rigs de luz**: `three_point`, `studio_soft`, `neon_rim`, `dramatic_top`,
  `godrays`. Las posiciones y la potencia escalan con el radio de la
  estructura, así que el mismo rig sirve para una molécula de 20 átomos o una
  espuma de 5000.
- **Haze**: medio participativo en una caja alrededor de la estructura. Es lo
  que hace visibles los haces y los rayos de luz. Se usa una caja y no un
  volumen de mundo porque renderiza mucho más rápido.

---

## 6. Efectos

### Electricidad (`effects/electricity.py`)

Arcos construidos por **desplazamiento de punto medio** (la construcción
fractal clásica del rayo), con bifurcaciones, núcleo blanco-caliente y halo
suave. Tres modos:

- `ATOMS` — descargas entre átomos elegidos al azar.
- `CAGE` — jaula de arcos sobre la esfera envolvente.
- `DISCHARGE` — arcos que salen hacia fuera desde los átomos de la superficie.

Opción **Flicker**: keyframes de visibilidad (interpolación constante) para
que la descarga chisporrotee en una animación.

### Láseres (`effects/lasers.py`)

Cada haz son tres cosas: núcleo emisivo fino, funda que se apaga en ángulos
rasantes, y punto de impacto con esfera emisiva **más una luz real**, para que
la estructura quede iluminada por el haz que la golpea. Los emisores se
reparten con una espiral de Fibonacci para que no se amontonen.

### Post-proceso (`effects/postfx.py`)

Cadena de composición: `Glare` (fog glow / streaks / ghosts / star), viñeta,
contraste y saturación. Se hace en el compositor y no con el bloom de EEVEE
porque Blender 4.2 eliminó ese bloom: así el resultado es idéntico en EEVEE,
EEVEE Next y Cycles.

---

## 7. Looks de portada (un clic)

| Look | Idea |
|---|---|
| `clean_journal` | Estudio blanco, cerámica, luz suave. La portada segura. |
| `neon_lab` | Marco grafito, dopantes encendidos, arcos eléctricos. |
| `plasma_storm` | Heteroátomos calientes dentro de una jaula de plasma. |
| `laser_lab` | Cristal atravesado por láseres en sala con humo. |
| `crystal_ice` | Vidrio esmerilado sobre degradado frío. |
| `gold_catalysis` | Metales cálidos, catálisis y plasmónica. |
| `xray_schematic` | Átomos transparentes con silueta luminosa; se lee como esquema. |
| `toon_outreach` | Cel-shading plano para notas de prensa y docencia. |
| `hologram` | Estructura emisiva con líneas de barrido en un campo de estrellas. |

---

## 8. Formatos de render

`preview_fast` (EEVEE, para iterar), `cover_a4_300` (2480×3508),
`cover_letter_300` (2550×3300), `cover_square` (3000×3000),
`toc_graphic` (2400×1260), `poster_uhd` (3840×2160),
`print_a4_600` (4961×7016) y `alpha_overlay` (film transparente).

Cada preset fija resolución, motor, muestras y gestión de color
(**AgX** en 4.x, **Filmic** en 3.x, con degradación automática).

---

## 9. Render sin interfaz (cola / servidor)

```bash
blender -b -P atomviz_studio/cli/render_cover.py -- \
    --xyz out/cnt/cnt.xyz \
    --look neon_lab \
    --format cover_a4_300 \
    --seed 7 \
    --out covers/cnt.png
```

Otras opciones: `--samples`, `--percentage`, `--focal`, `--azimuth`,
`--elevation`, `--save-blend`, `--no-render`, `--list` (lista looks y
formatos). Devuelve código de salida distinto de cero si el fichero no existe
o si Atomic Blender no está disponible, así que se puede encadenar en un
script de barrido.

Ejemplo de barrido sobre estructuras generadas con `nanocarbon_lab`:

```bash
for f in out/sweep/*.xyz; do
  blender -b -P atomviz_studio/cli/render_cover.py -- \
      --xyz "$f" --look plasma_storm --format toc_graphic \
      --out "covers/$(basename "${f%.xyz}").png"
done
```

---

## 10. Consejos para portadas

- Deja limpio el **15 % superior**: ahí va la cabecera de la revista.
- Formato vertical: el encuadre ya compensa la relación de aspecto (a 2480×3508
  el campo horizontal es el limitante, así que la cámara retrocede más).
- Focales largas (70–100 mm) aplanan la perspectiva y se ven editoriales; las
  cortas (28–40 mm) exageran la profundidad.
- Itera con `preview_fast` y cambia a un preset Cycles solo al final.
- Si la revista pide CMYK o un perfil concreto, renderiza en PNG 16 bits y
  convierte en el editor de imagen; Blender no exporta CMYK.

---

## 11. Desarrollo y pruebas

La mitad `core` (elementos, colores, paletas, geometría, presets) no importa
`bpy` y se prueba fuera de Blender:

```bash
cd blender_atomviz
pytest atomviz_studio/tests -q        # 85 pruebas
ruff check atomviz_studio
```

`tests/test_import_smoke.py` importa además **todos** los módulos que sí usan
`bpy` contra un stub mínimo, de forma que los errores de importación,
declaración de propiedades o referencias cruzadas se detectan sin abrir
Blender.

Compatibilidad: Blender 3.3 LTS – 4.5. Todo lo que cambió entre versiones
(nombres de sockets del Principled BSDF, EEVEE vs EEVEE Next, AgX vs Filmic,
modos de mezcla de materiales) pasa por `core/compat.py`.
