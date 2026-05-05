# Basketball CV Prototype

Primer prototipo para convertir video de baloncesto en detecciones, tracking, pista 2D y candidatos de bloqueo directo.

Pipeline oficial recomendado:

```powershell
python tools/run_game_pipeline.py --video "subra.mp4" --output-root runs/pipeline --run-name subra_game --with-ocr --device 0 --debug-possession
```

Ese es el unico flujo oficial soportado de extremo a extremo: `tools/run_game_pipeline.py` encadena analisis, OCR opcional y render final. Los comandos sueltos de abajo quedan como utilidades para calibrar, depurar o relanzar una etapa concreta sin contradecir ese flujo principal.

## 1. Calibrar la pista

La calibracion genera la homografia de pixeles a metros de pista.

```powershell
python tools/calibrate_court.py --video "2026-04-27 13-40-33.mp4" --output config/court_calibration.json --frame 120
```

Uso:

- En la ventana del video, haz click en un punto conocido de la pista.
- En la ventana de pista 2D, haz click en el punto equivalente.
- Repite al menos 4 veces; 6-10 puntos suele ir mejor.
- `Enter` guarda, `U` deshace, `Q` sale.

Buenos puntos: esquinas de campo, linea de medio campo con banda, esquinas de la zona, tiro libre, aro si se ve claro.

## 2. Analizar video

Comando recomendado para esta toma, usando la RTX local:

```powershell
python tools/analyze_video.py --video "2026-04-27 13-40-33.mp4" --calibration config/court_calibration.json --output-dir runs/stable --model yolo11m.pt --tracker trackers/bytetrack_basketball.yaml --imgsz 768 --conf 0.15 --device 0 --court-margin 0.9 --court-near-margin 1.2 --court-far-margin 0.5 --write-video
```

El analizador ahora tambien:

- interpola huecos cortos de balon;
- genera una trayectoria densa de balon para render, con estimaciones visuales entre detecciones;
- estima poseedor del balon por frame;
- exporta `ball_tracks.json`;
- genera eventos `pass` en `events.json`;
- renderiza toasts de pase cuando se escribe video.

Alternativa moderna con YOLO26 y la cabeza one-to-many/NMS:

```powershell
python tools/analyze_video.py --video "2026-04-27 13-40-33.mp4" --calibration config/court_calibration.json --output-dir runs/stable_yolo26 --model yolo26s.pt --tracker trackers/bytetrack_basketball.yaml --imgsz 768 --conf 0.15 --device 0 --use-nms-head --write-video
```

Comando rapido con modelo pequeno:

```powershell
python tools/analyze_video.py --video "2026-04-27 13-40-33.mp4" --calibration config/court_calibration.json --output-dir runs/sample --write-video
```

Salidas:

- `runs/sample/annotated.mp4`: video con detecciones y minimapa.
- `runs/sample/tracks.json`: detecciones por frame.
- `runs/sample/events.json`: candidatos de pick and roll.
- `runs/sample/summary.json`: resumen de la ejecucion.
- `runs/sample/track_summary.json`: resumen por track, equipo estable, frames activos y rasgos de camiseta.
- `runs/sample/player_summary.json`: IDs postprocesados (`player_id`) que unen fragmentos de `track_id` cuando hay oclusiones.

## 3. Revisar la calibracion

Despues de calibrar, genera una imagen con las lineas de la pista proyectadas sobre el frame:

```powershell
python tools/review_calibration.py --video "2026-04-27 13-40-33.mp4" --calibration config/court_calibration.json --output previews/calibration_review.jpg
```

Si las lineas amarillas no caen encima de las lineas reales de la pista, repite la calibracion con mas puntos o puntos mas separados.

## 4. Entrenar detector de balon

YOLO base casi no ve el balon en este video. Para crear un detector propio, etiqueta frames con:

```powershell
python tools/label_ball.py --video "2026-04-27 13-40-33.mp4" "2026-04-27 13-31-09.mp4" --output datasets/ball --step 8 --box-size 34
```

Tambien puedes etiquetar todos los MP4 de la carpeta:

```powershell
python tools/label_ball.py --video "2026-04-27 13-40-33.mp4" --video-glob "*.mp4" --output datasets/ball --step 8 --box-size 34
```

Controles:

- Click: centro del balon.
- `Space`: guarda frame con balon.
- `E`: guarda frame negativo sin balon visible.
- `N`: salta sin guardar.
- `+` / `-`: cambia tamano de caja.
- `U`: borra la marca actual.
- `Q`: termina.

Entrena local o en Colab:

```powershell
python tools/train_ball_detector.py --data datasets/ball/data.yaml --base-model yolo11s.pt --device 0 --epochs 80 --imgsz 960
```

Despues analiza usando el modelo entrenado:

```powershell
python tools/analyze_video.py --video "2026-04-27 13-40-33.mp4" --calibration config/court_calibration.json --ball-model runs/ball_train/yolo11s_ball/weights/best.pt --output-dir runs/sample_ball --write-video --device 0
```

Los pases solo seran buenos si el balon se detecta de forma consistente. Sin un detector propio de balon, la trayectoria densa sirve para visualizar, pero muchas posiciones son estimadas. En `ball_tracks.json`, `dense_ball_estimate=true` distingue estimaciones de detecciones reales.

## 5. OCR de dorsales

El flujo oficial para OCR y render final sigue siendo `python tools/run_game_pipeline.py --with-ocr ...`. Los comandos de esta seccion sirven para inspeccion, depuracion o reruns parciales cuando hace falta mirar una etapa por separado.

Primero genera tracks con el analizador. Despues extrae dorsales por jugador:

```powershell
python tools/extract_jersey_numbers.py --video "2026-04-27 13-40-33.mp4" --tracks runs/stable/tracks.json --player-summary runs/stable/player_summary.json --output-dir runs/stable_ocr --device cuda --max-crops-per-player 40 --sample-step 6 --save-crops
```

Renderiza el video usando dorsal cuando haya lectura fiable y `P{id}` como fallback:

```powershell
python tools/render_tracks.py --video "2026-04-27 13-40-33.mp4" --tracks runs/stable/tracks.json --calibration config/court_calibration.json --jersey-numbers runs/stable_ocr/jersey_numbers.json --events runs/stable/events.json --output-events runs/stable/events_with_ocr.json --output runs/stable/annotated_numbers.mp4
```

El render activa por defecto la trayectoria densa de balon y una estela corta. Para desactivarlo, usa `--no-dense-ball-track`; para permitir interpolaciones visuales mas largas o cortas, ajusta `--dense-ball-max-gap`.

Para este clip hay una correccion supervisada de identidad revisada visualmente:

```powershell
python tools/render_tracks.py --video "2026-04-27 13-40-33.mp4" --tracks runs/stable/tracks.json --calibration config/court_calibration.json --jersey-numbers runs/stable_ocr/jersey_numbers.json --identity-overrides config/identity_overrides_2026-04-27.json --output runs/stable/annotated_numbers_overrides.mp4
```

Por defecto el render interpola huecos cortos de dorsales estables de dos cifras. Si `#42` aparece antes y despues de un hueco, se dibuja una caja estimada durante los frames perdidos. Esas cajas existen solo para continuidad visual del render: no cuentan como evidencia OCR, no anaden lecturas nuevas y no deben interpretarse como una observacion real del dorsal en esos frames. Puedes desactivarlo con `--no-interpolate-jersey-gaps` o ajustar `--max-interpolation-gap`.

El OCR usa EasyOCR con allowlist de digitos. Se agregan varias lecturas por frame y luego varios frames por jugador, para evitar que una lectura aislada cambie el dorsal. Ademas se exportan `identity_segments`: votos de dorsal por `track_id`/equipo. El render usa esos segmentos antes que el dorsal agregado por `player_id`, asi que si un `player_id` mezcla a #42 y #93 tras un cruce, el dorsal no se propaga a todo el jugador equivocado. Si el mismo dorsal de un digito aparece en varios jugadores del mismo equipo, solo se conserva el candidato mas fuerte y los demas vuelven a `P{id}`. Los dorsales de dos cifras pueden agrupar fragmentos, por ejemplo `dark_42`.

## 6. Ruta SAM2 tipo Roboflow

Para cruces fuertes, puedes usar una ruta mas pesada `detector -> SAM2 -> limpieza de mascara -> tracker por caja limpia`:

```powershell
python tools/analyze_video_sam2.py --video "2026-04-27 13-40-33.mp4" --calibration config/court_calibration.json --output-dir runs/sam2_probe --detector-model yolo11m.pt --sam-model sam2_b.pt --device 0 --imgsz 768 --sam-imgsz 1024 --conf 0.15 --max-frames 240 --write-video
```

Para el clip completo quita `--max-frames`. El primer uso puede descargar pesos de SAM2. Si la RTX local se queda corta, usa este mismo comando en Colab con la carpeta subida.

La limpieza de mascaras sigue la idea de Roboflow: se queda con el componente conectado principal del jugador y elimina islas pequenas alejadas, reduciendo cajas que saltan a otro cuerpo cuando dos jugadores se pegan.

## Notas

- El modelo por defecto del script sigue siendo configurable, pero para esta toma YOLO11m con `--device 0` ha dado el mejor equilibrio de tracking. YOLO26s con `--use-nms-head` es la alternativa si priorizamos la linea mas nueva de Ultralytics o mas sensibilidad de deteccion.
- Si YOLO no ve el balon, existe un fallback experimental por blob naranja con `--ball-color-fallback`, pero no esta activado por defecto porque en esta pista puede confundir lineas/equipamiento rojo con el balon.
- Tambien puedes pasar un detector propio con `--ball-model`. Ese modelo puede ser de una sola clase (`ball`); el analizador lo transforma internamente a `sports ball`.
- La separacion de equipos ya no se decide frame a frame. Se estabiliza por track con un enfoque inspirado en Roboflow: crop central de camiseta, embedding visual, clustering en dos equipos y reglas fuertes solo cuando la camiseta es inequivoca. Esto reduce cambios de equipo y deja al arbitro como `unknown`.
- Si un `track_id` bruto contiene un cambio sostenido de uniforme, el analizador lo parte en pseudo-tracks antes del stitching. Esto evita que un salto de ByteTrack arrastre una identidad oscura sobre un jugador rojo, o al reves. Puedes desactivarlo con `--no-split-mixed-tracks` o ajustar `--mixed-track-min-segment-frames`.
- Los arbitros y personas cortadas por el borde inferior se excluyen por defecto de `in_play_player`. Si quieres mantener detecciones truncadas, usa `--keep-truncated-players`.
- El filtro de pista usa margen asimetrico. Hay algo mas de tolerancia en la banda cercana (`--court-near-margin`) y menos en la banda de la grada (`--court-far-margin`) para no meter mascota/publico como jugadores.
- Cada deteccion de persona incluye `player_candidate` y `in_play_player`. El segundo intenta quedarse con los 10 jugadores de pista por frame y es lo que usa el detector tactico.
- `trackers/bytetrack_basketball.yaml` aumenta la tolerancia a oclusiones frente al ByteTrack por defecto.
- `trackers/botsort_basketball.yaml` queda como alternativa experimental con ReID. En esta toma no supero a ByteTrack ajustado, pero puede servir en clips con movimientos de camara o cruces distintos.
- El analizador cose fragmentos de tracking en `player_id` usando equipo, embedding, distancia 2D y salto temporal. Puedes desactivarlo con `--no-stitch-tracks` o ajustar `--stitch-max-gap`, `--stitch-max-speed` y `--stitch-min-embedding-sim`.
- Tambien hay un corrector conservador de swaps en cruces entre companeros. Usa continuidad 2D y una firma visual de cuerpo completo; se puede desactivar con `--no-correct-crossings` o ajustar con `--crossing-min-appearance-improvement`.
- `basketball_cv/masks.py` incluye la limpieza de mascaras tipo Roboflow para cuando incorporemos SAM2: conserva el cuerpo principal y elimina segmentos pequenos alejados.
- `tools/analyze_video_sam2.py` ya incorpora esa limpieza en una ruta SAM2 independiente. Es mas lenta, pero esta pensada para estudiar y mejorar cruces tipo #42/#93.
- El detector de pick and roll es de reglas sobre coordenadas 2D. Primero busca manejador, defensor cercano y companero que bloquea cerca del defensor. Luego podremos afinarlo con mas clips etiquetados.
- En esta maquina hay una RTX 2060 y se instalo PyTorch CUDA. Compruebalo con:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Si devuelve `False`, la inferencia local funciona pero sera lenta. Para usar la GPU, instala PyTorch con CUDA desde el selector oficial de PyTorch y luego ejecuta `tools/analyze_video.py` con `--device 0`.

## Colab

Este mismo codigo se puede llevar a Google Colab cuando queramos entrenar o usar modelos mas pesados. La ruta razonable es:

1. Usar local para calibracion, pruebas rapidas y revisar salidas.
2. Usar Colab para fine-tuning del balon/jugadores/cancha.
3. Exportar pesos entrenados y volver a correr local con `--model ruta/al/modelo.pt`.
