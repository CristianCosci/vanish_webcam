# VanishCam

Sparisci (e riappari) dalla webcam con un gesto della mano, in diretta, durante le videochiamate. VanishCam riconosce il gesto, separa la tua sagoma dallo sfondo e manda il risultato alla webcam virtuale di OBS, che Teams, Meet, Zoom e le altre app vedono come una webcam qualsiasi.

Il video viene elaborato sul computer: non viene registrato né inviato a nessun server. Nell'uscita ci sono solo il video e gli effetti, senza menu, scritte o istruzioni sovrapposte.

## Le modalità

| | Modalità | Gesto | Cosa succede |
|---|---|---|---|
| 1 | **Normal Vanish** | segno a V | ti dissolvi dal basso verso l'alto, con un bordo luminoso |
| 2 | **Naruto Vanish** | palmi uniti davanti alla webcam | uno sbuffo di fumo e al tuo posto resta un tronco |
| 3 | **Teletrasporto** | indice e medio uniti sulla fronte, per 1 secondo | linee in stile fumetto, ti stringi in un raggio e sparisci |
| 4 | **Container** | indice verso l'alto, per 1 secondo | un container Maersk 40' HC ti cade in testa e ti schiaccia, con tonfo, polvere e tremolio |
| 5 | **Tung Tung Sahur** | pollice verso, per 1 secondo | Tung Tung Tung Sahur arriva con la mazza e a ogni bastonata diventi più piccolo, fino a restare schiacciato a terra |

**Per riapparire** abbassa la mano per mezzo secondo e ripeti il gesto della modalità (dove serve, tenendolo di nuovo per 1 secondo):

- dissolvenza, fumo e teletrasporto si riavvolgono al contrario;
- nel Container arriva una gru che solleva il container, e tu ti riallunghi da terra con un effetto molla;
- in Tung Tung Sahur arriva Bombardiro Crocodilo, ti aggancia con un cavo e ti tira su fino all'altezza normale.

Ogni modalità ha i suoi suoni. Il terminale dice quando il gesto viene riconosciuto; sul video non compare niente.

## Installazione

### macOS

Servono [Homebrew](https://brew.sh), OBS (solo per la sua webcam virtuale) e `ffmpeg`, che riproduce i suoni:

```sh
brew install --cask obs
brew install ffmpeg uv
git clone https://github.com/CristianCosci/vanish_webcam.git
cd vanish_webcam
uv venv --python 3.13
uv pip install -r requirements.txt
```

La prima volta apri OBS e premi **Avvia fotocamera virtuale**: macOS chiede di consentire l'estensione di sistema. Vai in **Impostazioni di Sistema → Generali → Elementi login ed estensioni**, scorri fino a **Estensioni**, apri **Estensioni fotocamera** (ⓘ) e attiva **OBS Virtual Camera**. Poi ferma la fotocamera virtuale e chiudi OBS: l'uscita la produce VanishCam.

Per avviare:

```sh
.venv/bin/python vanish_cam.py --virtual-camera --no-mirror
```

### Windows

Fai doppio clic su `windows\avvia.bat`: la prima volta crea l'ambiente Python in `.venv` e installa tutto, poi apre il menu. Le altre due varianti sono:

- `windows\avvia_virtuale.bat` richiede la webcam virtuale e si ferma con un errore se non c'è;
- `windows\avvia_obs.bat` mostra solo la finestra di anteprima, da catturare in OBS con una sorgente **Cattura finestra** su **Webcam - Anteprima**.

Al posto di OBS si può usare [Unity Capture](https://github.com/schellingb/UnityCapture) con `--virtual-backend unitycapture`.

`windows\build_exe.bat` crea `dist\VanishCam.exe`, un unico eseguibile con modelli, immagini e suoni dentro, che non richiede Python.

## Uso

1. Avvia VanishCam e scegli la modalità dal menu con il mouse o i tasti da `1` a `5`. Per saltare il menu usa `--mode normal`, `naruto`, `teleport`, `container` o `tung`.
2. Esci dall'inquadratura per 4 secondi mentre VanishCam acquisisce lo sfondo; il terminale ti dice quando rientrare.
3. Fai il gesto.

### In videochiamata

- Come fotocamera scegli **OBS Virtual Camera**.
- Spegni sfocatura e sfondi virtuali dell'app di videochiamata: lo sfondo lo gestisce VanishCam.
- Avvia con `--no-mirror`, così chi ti guarda legge dritte le scritte (ad esempio sul container). Teams mostra a specchio solo la tua anteprima, gli altri ti vedono dritto.
- I suoni li senti solo tu: la webcam virtuale porta solo il video.

### Tasti nella finestra di anteprima

| Tasto | Azione |
|---|---|
| `SPAZIO` | acquisisce di nuovo lo sfondo |
| `R` | ti fa riapparire subito |
| `V` | avvia l'effetto senza gesto |
| `F` | schermo intero |
| `Q` o `ESC` | esce e libera la webcam |

### Opzioni

| Opzione | |
|---|---|
| `--mode MODALITÀ` | salta il menu iniziale |
| `--camera N` | webcam da usare (predefinita: 0) |
| `--width`, `--height`, `--fps` | risoluzione e FPS massimi dell'uscita (predefiniti: 960×540, 30) |
| `--no-mirror` | non specchia l'immagine |
| `--virtual-camera`, `--no-virtual-camera` | richiede o disattiva la webcam virtuale; senza opzione la usa se c'è |
| `--virtual-backend`, `--virtual-device` | driver (`obs`, `unitycapture`, `v4l2loopback`) e dispositivo della webcam virtuale |
| `--fullscreen` | apre l'anteprima a schermo intero |
| `--gesture-debug` | stampa nel terminale cosa vede il riconoscimento dei gesti, utile se un gesto non parte |

### Voci di Tung Tung Sahur

Le voci originali dei meme sono in `assets/voci` (`tung.wav` e `bombardiro.wav`). Per cambiarle metti al loro posto un file che si chiami `tung` o `bombardiro` (MP3, WAV, M4A, OGG, AAC o FLAC). Se la cartella è vuota, VanishCam usa le voci di riserva in `assets/sounds`, fatte con la sintesi vocale di macOS. Colpi, eliche e "boing" restano comunque sincronizzati con l'animazione.

## Struttura

```text
vanish_cam.py          avvio (equivale a python -m vanishcam)
vanishcam/
  app.py               menu, webcam, riconoscimento dei gesti e ciclo video
  paths.py             dove si trovano asset e modelli
  sound.py             riproduzione dei suoni (pygame su Windows, ffplay altrove)
  effects/
    naruto.py          fumo e tronco
    teleport.py        linee e raggio del teletrasporto
    container.py       container, gru, polvere
    tung.py            Tung Tung Tung Sahur e Bombardiro Crocodilo
    squash.py          schiacciare, allungare e scuotere la sagoma
assets/                immagini, suoni e voci (dettagli in assets/README.md)
models/                modelli MediaPipe, scaricati al primo avvio
tests/                 test automatici
tools/                 script che disegnano le immagini e sintetizzano i suoni
windows/               avvio, build dell'eseguibile e icona per Windows
```

## Sviluppo

I test girano senza webcam:

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

Container, Tung Tung Tung Sahur, Bombardiro Crocodilo e i loro suoni si rigenerano con gli script in `tools/`; l'uso è scritto in cima a ogni script.

MediaPipe è fermo alla 1.2.0: con la 1.0.1 su Apple Silicon il modello delle mani manda in crash l'app appena viene caricato.

## Crediti

Le modalità Normal e Naruto e la base dell'app vengono da [vanish_webcam](https://github.com/Nicovic17/vanish_webcam) di Nicovic17.
