# Immagini e suoni

## `images/`

| File | Cosa è |
|---|---|
| `substitution_log.png` | Il tronco della modalità Naruto, un'illustrazione con sfondo trasparente. |
| `container_40hc.png` | La fiancata di un container 40' high cube azzurro (12,192 × 2,896 m): lamiera ondulata, longheroni, blocchi d'angolo, strisce gialle e nere da high cube, stella e scritta MAERSK, matricola ISO 6346 `MSKU 471225 1` con cifra di controllo valida, `45G1` e `9'6"`. |
| `tung_body.png` | Tung Tung Tung Sahur senza il braccio della mazza. Piedi in (210, 975), spalla in (332, 450). |
| `tung_bat.png` | Braccio e mazza, orizzontali, con la spalla in (30, 60): l'app li ruota attorno alla spalla. |
| `tung_burst.png` | Il fumetto "TUNG!" che compare a ogni bastonata. |
| `bombardiro.png` | Bombardiro Crocodilo rivolto a sinistra; il cavo parte dall'anello in (590, 303). |

Container, Tung Tung Tung Sahur, il fumetto e Bombardiro Crocodilo sono disegnati via codice dagli script in [`tools/`](../tools). Fumo, linee del teletrasporto, gru, polvere, gancio e stelline vengono disegnati in tempo reale dall'app, così seguono la persona e si adattano a ogni risoluzione.

## `sounds/`

| File | Quando suona |
|---|---|
| `vanish.MP3` | Naruto, alla sparizione e al ritorno. |
| `teleport.wav` | Teletrasporto: sweep da 300 a 1800 Hz, fruscio e un clic quando sparisci. |
| `container_drop.wav` | Container: fischio di caduta, poi tonfo metallico e piccoli rimbalzi. |
| `container_crane.wav` | Gru: argano, aggancio dello spreader, due clic dei twistlock, argano in salita e "boing". |
| `tung_beating.wav` | Tung Tung Sahur: passi di legno, colpi di tamburo di legno e fruscii delle bastonate. |
| `bombardiro_rescue.wav` | Bombardiro Crocodilo: eliche, verricello, gancio, cigolio dell'allungamento e "boing". |
| `tung_voice.wav`, `bombardiro_voice.wav` | Voci di riserva, con la sintesi vocale di macOS: Damayanti abbassata per "tung" e "sahur", Rocko per "Bombardiro Crocodilo!". |

I suoni partono circa 60 ms prima dell'animazione per compensare il ritardo di avvio di `ffplay`. Effetti e voci di Tung Tung Sahur suonano insieme e condividono lo stesso guadagno, così sommati non distorcono.

## `voci/`

Le voci originali dei meme: `tung.wav` (nove "tung" e "sahur") e `bombardiro.wav` (solo "Bombardiro Crocodilo"). Se in questa cartella c'è un file che si chiama `tung` o `bombardiro` (MP3, WAV, M4A, OGG, AAC o FLAC), l'app lo usa al posto della voce di riserva in `sounds/`.
