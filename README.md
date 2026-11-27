# Sistema di analisi automatica dello squat

Progetto di tesi triennale in Informatica per l'analisi
automatica dell'esecuzione dello squat attraverso tecniche
di Computer Vision e Machine Learning.

Il sistema utilizza MediaPipe per il rilevamento dei landmark
corporei e OpenCV per l'elaborazione dei video.

## Funzionalità

- Rilevamento dei landmark corporei.
- Costruzione del modello corporeo.
- Riconoscimento delle ripetizioni e delle fasi dello squat.
- Analisi deterministica degli errori di esecuzione.
- Controllo dell'affidabilità dei landmark rilevati.
- Generazione dello scheletro originale e corretto.
- Classificazione delle ripetizioni tramite Random Forest.
- Validazione di MediaPipe mediante marker fisici rilevati
  attraverso OpenCV.

## Installazione

Creare un ambiente virtuale Python:

```bash
python -m venv .venv
```

Attivare l'ambiente virtuale su Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Installare le dipendenze:

```bash
pip install -r requirements.txt
```

## Struttura del progetto

- `src/body`: costruzione del modello corporeo.
- `src/errors`: controlli deterministici degli errori.
- `src/mediapipe`: rilevamento dei landmark corporei.
- `src/ml`: creazione del dataset, estrazione delle feature,
  addestramento e classificazione.
- `src/output`: generazione e correzione dello scheletro.
- `src/squat`: riconoscimento delle fasi e gestione
  delle ripetizioni.
- `src/validation`: validazione dei risultati di MediaPipe.
- `dataset`: ripetizioni salvate in formato JSON.
- `models`: modelli utilizzati dal sistema.
- `videos`: video utilizzati come input.
- `outputs`: risultati generati dal programma.
- `main.py`: punto di ingresso del programma.

## Modalità di esecuzione

Tutti i comandi devono essere eseguiti dalla cartella
principale del progetto.

### Analisi dello squat

```bash
python main.py analyze --video videos/test_video_1.mp4
```

Il sistema riconosce le ripetizioni, individua gli errori
di esecuzione ed effettua la classificazione mediante
il modello di Machine Learning.

Vengono generati due video nella cartella `outputs`:

- Scheletro originale.
- Scheletro con le correzioni degli errori rilevati.

### Validazione

```bash
python main.py validation --video videos/nome_video.mp4
```

Questa modalità permette di confrontare i landmark rilevati
da MediaPipe con i marker fisici rilevati mediante OpenCV.

Il video utilizzato deve contenere i marker fisici necessari
alla validazione.

### Creazione del dataset

```bash
python main.py dataset --video videos/test_video_1.mp4
```

Le ripetizioni valide vengono salvate all'interno della
cartella `dataset` in formato JSON.

Le etichette delle ripetizioni devono essere assegnate
manualmente.

Attenzione: ripetere questo comando su un video già presente
nel dataset comporta la sovrascrittura del relativo file JSON
e delle etichette precedentemente assegnate.

## Machine Learning

Per addestrare il classificatore eseguire:

```bash
python -m src.ml.train_classifier
```

Il modello utilizza una Random Forest e rappresenta ogni
ripetizione attraverso 30 frame, considerando le coordinate
normalizzate dei 17 landmark corporei.

La separazione tra training set e test set avviene per
gruppi di partecipanti. La tecnica SMOTE viene utilizzata
per bilanciare le classi durante l'addestramento.

Nella configurazione finale le ripetizioni etichettate
come medium vengono associate alla classe wrong,
ottenendo una classificazione binaria:

- correct: squat generalmente corretto.
- wrong: squat non pienamente corretto.

Il modello addestrato viene salvato in:

`models/squat_classifier.pkl`

Il nuovo addestramento sovrascrive il modello precedente.

## Limitazioni

Il sistema dipende dalla qualità del rilevamento dei
landmark corporei e dalle condizioni di acquisizione
del video.

La classificazione mediante Machine Learning ha carattere
sperimentale e può produrre valutazioni non corrette.

La correzione dello scheletro rappresenta una visualizzazione
grafica delle correzioni suggerite dal sistema.