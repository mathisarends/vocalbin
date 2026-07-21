# Vision

## Was dieses Projekt macht

`vocalbin` ist eine kleine, eigenständige Python-Bibliothek für die dateibasierte
OpenAI Audio API. Sie stellt eine saubere, typisierte und asynchrone API für zwei
Anwendungsfälle bereit:

- Audiodateien oder Audio-Bytes mit OpenAI Speech-to-Text transkribieren
- Text mit OpenAI Text-to-Speech in Audio-Bytes umwandeln

Die Bibliothek kapselt OpenAI-spezifische SDK-Aufrufe hinter stabilen ABC-Ports,
validiert Modellfähigkeiten frühzeitig und normalisiert Antworten, ohne relevante
Rohdaten zu verlieren. Eine kleine Credentials-Grenze liest die für den OpenAI-Client
benötigten Umgebungsvariablen ein. Die Bibliothek bleibt unabhängig von Settings-,
Web-, Storage- und Domain-Code einer konkreten Anwendung.

Das Projekt soll bewusst klein bleiben. Eine geringe API-Oberfläche, klare Typen,
wenige Abhängigkeiten und vorhersehbares Verhalten sind wichtiger als eine große
Feature-Liste.

## Was dieses Projekt nicht macht

`vocalbin` ist keine allgemeine Audio- oder Voice-Agent-Plattform. Insbesondere
gehören folgende Aufgaben nicht zum Kernumfang:

- Audioaufnahme, Wiedergabe oder Gerätesteuerung
- Audio-Konvertierung, Schneiden, Normalisierung oder sonstige Signalverarbeitung
- Persistenz, Uploads, Objekt-Storage oder Datenbankintegration
- Realtime-Sprachdialoge, Voice-Agent-Orchestrierung oder Telefonie
- Web-Framework-, UI-, CLI- oder anwendungsspezifische Integrationen
- Eine allgemeine Credentials- oder Settings-Infrastruktur über das Einlesen der
  für den OpenAI-Client benötigten Umgebungsvariablen hinaus
- Anwendungsspezifische Geschäftslogik oder Abhängigkeiten zu anderen Projekten

Providerunabhängige Ports definieren die Grenze der Bibliothek. Sie sind kein Ziel,
beliebig viele Speech-Provider und deren unterschiedliche Konzepte in einer großen
Abstraktion zu vereinheitlichen.

## Leitlinie für zukünftige Pull Requests

Ein Feature ist grundsätzlich im Einklang mit der Vision, wenn es mindestens einen
der beiden Kernabläufe direkt verbessert und dabei die Bibliotheksgrenze respektiert.
Typische passende Änderungen sind:

- Neue oder geänderte OpenAI-Modelle, Stimmen, Formate und Endpoint-Parameter
- Korrekturen an Capability-Validierung und Antwortnormalisierung
- Verbesserungen an Typisierung, Fehlerverhalten oder Client-Konfiguration
- Rückwärtskompatible Vereinfachungen der öffentlichen API
- Tests, Dokumentation, Packaging und Wartbarkeit

Vor der Umsetzung oder Annahme eines Pull Requests sollten folgende Fragen mit Ja
beantwortet werden können:

1. Unterstützt die Änderung unmittelbar Speech-to-Text oder Text-to-Speech über die
   OpenAI Audio API?
2. Passt sie in die vorhandenen Request-, Response-, Port- und Client-Grenzen, ohne
   anwendungsspezifische Abhängigkeiten einzuführen?
3. Bleibt die öffentliche API klein, asynchron, typisiert und verständlich?
4. Ist die zusätzliche Abhängigkeit oder Komplexität durch einen konkreten Nutzen
   gerechtfertigt?
5. Sind Verhalten, Validierung und Rückwärtskompatibilität durch Tests abgesichert?

Wenn eine Änderung Audioverarbeitung, Framework-Integration, Persistenz oder
Orchestrierung benötigt, sollte sie bevorzugt in der aufrufenden Anwendung oder in
einem separaten Paket umgesetzt werden. Grenzfälle sollten nur aufgenommen werden,
wenn sie für mehrere Nutzer relevant sind, den Kernablauf deutlich vereinfachen und
keine dauerhafte Kopplung an einen einzelnen Anwendungskontext erzeugen.

Ein Pull Request sollte abgelehnt oder aufgeteilt werden, wenn er den Projektumfang
nur deshalb erweitert, weil die technische Umsetzung hier möglich ist. Maßgeblich
ist nicht, ob ein Feature mit Sprache oder Audio zu tun hat, sondern ob es die klare
Aufgabe dieser Bibliothek besser erfüllt.
