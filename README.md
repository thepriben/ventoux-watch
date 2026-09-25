# Mont Serein — veille

Le Raspberry Pi 5 regarde la webcam du Mont Serein. Chaque mouvement est gardé et interprété avec le jour, la nuit et la météo. Un avion, une voiture, un bus, un attroupement ou un incendie ne sont nommés que lorsque la lecture est assez sûre. Le dépôt est privé. Le site dans `site/` se consulte en local : sur un compte gratuit, GitHub Pages ne reste pas publié depuis un dépôt privé.

Le flux est celui déjà utilisé par [dataroads-fr84.info](https://dataroads-fr84.info/), source Vision-Environnement. Le site affiche ce direct. Il ne réhéberge pas la vidéo continue.

## Pipeline

1. Une image par seconde, différence avec le fond (OpenCV MOG2). Une tache compacte qui se déplace devient un passage. Un changement de lumière sur toute l’image est ignoré. La balise rouge du sommet est masquée dans `config/zones.json`.
2. YOLO nano, en ONNX, seulement sur le rectangle de ce passage.
3. Une règle pose le nom. Sans nom, la ligne reste dans `data/candidates.jsonl` sur le Pi et n’entre pas dans l’historique.

Un avion n’est publié avec son indicatif que s’il n’y en a qu’un dans le créneau OpenSky, ou un seul vraiment plus bas que les autres. Un bus prend le nom de la ligne Trans'CoVe ou ZOU seulement s’il n’y a qu’une course à ±15 minutes. Les autres passages restent dans l’historique avec une lecture : jour ou nuit, météo, et ce qu’on a pu en dire. Une lueur au crépuscule n’est pas un incendie. Les animaux ne sont pas encore une classe.

Le modèle s’améliore sur cette caméra. Le jour, la nuit et la météo changent la lecture. Chaque passage est conservé. Un endroit qui bouge souvent sans événement nouveau devient une habitude du cadrage, comptée dans `data/learning.json`.

## Sur le Mac, avant le Pi

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -r requirements-export.txt
.venv/bin/python scripts/export_model.py
.venv/bin/python -m unittest discover -s tests -v
```

`models/yolo11n.onnx` part avec le code. Le Pi n’installe pas PyTorch.

Les zones sont dessinées sur l’image de nuit `data/reference.jpg` :

```bash
.venv/bin/python scripts/draw_zones.py
```

Le fichier `data/zones-preview.jpg` sert à les corriger. Les coordonnées dans `config/zones.json` vont de 0 à 1.

## Quand le Pi est branché

Le port USB-C alimente le Pi. La liaison avec l’ordinateur est un câble Ethernet ou le Wi-Fi, puis SSH.

```bash
sudo mkdir -p /opt/ventoux-watch
sudo rsync -a --exclude .venv ./ /opt/ventoux-watch/
cd /opt/ventoux-watch
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
sudo cp deploy/ventoux-watch.service /etc/systemd/system/
sudo systemctl enable --now ventoux-watch
```

Secrets, uniquement sur le Pi, dans `config/local.json` :

```json
{
  "opensky": {"username": "...", "password": "..."},
  "drive": {"credentials": "secrets/drive.json", "folder_id": "..."}
}
```

OpenSky et Drive sont facultatifs. Sans compte OpenSky, l’archive des avions reste anonyme et plus limitée. Sans clé Drive, les photos sont publiées, pas les extraits. La clé Google et le jeton git ne vont pas dans le dépôt.

Le service pousse `data/events.json`, `data/learning.json` et `data/thumbs/` au plus toutes les quinze minutes. Le dépôt étant privé, cette poussée met le code à jour sans rouvrir le site au public.

Pour les extraits : `pip install -r requirements-drive.txt`, un compte de service, et le dossier Drive partagé avec ce compte.

## Voir le site en local

```bash
mkdir -p _site/data/thumbs
cp -R site/. _site/
cp data/events.json _site/data/events.json
cp -R data/thumbs/. _site/data/thumbs/
.venv/bin/python -m http.server 8765 --directory _site
```
