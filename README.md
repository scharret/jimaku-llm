# jimaku-llm (字幕)

Traduit automatiquement les `*.en.srt` d'une bibliothèque Jellyfin en `*.fr.srt` / `*.ja.srt`
(+ `*.ja.ass` avec furigana), via Gemini, Claude ou n'importe quelle API OpenAI-compatible (Ollama, vLLM, LiteLLM).

## Comportement
- **Idempotent** : une sortie existante n'est jamais réécrite (`--force` pour forcer). Un vrai sous-titre FR
  téléchargé par Bazarr est donc prioritaire.
- **Reprise** : chaque lot traduit est mis en cache SQLite → un 503 au milieu d'un film ne fait perdre que le lot courant.
- **Résilience** : backoff exponentiel + jitter sur 408/429/5xx, bascule sur le modèle suivant de `MODELS`,
  puis découpe du lot en deux si le JSON revient cassé.
- **Écriture atomique** (`.tmp` + rename) : Jellyfin ne voit jamais un fichier à moitié écrit.
- Ignore les sources modifiées depuis moins de `SETTLE_S` (Bazarr encore en train d'écrire).
- Clé API passée en header, jamais dans l'URL → elle ne fuit pas dans les logs.

## Lancer
```bash
cp .env.example .env && vi .env           # API_KEY=...
docker compose up -d --build
docker compose logs -f
# one-shot sur un fichier :
docker compose run --rm subtrans file "/media/Movies/X (1993)/X.en.srt"
```

## Glossaire (noms propres cohérents)
`glossary.txt` dans le dossier du film (ou `GLOSSARY_FILE` global) :
```
Burikin Hotel => ブリキンホテル / Hôtel Burikin
Bamboo copter => タケコプター / hélicoptère de bambou
```

## Variables
| Var | Défaut | |
|---|---|---|
| `BACKEND` | `gemini` | `gemini` \| `anthropic` \| `openai` \| `dummy` |
| `MODELS` | `gemini-2.5-flash` | liste ordonnée, repli |
| `OPENAI_BASE_URL` | `http://ollama:11434/v1` | pour `openai` |
| `TARGET_LANGS` | `fr,ja` | |
| `SOURCE_SUFFIXES` | `.en.srt,.eng.srt` | |
| `FURIGANA` | `1` | génère `.ja.ass` |
| `BATCH_SIZE` / `CONTEXT_CUES` | `60` / `8` | |
| `MAX_RETRIES` / `MIN_INTERVAL_S` | `6` / `1.0` | |
| `LIBRARY_PATHS` | `/media` | séparés par `:` |

## Image
`ghcr.io/scharret/jimaku-llm` (amd64/arm64), construite par la CI à chaque push sur `main` et sur les tags `v*`.

## Tests
```bash
pip install requests "fugashi[unidic-lite]" pytest && pytest -q
```
