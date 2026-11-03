# pln-classificador-de-texto

Classificador de texto em português feito para a apresentação de **Classificação de Texto** da disciplina de Processamento de Linguagem Natural (PLN), Fatec Cotia.

Dado um texto, o modelo decide se ele é **Spam**, **Notícia** ou **Suporte**, seguindo o pipeline do slide:

`Texto original -> Limpeza -> TF-IDF -> Modelo decide -> Resposta`

Site: https://pln-classificador-texto.vercel.app

## Como funciona

- **Limpeza** (`limpar` em `app.py`): minúsculas, sem acento, sem pontuação e sem stopwords.
- **TF-IDF** (`TfidfVectorizer`): transforma o texto em números, dando mais peso às palavras que identificam uma categoria.
- **Regressão Logística** (`LogisticRegression`): decide a categoria e dá a probabilidade de cada uma.
- **Explicação**: mostra as palavras do texto que mais pesaram na decisão.

O corpus (`dados.csv`) tem 60 frases escritas para a aula (20 por categoria). É pequeno e didático; a acurácia em validação cruzada fica em torno de 87%.

## Estrutura

| Arquivo | Para que serve |
|---|---|
| `app.py` | Servidor Flask: treina o modelo e expõe `/`, `/api/classificar` e `/api/info` |
| `dados.csv` | Frases de treino com o rótulo de cada uma |
| `templates/index.html` | Página do site |
| `requirements.txt` | Bibliotecas Python |

## Rodar localmente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
flask --app app run
```

Abra http://127.0.0.1:5000

## Deploy

Hospedado na Vercel, que reconhece o Flask pelo `app.py`:

```bash
npx vercel deploy --prod
```

Arthur Almeida
