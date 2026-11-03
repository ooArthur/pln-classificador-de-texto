# ============================================================
# CLASSIFICADOR DE TEXTO - PLN
# ------------------------------------------------------------
# Tema da apresentação: Classificação de Texto.
# O programa lê um texto em português e decide sozinho em qual
# categoria ele se encaixa: Spam, Notícia ou Suporte.
#
# Segue o pipeline mostrado no slide:
#   Texto original -> Limpeza -> TF-IDF -> Modelo decide -> Resposta
#
# Hospedagem: Vercel (o Flask é reconhecido pelo arquivo app.py
# e pela variável "app").
# ============================================================

# ------------------------------------------------------------
# Bibliotecas
# ------------------------------------------------------------
import csv                      # lê o dados.csv (biblioteca padrão do Python, não precisa instalar)
import re                       # expressões regulares: usada para separar o texto em palavras
import unicodedata              # remove acentos (ex.: "grátis" vira "gratis")
from pathlib import Path        # monta o caminho do dados.csv de forma segura, em qualquer sistema

import numpy as np              # contas com vetores/matrizes (argmax, ordenação dos pesos)
from flask import Flask, jsonify, render_template, request
#   Flask: framework web leve. Cria as rotas (/ e /api/...).
#   jsonify: devolve um dicionário Python como resposta JSON.
#   render_template: entrega a página HTML da pasta templates/.
#   request: acessa o que o navegador enviou (o texto digitado).
from sklearn.feature_extraction.text import TfidfVectorizer
#   TfidfVectorizer: transforma texto em números (TF-IDF). É a etapa
#   "Transformar em números" do slide.
from sklearn.linear_model import LogisticRegression
#   LogisticRegression: o modelo que decide a categoria e dá a
#   probabilidade de cada uma. É a etapa "O modelo decide".
from sklearn.model_selection import cross_val_score
#   cross_val_score: validação cruzada, para medir a acurácia do modelo.
from sklearn.pipeline import Pipeline
#   Pipeline: junta TF-IDF + modelo em um objeto só (fit/predict de uma vez).

app = Flask(__name__)

# ------------------------------------------------------------
# Configuração
# ------------------------------------------------------------
# Palavras muito comuns que não ajudam a diferenciar categorias.
# Já estão sem acento, porque a limpeza remove os acentos antes.
STOPWORDS = set(
    "a o as os um uma de do da dos das em no na nos nas por para com que e ou se ao "
    "aos meu minha seu sua eu voce nao mas ja foi sao esta estou ainda isso este essa".split()
)

# O rótulo no CSV é "noticia" (sem acento); na tela mostramos "Notícia".
NOMES = {"spam": "Spam", "noticia": "Notícia", "suporte": "Suporte"}


# ------------------------------------------------------------
# Etapa 2 do pipeline: limpeza do texto
# ------------------------------------------------------------
def limpar(texto: str) -> str:
    """Deixa o texto em um formato padrão para o modelo comparar palavras."""
    # 1) tudo em minúsculas; NFKD separa a letra do acento ("é" vira "e" + "´")
    t = unicodedata.normalize("NFKD", texto.lower())
    # 2) joga fora os acentos soltos, ficando só com as letras
    t = "".join(c for c in t if not unicodedata.combining(c))
    # 3) pega só sequências de letras/números (descarta pontuação, links e símbolos)
    palavras = re.findall(r"[a-z0-9]+", t)
    # 4) remove as stopwords e devolve o texto limpo
    return " ".join(p for p in palavras if p not in STOPWORDS)


# ------------------------------------------------------------
# Treino do modelo
# ------------------------------------------------------------
# Corpus = coleção de textos de exemplo, cada um com seu rótulo
# (a categoria certa). Cada linha do CSV é: texto,rotulo
with open(Path(__file__).parent / "dados.csv", encoding="utf-8", newline="") as f:
    linhas = list(csv.DictReader(f))

X = [limpar(l["texto"]) for l in linhas]   # textos já limpos
y = [l["rotulo"] for l in linhas]          # categorias certas

# Pipeline em duas peças:
#   tfidf -> dá peso alto a palavras que identificam uma categoria
#            (ex.: "gratis" no spam) e peso baixo às que aparecem em tudo.
#            ngram_range=(1, 2) usa palavras soltas e pares de palavras
#            ("fazer login"). sublinear_tf=True evita que uma palavra
#            repetida muitas vezes domine o resultado.
#   clf   -> Regressão Logística aprende quais palavras empurram para
#            cada categoria. C=10 deixa a regularização mais fraca, pois
#            temos poucos exemplos.
modelo = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=10, max_iter=1000)),
    ]
)

# Acurácia em validação cruzada (cv=5): divide o corpus em 5 partes e
# treina/testa 5 vezes, sempre testando em textos que o modelo não viu.
acuracia_cv = float(cross_val_score(modelo, X, y, cv=5).mean())

# Treino final com todos os exemplos. O corpus é pequeno, então treina
# em milissegundos toda vez que o servidor sobe (não precisa salvar
# um arquivo .joblib, que dependeria da versão do scikit-learn).
modelo.fit(X, y)


# ------------------------------------------------------------
# Rotas
# ------------------------------------------------------------
@app.get("/")
def index():
    # Entrega a página (templates/index.html)
    return render_template("index.html")


@app.get("/api/info")
def info():
    # Dados do modelo para mostrar no rodapé da página
    return jsonify(
        exemplos=len(y),
        classes={NOMES[c]: y.count(c) for c in NOMES},
        acuracia_cv=round(acuracia_cv, 3),
    )


@app.post("/api/classificar")
def classificar():
    # Pega o texto enviado pelo navegador (JSON) e tira espaços das pontas
    texto = (request.get_json(silent=True) or {}).get("texto", "").strip()

    # Validações: vazio, grande demais ou sem nada depois da limpeza
    if not texto:
        return jsonify(erro="Digite um texto."), 400
    if len(texto) > 2000:
        return jsonify(erro="Texto muito longo (máx. 2000 caracteres)."), 400
    limpo = limpar(texto)
    if not limpo:
        return jsonify(erro="Nada para classificar após a limpeza."), 400

    # Pega as duas peças de dentro do pipeline para poder explicar a resposta
    tfidf, clf = modelo.named_steps["tfidf"], modelo.named_steps["clf"]

    # Etapa 3: texto -> vetor de números (TF-IDF)
    vetor = tfidf.transform([limpo])

    # Etapa 4: probabilidade de cada categoria; a maior vence
    probs = clf.predict_proba(vetor)[0]
    i = int(np.argmax(probs))

    # Explicação da resposta (slide 4): entre as palavras do texto,
    # quais mais empurraram para a categoria escolhida?
    # contribuição = valor TF-IDF da palavra x peso dela na categoria
    termos = tfidf.get_feature_names_out()
    contrib = vetor.multiply(clf.coef_[i]).toarray()[0]
    top = [
        {"termo": termos[j], "peso": round(float(contrib[j]), 3)}
        for j in np.argsort(contrib)[::-1][:5]   # as 5 maiores
        if contrib[j] > 0
    ]

    # Etapa 5: resposta final
    return jsonify(
        limpo=limpo,
        categoria=NOMES[clf.classes_[i]],
        confianca=round(float(probs[i]), 3),
        probabilidades={NOMES[c]: round(float(p), 3) for c, p in zip(clf.classes_, probs)},
        explicacao=top,
        # quantas palavras do texto o modelo conhece (0 = chute)
        vocabulario_reconhecido=int(vetor.nnz),
    )
