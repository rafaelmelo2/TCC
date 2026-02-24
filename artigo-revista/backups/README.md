# Artigo para revista — Predição de indicativos financeiros (B3)

Versão em formato de **artigo** (5–10 páginas) para divulgação/revista, baseada na monografia (TCC) UFG *Predição Automática de Indicativos Financeiros para Bolsa de Valores Considerando o Aspecto Temporal*.

## Estrutura modular

O artigo está dividido em arquivos separados para facilitar edição:

| Arquivo | Conteúdo |
|---------|----------|
| `main.tex` | Entrada principal: preâmbulo, título, autor e `\input` das seções |
| `resumo.tex` | Resumo e palavras-chave |
| `introducao.tex` | Seção Introdução |
| `desenvolvimento.tex` | Seção Desenvolvimento (contexto, dados, modelo, validação, resultados, limitações) |
| `conclusao.tex` | Seção Conclusão |
| `referencias.tex` | Referências bibliográficas |

## Como compilar

Requisitos: LaTeX (pdflatex ou latexmk) com pacotes `babel`, `graphicx`, `geometry`, `amsmath`, `booktabs`, `hyperref`, `newtxtext`, `newtxmath`.

Como as referências estão em **BibTeX** (`referencias.bib`), é necessário rodar também o `bibtex`:

```bash
cd artigo-revista
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Ou, com `latexmk` (ele detecta o BibTeX):

```bash
cd artigo-revista
latexmk -pdf main.tex
```

O PDF gerado será `main.pdf`.

### Fonte (letra)

O artigo usa a família **Times-like** (`newtxtext` + `newtxmath`), comum em revistas científicas. Se a “letra” parecer diferente de outro documento (por exemplo, do TCC em PDF), é porque o TCC pode usar outra fonte (ex.: Computer Modern ou a classe ABNT). Para deixar igual ao TCC, você pode comentar no `main.tex` as linhas `\usepackage{newtxtext}` e `\usepackage{newtxmath}` e recompilar; o documento passará a usar a fonte padrão do LaTeX (Computer Modern).

## Imagens

O artigo referencia figuras que devem ficar na pasta `../images/` em relação a `artigo-revista/` (ou seja, na pasta `images` na raiz do repositório):

- `delimitacao-tarefa.png` — esquema da tarefa de previsão direcional
- `validacao-temporal-walk-forward.png` — esquema da validação walk-forward
- `dm_heatmap_pvalores.png` — heatmap dos p-valores do teste de Diebold–Mariano

Se essas imagens estiverem no projeto do TCC (por exemplo em `TCC UFG/` ou em outra pasta), copie-as para `images/` na raiz do repositório ou ajuste `\graphicspath` em `artigo.tex` para o caminho correto.

## Base

Conteúdo extraído e adaptado do PDF da monografia e do texto em `codigo/pipeline/others/generate_text_for_ia_by_pdf/thesis.txt`.
