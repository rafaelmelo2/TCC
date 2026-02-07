# TCC UFG — LaTeX

Documento principal do TCC em LaTeX (classe UFGRC + abnTeX2). Este README descreve como rodar o projeto **em outro computador** depois de clonar o repositório.

---

## O que já vem no repositório

- **Texto e estrutura**: `thesis.tex`, pasta `tex/`, `references.bib`
- **Classe e pacotes**: pasta `packages/` (ufgrc.cls, abntex2, etc.) e `u.cmap`
- **Configuração do editor**: `.vscode/settings.json` na raiz do repo (LaTeX Workshop, paths, PDF em aba, etc.)

Nada disso precisa ser copiado ou alterado ao mudar de máquina.

---

## O que fazer no outro computador

### 1. Clonar/abrir o repositório

```bash
git clone <url-do-repositorio> TCC
cd TCC
```

(ou abrir a pasta do repo no Cursor/VS Code.)

### 2. Instalar uma distribuição LaTeX

É preciso ter **TeX Live** (ou equivalente) com `pdflatex` e `latexmk` no PATH.

| Sistema | Comando / como instalar |
|--------|-------------------------|
| **Linux (Debian/Ubuntu/WSL)** | `sudo apt update && sudo apt install -y texlive-latex-extra texlive-lang-portuguese latexmk` |
| **Linux (instalação mais completa)** | `sudo apt install -y texlive-full` |
| **Windows** | [MiKTeX](https://miktex.org/download) ou [TeX Live](https://tug.org/texlive/) — durante a instalação, marque opção de adicionar ao PATH. |
| **macOS** | [MacTeX](https://www.tug.org/mactex/) ou `brew install --cask mactex` |

Testar no terminal:

```bash
pdflatex --version
latexmk -v
```

### 3. Instalar Cursor (ou VS Code) e LaTeX Workshop

- Instalar [Cursor](https://cursor.sh) ou [VS Code](https://code.visualstudio.com).
- Instalar a extensão **LaTeX Workshop** (James-Yu): no Cursor/VS Code, aba Extensions, buscar por "LaTeX Workshop".

### 4. Abrir o projeto e compilar

- Abrir a **pasta raiz do repositório** no Cursor (a que contém `TCC UFG` e `.vscode`).
- Abrir `TCC UFG/thesis.tex`.
- Compilar: **Ctrl+Alt+B** (Build) ou Command Palette → "LaTeX Workshop: Build LaTeX project".
- Ver o PDF: **Ctrl+Alt+V** (View LaTeX PDF) — use esse viewer para o PDF atualizar após cada build.

---

## Resumo rápido

1. **Clone o repo** → já traz `.vscode`, `packages/`, `u.cmap`, etc.  
2. **Instale TeX Live (ou MiKTeX no Windows)** com `pdflatex` e `latexmk`.  
3. **Instale Cursor/VS Code** e a extensão **LaTeX Workshop**.  
4. **Abra a raiz do repo**, abra `thesis.tex`, dê Build e use "View LaTeX PDF" para ver/atualizar o PDF.

Não é necessário alterar ou adicionar nada dentro do repositório no outro computador; só instalar o TeX e o editor com a extensão.
