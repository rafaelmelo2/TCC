%%
%% Customizações do abnTeX2 (http://abnTeX2.googlecode.com) para o
%% Curso de Ciências da computação, Unidade Acadêmica Especial de 
%% Biotecnologia, Universidade Federal de Goiás.
%%
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%   http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%% 
%% The Current Maintainer of this work is Humberto Lidio Antonelli, 
%% Contact: humbertoantonelli@gmail.com
%%
%% Further information about abnTeX2 are available on 
%% http://abntex2.googlecode.com/
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{packages/ufgrc}[2016/08/16 v1.2.9 Classe UFGRC baseado em abnTeX2 - Trabalhos de Conclusão de Curso de Ciência da Computação - Ibiotec/UFCAT]
\RequirePackage{xifthen}

%================================================================================
% Declaração de opções
%================================================================================
% Opções:
% 	tcc1:           Formata documento para qualificação
%   tcc2:           
% 	pre-defesa:     Formata documento para versão de defesa (utilizado 
%                   apenas para versão completa)
%   pro-defesa:     Formata documento para versão final (utilizado apenas
%                   para versão completa)
% -------------------------------------------------------------------------------

\DeclareOption*{%
    \PackageWarning{ufgrc}{Unknown option ‘\CurrentOption’}%
}

\newboolean{completo}
\DeclareOption{tcc1}{
    \setboolean{completo}{false}
}
\DeclareOption{tcc2}{
    \setboolean{completo}{true}
}
\ExecuteOptions{tcc2}

\newboolean{opt-pre-defesa}
\DeclareOption{pre-defesa}{
    \setboolean{opt-pre-defesa}{true}
}
\DeclareOption{pos-defesa}{
    \setboolean{opt-pre-defesa}{false}
}
\ExecuteOptions{pos-defesa}

%
% Repassa demais opções a abntex	
%
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{abntex2}%
}
\ProcessOptions\relax


%================================================================================
% Herança da classe `abntex2' e inclusão de pacotes e arquivos necessários
%================================================================================
\LoadClass[% -- opções da classe memoir --
	12pt,				% tamanho da fonte
	openright,			% capítulos começam em pág ímpar 
	                    % (insere página vazia caso preciso)
	twoside,			% para impressão em verso e anverso. Oposto a oneside
	a4paper,			% tamanho do papel. 
	chapter=TITLE		% títulos de capítulos convertidos em letras maiúsculas
	]{packages/abntex2}
	

%================================================================================
% Pacotes fundamentais 
%================================================================================
\RequirePackage{babel}
\RequirePackage{morewrites}         % Resolve o erro “No room for a new \write”
\RequirePackage{cmap}				% Mapear caracteres especiais no PDF
\RequirePackage{lmodern}			% Usa a fonte Latin Modern.
\RequirePackage{lastpage}			% Usado pela Ficha catalográfica
\RequirePackage{indentfirst}		% Indenta o primeiro parágrafo de cada seção.
\RequirePackage{color, xcolor}      % Controle das cores
\RequirePackage{colortbl}           % Controle das cores das células da tabela
                                    % (utilizado no título)
\RequirePackage{graphicx}			% Inclusão de gráficos
\RequirePackage{caption}            % Pacote para definição dos captions
\RequirePackage{subcaption}         % Pacote para definição dos subcaptions 
                                    % (ambientes de subfigures)
\RequirePackage{microtype} 		    % para melhorias de justificação
\RequirePackage{pdfpages}           % inclusão de arquivos PDF

\RequirePackage{hyphenat}           % Pacote para retirar a hifenização do
                                    % Preâmbulo
\RequirePackage{xstring, stringstrings}            % Pacate para manipulação de Strings
\RequirePackage{etoolbox}           % Pacote utilizado para armazenamento de
                                    % resumos
\RequirePackage{amsthm, amsfonts, amsmath, amssymb} % Pacote para a definição dos ambientes matemáticos
\RequirePackage{mathptmx}           % Usando times para equações, opcional
\RequirePackage{listings}           % Pacote para formatação de diversos 
                                    % códigos fontes
\RequirePackage{algorithm, algpseudocode}  % Pacote para escrever algoritmos em pseudo-código
\RequirePackage{pgffor}             % Pacote utilizado para gerar as
                                    % palavras-chave da ficha catalográfica
\RequirePackage{nomencl}            % Pacote para criação da Lista de Siglas
\RequirePackage[noredefwarn,translate=babel,nonumberlist,style=list,toc]{glossaries}	 % Pacote para gerar o glossário
\RequirePackage[brazilian,hyperpageref]{backref} % Paginas com as citações na bibl
\RequirePackage[alf,bibjustif,abnt-etal-text=it,abnt-emphasize=bf,abnt-etal-cite=3,abnt-etal-list=0]{abntex2cite}   % Citações padrão ABNT
\RequirePackage{float}              % Pacote para criação de novos ambientes
\RequirePackage[T1]{fontenc}        % Permite hifenização de palavras com acentos e cedilhas
\RequirePackage[utf8]{inputenc}     % Permitir assentos para português do Brasil
\RequirePackage[figure,table]{totalcount}


%================================================================================
% Comando para configuração do tipo de trabalho
%================================================================================

\providecommand*{\tipoexemplar}{}
\providecommand*{\degree}{}
\providecommand*{\degreecourse}{}
\providecommand*{\tipocurso}{}

% Configuração da situação do trabalho entregue
\ifthenelse{\boolean{completo}}{
    \ifthenelse{\boolean{opt-pre-defesa}}{
        \addto\captionsenglish{% ingles
            \renewcommand{\tipoexemplar}{EXAMINATION BOARD PRESENTATION COPY II}
        }
        \addto\captionsbrazil{% portugues-brasil
            \renewcommand{\tipoexemplar}{EXEMPLAR DE DEFESA II}
        }
    }{
        \addto\captionsenglish{% ingles
            \renewcommand*{\tipoexemplar}{FINAL VERSION}
        }
        \addto\captionsbrazil{% portugues-brasil
            \renewcommand{\tipoexemplar}{VERS{\~A}O REVISADA}
        }
    }
}{
    \addto\captionsenglish{% ingles
        \renewcommand{\tipoexemplar}{EXAMINATION BOARD PRESENTATION COPY I}
    }
    \addto\captionsbrazil{% portugues-brasil
        \renewcommand{\tipoexemplar}{EXEMPLAR DE DEFESA I}
    }
}

\addto\captionsenglish{% ingles
    \renewcommand{\degree}{Bachelor}
    \renewcommand{\degreecourse}{Baccalaureate}
    \tipotrabalho{Monograph}
    \renewcommand{\tipocurso}{Undergraduate}
    % Configuração do texto impresso no Preambulo
    \renewcommand{\imprimirpreambulo}{\nohyphens{\imprimirtipotrabalho~submitted to the Course of \degreecourse~in \imprimircurso~of the \imprimiruniversidade~-- \imprimirregional, in partial fulfillment of the requirements for the degree of \degree~in \imprimircurso. \textit{\tipoexemplar}}}
}
\addto\captionsbrazil{% portugues-brasil
    \ifthenelse{\equal{\imprimirgenero}{M}}
        {\renewcommand{\degree}{Bacharel}}
        {\renewcommand{\degree}{Bacharela}}
    \renewcommand{\degreecourse}{Bacharelado}
    \tipotrabalho{Monografia}
    \renewcommand{\tipocurso}{Gradua{\c c}\~ao}
    % Configuração do texto impresso no Preambulo
    \renewcommand{\imprimirpreambulo}{\nohyphens{\imprimirtipotrabalho~apresentada ao Curso de \degreecourse~em \imprimircurso~da \imprimiruniversidade~-- \imprimirregional, como parte dos requisitos para obten{\c c}{\~a}o do grau de \degree~em \imprimircurso. \textit{\tipoexemplar}}}
}


%================================================================================
% Comando auxiliares
%================================================================================

% Comando para marcar texto para revisão
\newcommand{\rev}[1]{\textcolor{red}{[#1]}}

% Legendas utilizadas em Figuras / Tabelas / Etc.
\newcommand{\fautor}{\legend{Fonte: Elaborada pelo autor.}}
\newcommand{\fadaptada}[2][]{\legend{Fonte: Adaptada de \citeonline[#1]{#2}.}}
\newcommand{\fdireta}[2][]{\legend{Fonte: \citeonline[#1]{#2}.}}
\newcommand{\fdadospesquisa}{\legend{Fonte: Dados da pesquisa.}}

\addto\captionsenglish{% ingles
    % Legendas utilizadas em Figuras / Tabelas / Etc.
    \renewcommand{\fautor}{\legend{Source: Elaborated by the author.}}
    \renewcommand{\fadaptada}[1]{\legend{Source: Adapted from \citeonline{#1}.}}
    \renewcommand{\fdireta}[1]{\legend{Source: \citeonline{#1}.}}
    \renewcommand{\fdadospesquisa}{\legend{Source: Research data.}}
}

% Automatiza o processo de aspas
\newcommand{\aspas}[1]{``{#1}''}

% Comandos utilizados para referenciar corpos flutuantes no texto
\newcommand{\qref}[1]{\quadrorefname~\ref{#1}}      % Quadro X
\newcommand{\aref}[1]{\algoritmorefname~\ref{#1}}   % Algoritmo X
\newcommand{\cref}[1]{\codigorefname~\ref{#1}}      % Código X

\newcommand{\ifequals}[3]{\ifthenelse{\equal{#1}{#2}}{#3}{}}
\newcommand{\case}[2]{#1 #2} % Dummy, so \renewcommand has something to overwrite...
\newenvironment{switch}[1]{\renewcommand{\case}{\ifequals{#1}}}{}

% Palavras-chave do resumo
\newcommand{\palavraschave}{}

%================================================================================
% Pacote para criação da lista de siglas e abreviaturas
%================================================================================

% Configuração do pacote nomencl
% Tamanho da Label da sigla
\setlength{\nomlabelwidth}{0.15\hsize} 
% Separa as abreviaturas de suas definições com pontos (....)
\renewcommand{\nomlabel}[1]{#1 \hfill}
\setlength{\nomitemsep}{-.05in} 
% Comando necessário para a criação da lista de abreviaturas
\makenomenclature
% Traduz o título da lista de abreviaturas
\renewcommand{\nomname}{\listadesiglasname}

% Comando para criação de abreviaturas e siglas
\def\sigla{\@ifstar\@sigla\@@sigla}
% Apenas faz o registro na tabela de siglas
\def\@sigla#1#2{\nomenclature{#1}{#2}}          % \sigla*{}{} 
% Faz o registro na tabela de siglas e insere os dados no corpo do documento
\def\@@sigla#1#2{#2 (#1)\nomenclature{#1}{#2}}  % \sigla{}{}

%================================================================================
% Pacote para criação do GLOSSÁRIO
%================================================================================
% Gera o indíce do Glossário
\makeglossaries
% Traduz o nome do glossário
\addto\captionsbrazil{% portugues-brasil
    \renewcommand{\glossaryname}{Gloss\'ario}
}

% Contador de definições do Glossário
\newcounter{count_glossario}
% Comando para simplificar a definição de palavras no glossário
\newcommand{\newword}[2]{%
    \stepcounter{count_glossario}% Incrementa o contador do glossário
    \newglossaryentry{Def.\arabic{count_glossario}}{%
        name={#1:},%
        description={#2},%
    }%
}

% Alteração no formato do nome do glossário que aparece no sumário
\ifthenelse{\boolean{ABNTEXupperchapter}}{
    \renewcommand{\glssettoctitle}[1]{%
        \renewcommand{\glossarytoctitle}{%
            \texorpdfstring{\MakeTextUppercase{\csname @glotype@#1@title\endcsname}}{\csname @glotype@#1@title\endcsname}
        }
    }
}{}



%================================================================================
% Pacotes de citações
%================================================================================

% Configurações do pacote backref
% Texto padrão antes do número das páginas
\renewcommand{\backref}{}
% Define os textos da citação
\addto\captionsbrazil{% portugues-brasil
    % Usado sem a opção hyperpageref de backref
    \renewcommand{\backrefpagesname}{Citado na(s) p{\'a}gina(s):~}
    \renewcommand*{\backrefalt}[4]{
    	\ifcase #1 %
    		Nenhuma cita{\c c}{\~a}o no texto.%
    	\or
    		Citado na p{\'a}gina #2.%
    	\else
    		Citado nas p{\'a}ginas #2.%
    	\fi}%
}
\addto\captionsenglish{% ingles
    % Usado sem a opção hyperpageref de backref
    \renewcommand{\backrefpagesname}{Cited on page(s):~}
    \renewcommand*{\backrefalt}[4]{
    	\ifcase #1 %
    		No citation.%
    	\or
    		Citation on page #2.%
    	\else
    		Citations on pages #2.%
    	\fi}%
}
% ---

%================================================================================
% Definição/Alteração de comandos uitlizados
%================================================================================

% Permite comandos com @ no nome
\makeatletter

% Checagem dos dialetos da língua portuguesa (portuges, portuguese, brazilian, brazil)
\newcommand*{\IfLangPortuguese}{%
  % O comando pode ser otimizado, colocando nomes por ordem de frequência de utilização.
  \iflanguage{brazil}\@firstoftwo{%
  \iflanguage{brazilian}\@firstoftwo{%
  \iflanguage{portuguese}\@firstoftwo{%
  \iflanguage{portuges}\@firstoftwo\@secondoftwo}}}%
}

% Comando para definir o gênero do autor (M = Masculino / F = Feminino)
\providecommand{\imprimirgenero}{M}
\newcommand{\genero}[1]{\renewcommand{\imprimirgenero}{#1}}

% Comando para definição do nome e sobrenome do author
\renewcommand\autor[2][Nome para Refer\^encia]{
    
    \StrCount{#2}{\space}[\nbmatch]%
    \StrCut[\nbmatch]{#2}{\space}\strfirstautor\strsecondautor

    \global\def\@authname{\strfirstautor}
    \global\def\@authsurn{\strsecondautor}
    \def\ABNTautordata{\@authname~\@authsurn}
    \renewcommand{\imprimirautor}{\@authname~\@authsurn}
    \def\ABNTautorreferencia{#1}
    \def\ABNTautordatabib{\@authsurn, \@authname}
    \global\def\@author{\@authname~\@authsurn}
}

% Retira os dois pontos (:) que vem por padrão da classe Abntex2
\addto\captionsbrazil{% portugues-brasil
    \renewcommand{\orientadorname}{Orientador}
    \renewcommand{\coorientadorname}{Coorientador}
}
\addto\captionsenglish{% ingles
    %% adjusts names from abnTeX2
    \renewcommand{\orientadorname}{Advisor}
    \renewcommand{\coorientadorname}{Co-advisor}
}

% Redefinição do comando de orientador
\renewcommand{\orientador}[3][\orientadorname]{

    \StrCount{#3}{\space}[\nbmatch]%
    \StrCut[\nbmatch]{#3}{\space}\strfirstorientador\strsecondorientador
    
    \addto\captionsbrazil{% portugues-brasil
        \renewcommand{\imprimirorientadorRotulo}{#1}
    }
    \addto\captionsenglish{
        \renewcommand{\imprimirorientadorRotulo}{\orientadorname}
    }
    \newcommand{\titulacaoorientador}{#2}
    \def\ABNTorientadordatabib{\strsecondorientador, \strfirstorientador}
    \renewcommand{\imprimirorientador}{\strfirstorientador~\strsecondorientador}
}

% Redefinição do comando de co-orientador
\renewcommand{\coorientador}[3][\coorientadorname]{

    \StrCount{#3}{\space}[\nbmatch]%
    \StrCut[\nbmatch]{#3}{\space}\strfirstcoorientador\strsecondcoorientador
    
    \addto\captionsbrazil{% portugues-brasil
        \renewcommand{\imprimircoorientadorRotulo}{#1}
    }
    \addto\captionsenglish{
        \renewcommand{\imprimircoorientadorRotulo}{\coorientadorname}
    }
    \newcommand{\titulacaocoorientador}{#2}
    \def\ABNTcoorientadordatabib{\strsecondcoorientador, \strfirstcoorientador}
    \renewcommand{\imprimircoorientador}{\strfirstcoorientador~\strsecondcoorientador}
}

% Comando para definição dos membros da banca
\newcount\@membrobancaconta\@membrobancaconta0
\newcommand\membrobanca[2]{
  \global\@namedef{NomeMembroBanca:\expandafter\the\@membrobancaconta}{#1}
  \global\@namedef{LocalMembroBanca:\expandafter\the\@membrobancaconta}{#2}
  \global\advance\@membrobancaconta by 1
}

% Comando para definição da Universidade
\providecommand{\imprimiruniversidade}{}
\newcommand{\universidade}[1]{\renewcommand{\imprimiruniversidade}{#1}}

% Comando para definição da Regional
\providecommand{\imprimirregional}{}
\newcommand{\regional}[1]{\renewcommand{\imprimirregional}{#1}}

% Comando para definição do Instituto
\providecommand{\imprimirinstituto}{}
\newcommand{\instituto}[1]{\renewcommand{\imprimirinstituto}{#1}}

% Comando para definição do nome do Curso
\providecommand{\imprimircurso}{}
\newcommand{\curso}[1]{\renewcommand{\imprimircurso}{#1}}

% Redefinição do comando \date do Latex
\renewcommand{\date}[3]{%
  \gdef\@day{#1}%
  \gdef\@month{#2}%
  \gdef\@year{#3}%
}

% Define a data
\date{\number\day}{\number\month}{\number\year}

% Redefinição do comando de data do ABNTex
\renewcommand{\data}[3]{%
    \date{#1}{#2}{#3}%
}

% Nome do Mês
\providecommand*{\Monthname}{}
\addto\captionsbrazil{
    \renewcommand\Monthname[1][EMPTY]{%
        \ifnum1=#1Janeiro\else
        \ifnum2=#1Fevereiro\else
        \ifnum3=#1Março\else
        \ifnum4=#1Abril\else
        \ifnum5=#1Maio\else
        \ifnum6=#1Junho\else
        \ifnum7=#1Julho\else
        \ifnum8=#1Agosto\else
        \ifnum9=#1Setembro\else
        \ifnum10=#1Outubro\else
        \ifnum11=#1Novembro\else
        \ifnum12=#1Dezembro\else
        \fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi
    }
}
\addto\captionsenglish{% ingles
    \renewcommand\Monthname[1][EMPTY]{%
      \ifnum1=#1January\else
      \ifnum2=#1February\else
      \ifnum3=#1March\else
      \ifnum4=#1April\else
      \ifnum5=#1May\else
      \ifnum6=#1June\else
      \ifnum7=#1July\else
      \ifnum8=#1August\else
      \ifnum9=#1September\else
      \ifnum10=#1October\else
      \ifnum11=#1November\else
      \ifnum12=#1December\else
      \fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi
    }
}

% Comando para definição dos resumos e palavras-chave
\newcounter{numresumo}
\newcommand{\textoresumo}[3][\languagename]{
  \stepcounter{numresumo}%
  \csdef{res-language-\thenumresumo}{#1}
  \csdef{res-text-\thenumresumo}{#2}
  \csdef{res-keywords-\thenumresumo}{#3}
  \IfStrEq{#1}{\languagename}{\renewcommand{\palavraschave}{#3}}{}
}

% Opção para exibir a lista de figuras
\newcommand{\incluilistadefiguras}{
  \gdef\@incluilistadefiguras{true}
}

% Opção para exibir a lista de tabelas
\newcommand{\incluilistadetabelas}{
  \gdef\@incluilistadetabelas{true}
}

% Opção para exibir a lista de siglas
\newcommand{\incluilistadesiglas}{
  \gdef\@incluilistadesiglas{true}
}

% Opção para exibir a lista de símbolos
\newcommand{\incluilistadesimbolos}{
  \gdef\@incluilistadesimbolos{true}
}

% Opção para exibir a lista de quadros
\newcommand{\incluilistadequadros}{
  \gdef\@incluilistadequadros{true}
}

% Opção para exibir a lista de algoritmos
\newcommand{\incluilistadealgoritmos}{
  \gdef\@incluilistadealgoritmos{true}
}

% Opção para exibir a lista de códigos
\newcommand{\incluilistadecodigos}{
  \gdef\@incluilistadecodigos{true}
}

% Comando para inclusão da ficha catalográfica
\newboolean{opt-fichacatalografica}
\setboolean{opt-fichacatalografica}{false}
\def\incluifichacatalografica{\@ifstar\@incluifichacatalografica\@@incluifichacatalografica}
% Incluir um arquivo externo como ficha catalográfica
\def\@incluifichacatalografica#1{\setboolean{opt-fichacatalografica}{true}\gdef\@arquivofichacatalografica{#1}}          % \incluifichacatalografica*{}
\def\@@incluifichacatalografica{\setboolean{opt-fichacatalografica}{true}}  % \incluifichacatalografica


%================================================================================
% Definição dos ambientes matemáticos
%================================================================================

\providecommand*{\theoremaname}{Theorem}
\providecommand*{\proposicaoname}{Proposition}
\providecommand*{\lemaname}{Lemma}
\providecommand*{\corolarioname}{Corollary}
\providecommand*{\exemploname}{Example}
\providecommand*{\observacaoname}{Note}
\providecommand*{\definicaoname}{Definition}
\providecommand*{\demonstracaoname}{Demonstration}

\providecommand*{\sourcecodeautorefname}{Source code}
\addto\captionsbrazil{% portugues-brasil
    %% ajusta nomes padroes do babel
    \renewcommand{\theoremaname}{Teorema}
    \renewcommand{\proposicaoname}{Proposi{\c c}{\~a}o}
    \renewcommand{\lemaname}{Lema}
    \renewcommand{\corolarioname}{Corol{\'a}rio}
    \renewcommand{\exemploname}{Exemplo}
    \renewcommand{\observacaoname}{Observa{\c c}{\~a}o}
    \renewcommand{\definicaoname}{Defini{\c c}{\~a}o}
    \renewcommand{\demonstracaoname}{Demonstra{\c c}{\~a}o}
}

\theoremstyle{definition}
\newtheorem{teorema}{\theoremaname}
\newtheorem{proposicao}{\proposicaoname}
\newtheorem{lema}{\lemaname}
\newtheorem{corolario}{\corolarioname}
\newtheorem{exemplo}{\exemploname}
\newtheorem{observacao}{\observacaoname}
\newtheorem{definicao}{\definicaoname}
\newenvironment{demonstracao}{\begin{proof}[\demonstracaoname]}{\end{proof}}

%================================================================================
% Criação dos Ambientes Flutuantes para Quadros, Algoritmos e Cógidos
%================================================================================

\newcommand{\listquadroname}{Lista de quadros}
\newcommand{\quadroname}{Quadro}
\newcommand{\quadrorefname}{Quadro}

\newcommand{\listalgoritmoname}{Lista de algoritmos}
\newcommand{\algoritmoname}{Algoritmo}
\newcommand{\algoritmorefname}{Algoritmo}

\newcommand{\listcodigoname}{Lista de c{\'o}digos-fonte}
\newcommand{\codigoname}{C{\'o}digo-fonte}
\newcommand{\codigorefname}{C{\'o}digo-fonte}

\addto\captionsenglish{% ingles
    %% adjusts names from abnTeX2
    \renewcommand{\listquadroname}{List of charts}
    \renewcommand{\quadroname}{Chart}
    \renewcommand{\quadrorefname}{Chart}
    \renewcommand{\listalgoritmoname}{List of algorithms}
    \renewcommand{\algoritmoname}{Algorithm}
    \renewcommand{\algoritmorefname}{Algorithm}
    \renewcommand{\listcodigoname}{List of source codes}
    \renewcommand{\codigoname}{Source code}
    \renewcommand{\codigorefname}{Source code}
}


% ---
% Formatação de ambiente Quadro
% ---
\newfloat{quadro}{htbp}{loq}[chapter]
\floatname{quadro}{\quadroname}
\floatstyle{plaintop}
\restylefloat{quadro}
\newlistof{listofquadros}{loq}{\listquadroname}
\newlistentry{quadro}{loq}{0}
\renewcommand{\thequadro}{\thechapter.\@arabic\c@quadro}
\setfloatadjustment{quadro}{\centering}

\renewcommand{\cftquadroname}{\quadroname\space}
\renewcommand*{\cftquadroaftersnum}{\hfill\textendash\hfill}


% ---
% Formatação de ambiente Algoritmo
% ---

% Altera o rótulo a ser usando no elemento pré-textual "Lista de algoritmos"
\renewcommand{\listalgorithmname}{\listalgoritmoname}%
% Altera o nome padrão do rótulo
%\renewcommand{\algorithmname}{\algoritmoname}%
\floatname{algorithm}{\algoritmoname}

% Configura a ``Lista de algoritmos'' conforme as regras da ABNT (para abnTeX2)
\begingroup
\let\newcounter\@gobble\let\setcounter\@gobbletwo
  \globaldefs\@ne \let\c@loldepth\@ne
  \newlistof{algorithmt}{loa}{\listalgorithmname}
  \newlistentry{algorithmt}{loa}{0}
\endgroup

   \let\oldlistofalgorithms\listofalgorithms
\newcommand{\NEWnumberline}[1]{\hspace*{-12pt}Algoritmo #1~~\textendash~~}%
\renewcommand{\listofalgorithms}{%
\let\OLDnumberline\numberline%
\let\numberline\NEWnumberline%
\oldlistofalgorithms%
\let\numberline\OLDnumberline%
}%

% Cria um novo ambiente "algoritmo" que inclui o ambiente "algorithm"
\newenvironment{algoritmo}[1][htb]{
    \begin{algorithm}[#1]%
}{\end{algorithm}}


% ---
% Formatação de código-fonte
% ---

% Altera o nome padrão do rótulo 
\renewcommand{\lstlistingname}{\codigoname}
% Altera o rótulo a ser usando no elemento pré-textual "Lista de códigos"
\renewcommand{\lstlistlistingname}{\listcodigoname}

% Configura a ``Lista de códigos'' conforme as regras da ABNT (para abnTeX2)
\begingroup
\let\newcounter\@gobble\let\setcounter\@gobbletwo
  \globaldefs\@ne \let\c@loldepth\@ne
  \newlistof{listings}{lol}{\lstlistlistingname}
  \newlistentry{lstlisting}{lol}{0}
\endgroup

\renewcommand{\cftlstlistingaftersnum}{\hfill\textendash\hfill}

\let\oldlstlistoflistings\lstlistoflistings
\renewcommand{\lstlistoflistings}{%
   \begingroup%
   \let\oldnumberline\numberline%
   \renewcommand{\numberline}{\lstlistingname\space\oldnumberline}%
   \oldlstlistoflistings%
   \endgroup}
   
\def\lst@numbersymbol{}
\lst@Key{numbersymbol}{}{\def\lst@numbersymbol{#1}}
\lst@Key{numbers}{none}{%
    \IfStrEq{#1}{none}{%
      \let\lst@PlaceNumber\@empty
    }{%
      \IfStrEq{#1}{left}{%
        \def\lst@PlaceNumber{\llap{\normalfont\lst@numberstyle{\thelstnumber\lst@numbersymbol}\kern\lst@numbersep}}%
      }{%
        \IfStrEq{#1}{right}{%
          \def\lst@PlaceNumber{\rlap{\normalfont\kern\linewidth \kern\lst@numbersep\lst@numberstyle{\lst@numbersymbol\thelstnumber}}}%
        }{%
          \PackageError{Listings}{Numbers #1 unknown}\@ehc
        }%
      }%
    }%
}

% Cria um ambiente "codigo" estilizando o ambiente "lstlisting" 
\newcounter{countcodigo}
\setcounter{countcodigo}{1}
\lstnewenvironment{codigo}[1][]
  {\lstset{basicstyle=\ttfamily\ABNTEXfontereduzida,
            numbers=left, 
            numbersep=6pt,
            numberstyle=\ABNTEXfontereduzida,
            frame=tb,
            framerule=0.8pt,
            xleftmargin=20.25pt,
            framexleftmargin=21.25pt,
            framexbottommargin=5pt,
            framextopmargin=5pt,
            abovecaptionskip=10pt,
            numbersymbol=$:$,
            #1
          }
}
{\setcounter{countcodigo}{\value{lstnumber}}}

\addto\captionsbrazil{% portugues-brasil
% Declaracoes em Português
\algrenewcommand\algorithmicend{\textbf{fim}}
\algrenewcommand\algorithmicdo{\textbf{fa{\c c}a}}
\algrenewcommand\algorithmicwhile{\textbf{enquanto}}
\algrenewcommand\algorithmicfor{\textbf{para}}
\algrenewcommand\algorithmicif{\textbf{se}}
\algrenewcommand\algorithmicthen{\textbf{ent{\~a}o}}
\algrenewcommand\algorithmicelse{\textbf{sen{\~a}o}}
\algrenewcommand\algorithmicreturn{\textbf{retorna}}
\algrenewcommand\algorithmicfunction{\textbf{fun{\c c}{\~a}o}}
\algrenewcommand\algorithmicforall{\textbf{para todo}}
\algrenewcommand\algorithmicloop{\textbf{la{\c c}o}}
\algrenewcommand\algorithmicrepeat{\textbf{repita}}
\algrenewcommand\algorithmicuntil{\textbf{at{\'e}}}
\algrenewcommand\algorithmicprocedure{\textbf{procedimento}}
\algrenewcommand\algorithmicrequire{\textbf{Requer:}}
\algrenewcommand\algorithmicensure{\textbf{Assegure:}}
}
% Rearranja os finais de cada estrutura
\algrenewtext{EndWhile}{\algorithmicend\ \algorithmicwhile}
\algrenewtext{EndFor}{\algorithmicend\ \algorithmicfor}
\algrenewtext{EndIf}{\algorithmicend\ \algorithmicif}
\algrenewtext{EndFunction}{\algorithmicend\ \algorithmicfunction}
% O comando For, a seguir, retorna 'para #1 -- #2 até #3 faça'
\algnewcommand\algorithmicto{\textbf{at{\'e}}}
\algrenewtext{For}[3]{\algorithmicfor\ #1 $\gets$ #2 \algorithmicto\ #3 \algorithmicdo}


%================================================================================
% Comando para configurar os captions no documento
%================================================================================
\renewcommand{\ABNTEXfontereduzida}{\small} % Fonte reduzida para 11pt, conforme recomendação da USP
\DeclareCaptionFont{ABNTEXfontereduzida}{\ABNTEXfontereduzida}
\captionsetup{position=above, labelfont=ABNTEXfontereduzida, font=ABNTEXfontereduzida}

\DeclareCaptionFormat{listing}{\hrule height.8pt depth0pt #1#2#3}
\captionsetup[lstlisting]{format=listing,singlelinecheck=false, margin=0pt,labelsep=endash,labelfont=bf, font=normalsize}
\captionsetup[algorithm]{singlelinecheck=false, margin=0pt,labelsep=endash,labelfont=bf, font=normalsize}


%================================================================================
% Lista de Símbolos
%================================================================================

% Comando para inserir simbolos na lista de símbolos
\newcommand{\simbolo}[2]{
  \addcontentsline{los}{simbolo}{\ensuremath{#1} --- #2}%
}

% Apresentação dos símbolos na lista de símbolos
\newcommand\l@simbolo[2]{\par #1}

% Criação da lista de símbolos
\newcommand{\inserelistadesimbolos}
{
  \@ifundefined{@incluilistadesimbolos}{}{
	 \pdfbookmark[0]{\listadesimbolosname}{listadesimbolos}
     \pretextualchapter{\listadesimbolosname}
     {\setlength{\parindent}{0cm}
       \@starttoc{los}
     }
     \cleardoublepage
  }
}


%================================================================================
% Definição de comandos a serem inseridos diretamente no início do documento
%================================================================================

\newcommand{\inserelistadefiguras}
{
  \@ifundefined{@incluilistadefiguras}{}{
    \pdfbookmark[0]{\listfigurename}{lof}
    \listoffigures*
    \cleardoublepage
  }
}

\newcommand{\inserelistadetabelas}
{
  \@ifundefined{@incluilistadetabelas}{}{
    \pdfbookmark[0]{\listtablename}{lot}
    \listoftables*
    \cleardoublepage
  }
}

\newcommand{\inserelistadesiglas}
{
  \@ifundefined{@incluilistadesiglas}{}{
    \pdfbookmark[0]{\listadesiglasname}{listadesiglas}
    \printnomenclature
    \cleardoublepage
  }
}

\newcommand{\inserelistadequadros}
{
  \@ifundefined{@incluilistadequadros}{}{
    \pdfbookmark[0]{\listquadroname}{loq}
    \listofquadros*
    \cleardoublepage
  }
}

\newcommand{\inserelistadealgoritmos}
{
  \@ifundefined{@incluilistadealgoritmos}{}{
    \pdfbookmark[0]{\listalgoritmoname}{alg}
    \listofalgorithms
    \cleardoublepage
  }
}

\newcommand{\inserelistadecodigos}
{
  \@ifundefined{@incluilistadecodigos}{}{
    \pdfbookmark[0]{\lstlistlistingname}{lol}
    \begin{KeepFromToc}
    \lstlistoflistings
    \end{KeepFromToc}
    \cleardoublepage
  }
}

%================================================================================
% INDICE REMISSIVO
%================================================================================

\makeindex

%================================================================================
% Configuração do modelo de capa exigido pelo DC/UFG
%=============================================================================

% alterando a capa
\renewcommand{\imprimircapa}{%
  \begin{capa}%
    \center
    \begin{Spacing}{1.1}
    \begin{otherlanguage}{brazil}
        {\sffamily\Large\imprimiruniversidade} \\
        {\sffamily\large\imprimirregional}\\
        {\sffamily\large\imprimirinstituto} \\
        {\sffamily\large Curso de \degreecourse\ em \imprimircurso}\\
    \end{otherlanguage}
    \end{Spacing}
    
    \vspace*{\fill}
    \begin{minipage}{14.8cm}
        \center
        \rule{\textwidth}{1pt} \\
        \vspace{8mm}
        \begin{Spacing}{1.5}
            {\sffamily\LARGE\imprimirtitulo} \\
        \end{Spacing}
        \vspace{10mm}
        \nohyphens{\sffamily\large\textbf\imprimirautor} \\
        \vspace{5mm}
        \rule{\textwidth}{1pt} \\
    \end{minipage}%
    \vspace*{\fill}
 
    {\sffamily\large\imprimirlocal}\\
    {\sffamily\large\@year}
    
    \vspace*{1cm}
  \end{capa}
}

%==============================================================================
% Folha de Rosto
%==============================================================================

\renewcommand{\folhaderostocontent}{

    \begin{center}
    
        \vspace{0.5cm}
        
        {\sffamily\large\bfseries\nohyphens{\imprimirautor}}
        
        \vfill
        
        \begin{Spacing}{1.5}
            {\sffamily\LARGE\nohyphens{\imprimirtitulo}}
        \end{Spacing}
        
        \vfill
        
        \abntex@ifnotempty{\imprimirpreambulo}{%
          \hspace{.43\textwidth}
          \begin{minipage}{8.4cm}
            \begin{Spacing}{0.95}
                {\sffamily\imprimirpreambulo}%
                
                \vspace{15pt}
                {\sffamily\imprimirorientadorRotulo: \titulacaoorientador{} \nohyphens{\sffamily\imprimirorientador}}%
                \abntex@ifnotempty{\imprimircoorientador}{%
                    \\{\sffamily\imprimircoorientadorRotulo: \titulacaocoorientador{} \nohyphens{\sffamily\imprimircoorientador}}
                }%
            \end{Spacing}
          \end{minipage}%
        }%
        
        
        \vspace*{\fill}
        
        {\sffamily\large\imprimirlocal}\\
        {\sffamily\large\@year}
        \vspace*{1cm}

  \end{center}
  
} %folha de rosto


%================================================================================
% Ficha Catalografica
%================================================================================

% Contador das palavras-chave
\newcounter{count}
% Enumera a lista de palavras-chave
\newcommand{\PROCESS}[1]{%
    \foreach\i in{#1}{%
\stepcounter{count}%
\arabic{count}.\enspace\i.\ %
    }%
}

\newcommand{\inserefichacatalografica}
{
    \@ifundefined{@arquivofichacatalografica}
    {
    
        \begin{fichacatalografica}
        	\vspace*{\fill}					% Posição vertical
        	
        	% Conteúdo
	\begin{otherlanguage}{brazil}
	\begin{tabular}{|p{1.4cm}p{10.1cm}p{1.4cm}|}
	
	\multicolumn{3}{c}{\sffamily\small Ficha de identificação da obra elaborada pelo autor, através do} \\
	\multicolumn{3}{c}{\sffamily\small  Programa de Geração Automática do Sistema de Bibliotecas da UFG.} \\ \hline
	& & \\
	& & \\
	& {\fontsize{10.3pt}{1em}\selectfont\sffamily\ABNTautordatabib} &\\ 
	& {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}\imprimirtitulo~ [manuscrito] / \imprimirautor. -- \@year.} & \\ 
	& {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}\pageref{LastPage} p.\iftotalfigures: il. \fi} &\\
	& &\\
	& {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}\imprimirorientadorRotulo: \titulacaoorientador~\imprimirorientador} &\\ 
	\abntex@ifnotempty{\imprimircoorientador}{%
	   & {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}\imprimircoorientadorRotulo: \titulacacoorientador~\imprimircoorientador} & \\
	}
	& {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}\imprimirtipotrabalho~(\tipocurso) -- \imprimiruniversidade, \imprimirinstituto, \imprimircurso, \@year.} &\\
	& {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}Bibliografia.} & \\
	%& {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}Inclui.} & \\
	%Inclui siglas, abreviaturas, símbolos, tabelas, algoritmos, lista de figuras, lista de tabelas
	& & \\ 
	& {\fontsize{10.3pt}{1em}\selectfont\sffamily\hspace{0.6cm}\expandafter\PROCESS\expandafter{\palavraschave}I. \ABNTorientadordatabib, orient.\ II.\ \abntex@ifnotempty{\imprimircoorientador}{%
	\ABNTcoorientadordatabib, coorient.\ III.\ }Título.} &\\
	& & \\
	& & \\
	& & \\
	& & \\
	& & \\
	& \begin{flushright} CDU 004\end{flushright} & \\ \hline
	\end{tabular}
	\end{otherlanguage}
	
	\vspace{1.5cm}

        \end{fichacatalografica}
    }{
        \begin{fichacatalografica}
            \includepdf{\@arquivofichacatalografica}
        \end{fichacatalografica}
    }
}


%==============================================================================
% Folha de Aprovação
%==============================================================================

% Definição da folha de aprovação
\newcommand{\folhadeaprovacaoUFG}{
    \begin{folhadeaprovacao}
        \begin{center}
            
            \vspace{0.5cm}
            {\sffamily\large\bfseries\nohyphens{\imprimirautor}}
            
            \vspace{4cm}
            {\sffamily\LARGE\imprimirtitulo}
            \vspace*{\fill}
        
            \hspace{.45\textwidth}
            \begin{minipage}{.5\textwidth}
                \begin{Spacing}{0.95}
                    \nohyphens{\sffamily\imprimirtipotrabalho~apresentada ao curso de \degreecourse~ em \imprimircurso~da \imprimiruniversidade~-- \imprimirregional.}
                \end{Spacing}
            \end{minipage}%
            \vspace*{\fill}
       
            {\sffamily Trabalho aprovado em {\@day} de \Monthname[\@month] de {\@year}.}
       \end{center}   
    	
    	\assinatura{\sffamily\textbf{\imprimirorientador}\\\imprimirorientadorRotulo} 
    	
        \count1=0
        \@whilenum \count1<\@membrobancaconta \do {
          \assinatura{
            \sffamily\textbf{ \csname NomeMembroBanca:\the\count1 \endcsname }\\%
            \csname LocalMembroBanca:\the\count1 \endcsname
          }
          \advance\count1 by 1
        }
        
        \vspace*{\fill}  
       \begin{center}
        
        {\sffamily\large\imprimirlocal}\\
        {\sffamily\large\@year}
        \vspace*{1cm}
        
      \end{center}
      
    \end{folhadeaprovacao}
}% fim da \folhadeaprovacaoufgcac


%================================================================================
% Dedicatória
%================================================================================

% Comando para definição da dedicatória
\def\textodedicatoria{\@ifstar\@textodedicatoria\@@textodedicatoria}
% Inclui texto de um arquivo como dedicatória
\def\@textodedicatoria#1{\gdef\@dedicatoria{\input{#1}}} % \textodedicatoria*{}{}
% Inclui o texto como dedicatória
\def\@@textodedicatoria#1{\gdef\@dedicatoria{#1}}        % \textodedicatoria{}{}

% Criação da dedicatória
\newcommand{\inserededicatoria}
{
  \@ifundefined{@dedicatoria}{}{
    \begin{dedicatoria}
   \vspace*{\fill}
   \centering
   \noindent
      \textit{\@dedicatoria}
     \vspace*{\fill}
\end{dedicatoria}
  }
}


%================================================================================
% Agradecimentos
%================================================================================

%{\@month}ição dos agradecimentos
\def\textoagradecimentos{\@ifstar\@textoagradecimentos\@@textoagradecimentos}
% Inclui texto de um arquivo como agradecimentos
\def\@textoagradecimentos#1{\gdef\@agradecimentos{\input{#1}}} % \textoagradecimentos*{}{}
% Inclui o texto como agradecimentos
\def\@@textoagradecimentos#1{\gdef\@agradecimentos{#1}} % \textoagradecimentos{}{}

% Criação dos agradecimentos
\newcommand{\insereagradecimentos}
{
  \@ifundefined{@agradecimentos}{}{
    \begin{agradecimentos}
      \@agradecimentos
    \end{agradecimentos}
  }
}


%================================================================================
% Epígrafe
%================================================================================

% Comando para definição da epígafe
\def\textoepigrafe{\@ifstar\@textoepigrafe\@@textoepigrafe}
% Inclui texto de um arquivo como epígafe
\def\@textoepigrafe#1{\gdef\@epigrafe{\input{#1}}}      % \textoepigrafe*{}{} 
% Inclui o texto como epígafe
\def\@@textoepigrafe#1{\gdef\@epigrafe{#1}}             % \textoepigrafe{}{}

% Criação da epígafe
\newcommand{\insereepigrafe}
{
  \@ifundefined{@epigrafe}{}{
	\begin{epigrafe}
      \vspace*{\fill}
	  \begin{flushright}
		\textit{\@epigrafe}
	 \end{flushright}
    \end{epigrafe}	 
    
  }
}


%================================================================================
% Resumo
%================================================================================

% Criação do resumo e palavras chaves no documento
\newcommand{\insereresumo}
{

% Títulos do Resumo
\csdef{res-title-english}{Abstract}
\csdef{res-title-brazil}{Resumo}
\csdef{res-title-french}{Résumé}
\csdef{res-title-spanish}{Resumen}
% Labels: Palavras-chave
\csdef{res-label-keywords-english}{Keywords}
\csdef{res-label-keywords-brazil}{Palavras-chave}
\csdef{res-label-keywords-french}{Mots-clés}
\csdef{res-label-keywords-spanish}{Palabras clave}

  \setcounter{count}{0}
  \whileboolexpr
    { test {\ifnumcomp{\value{count}}{<}{\thenumresumo}} }%
    {
        \stepcounter{count}
        
        \if@openright\cleardoublepage\else\clearpage\fi
        \phantomsection
        \begin{resumo}[\csuse{res-title-\csuse{res-language-\thecount}}]
            \begin{otherlanguage*}{\csuse{res-language-\thecount}}
                \begin{SingleSpace}
                    {\MakeUppercase{\ABNTautorreferencia}}. {\textbf{\imprimirtitulo}.} \@year. \pageref{LastPage} p. \imprimirtipotrabalho~(\tipocurso) -- \imprimirinstituto, \imprimiruniversidade~-- \imprimirregional, \imprimirlocal.
                \end{SingleSpace}
                \vskip 20pt
                \csuse{res-text-\thecount}
                
                \vspace{\onelineskip}
                
                \noindent
                \textbf{\csuse{res-label-keywords-\csuse{res-language-\thecount}}:}
                \@tempswafalse
                    \@for\next:=\csuse{res-keywords-\thecount}\do
                    {\if@tempswa,~\else\@tempswatrue\fi\next}.
            \end{otherlanguage*}
        \end{resumo}
    }
}




%================================================================================
% Configuração dos Estilo dos Capítulo
%================================================================================

\makechapterstyle{icmc}{%
  \renewcommand{\chapterheadstart}{} 
  
    % Secao secundaria (Section) Caixa baixa, Negrito
    \renewcommand*{\cftsectionfont}{\bfseries}
    % Secao terciaria (Subsection) Caixa baixa, Negrito, italico
    \renewcommand*{\cftsubsectionfont}{\itshape\bfseries}
    % Secao quaternaria (Subsubsection) Caixa baixa, italico
    \renewcommand*{\cftsubsubsectionfont}{\itshape}
    % Secao quinquenária (Subsubsubsection) Caixa baixa
    \renewcommand*{\cftparagraphfont}{\normalsize}


  % tamanhos de fontes de chapter e part	
   \ifthenelse{\equal{\ABNTEXisarticle}{true}}{%
     \setlength\beforechapskip{\baselineskip}
     \renewcommand{\chaptitlefont}{\ABNTEXsectionfont\ABNTEXsectionfontsize}
   }{%else
     \setlength{\beforechapskip}{0pt}
     \renewcommand{\ABNTEXchapterfontsize}{\LARGE}
    
    %\renewcommand{\ABNTEXchapterfont}{\sffamily\bfseries}  %alteração da fonte dos capítulos, seções e subseções
    \renewcommand{\chaptitlefont}{\ABNTEXchapterfont\bfseries\ABNTEXchapterfontsize}
   }
  
  \renewcommand{\chapnumfont}{\chaptitlefont}
  \renewcommand{\parttitlefont}{\ABNTEXpartfont\ABNTEXpartfontsize}
  \renewcommand{\partnumfont}{\ABNTEXpartfont\ABNTEXpartfontsize}
  \renewcommand{\partnamefont}{\ABNTEXpartfont\ABNTEXpartfontsize}
  
  % tamanhos de fontes de section, subsection, subsubsection e subsubsubsection
  \setsecheadstyle{\ABNTEXsectionfont\ABNTEXsectionfontsize\bfseries\ABNTEXsectionupperifneeded}
  \setsubsecheadstyle{\ABNTEXsubsectionfont\ABNTEXsubsectionfontsize\bfseries\itshape\ABNTEXsubsectionupperifneeded}
  \setsubsubsecheadstyle{\ABNTEXsubsubsectionfont\ABNTEXsubsubsectionfontsize\itshape\ABNTEXsubsubsectionupperifneeded}
  \setsubsubsubsecheadstyle{\ABNTEXsubsubsubsectionfont\ABNTEXsubsubsubsectionfontsize\ABNTEXsubsubsubsectionupperifneeded}
 
  % impressao do numero do capitulo
  \renewcommand{\chapternamenum}{}
  
  % impressao do nome do capitulo
  \renewcommand{\printchaptername}{%
   %\chaptitlefont
   %\ifthenelse{\boolean{abntex@apendiceousecao}}{\appendixname}{\chaptername}%
  }
    
  % impressao do titulo do capitulo
  \def\printchaptertitle##1{%
    
    \setboolean{ABNTEXupperchapter}{true}
    
    \ifthenelse{\boolean{abntex@innonumchapter}}{
        \vskip 0ex \hrulefill\chaptitlefont\bfseries\ABNTEXchapterupperifneeded{##1}
        \vskip -0.6ex\hfill\rule{.8\textwidth}{0.5pt} 
        \vskip -2.8ex\hfill\rule{.8\textwidth}{2pt}
        \vskip 1.5ex
    
    }{%
    % else
        {\hrulefill
        {\renewcommand{\arraystretch}{1.5} %  1 is the default, change whatever you need
        \begin{tabular}{|c|}
            \rowcolor{black}\color{white}\normalsize\ABNTEXchapterfont
              \ifthenelse{\boolean{abntex@apendiceousecao}}{\MakeTextUppercase{\appendixname}}{\MakeTextUppercase{\chaptername}}  \\ 
            \vspace{-1.5ex}\\ %coloquei para aumentar o espaço entre o título e o número
            \resizebox{!}{1.1cm}{\ABNTEXchapterfont\thechapter}
            \\[2.5ex]
            \hline
        \end{tabular}}} \\
        \vskip 4.3ex \flushright\chaptitlefont\bfseries\ABNTEXchapterupperifneeded{##1} \\
        \vskip -0.6ex\hfill\rule{.8\textwidth}{0.5pt} \\
        \vskip -2.8ex\hfill\rule{.8\textwidth}{2pt}\\
        \vskip 1.5ex
	}    
        
  }
       
  % impressao do numero do capitulo     	
  \renewcommand{\printchapternum}{%
    \setboolean{abntex@innonumchapter}{false}
  }
  \renewcommand{\afterchapternum}{}
  
  % impressao do capitulo nao numerado
  \renewcommand\printchapternonum{%
     \setboolean{abntex@innonumchapter}{true}%
    }
}
\chapterstyle{icmc}


%================================================================================
% Configurações diversas
%================================================================================

% ---
% Configurações de aparência do PDF final
% ---
% alterando o aspecto da cor azul
\definecolor{blue}{RGB}{41,5,195}

% informações do PDF
\hypersetup{
        unicode=true, % Codificação dos caracteres especiais no Bookmark
		pdftitle={\@title}, 
		pdfauthor={\@author},
    	pdfsubject={\imprimirpreambulo},
	    pdfcreator={LaTeX with abnTeX2/ICMC-USP},
		pdfkeywords={\palavraschave}, 
		colorlinks=true,       		% false: boxed links; true: colored links
    	linkcolor=blue,          	% color of internal links
    	citecolor=blue,        		% color of links to bibliography
    	filecolor=magenta,      	% color of file links
		urlcolor=blue,
		bookmarksdepth=4
}

% alterando o aspecto da cor azul
\definecolor{blue}{RGB}{41,5,195}
\definecolor{white}{RGB}{255,255,255}
\definecolor{black}{RGB}{0,0,0}

% Nomes constantes do abnTeX2/UFGRC (definicoes padroes)
\providecommand*{\chartautorefname}{Chart}
\providecommand*{\algorithmautorefname}{Algorithm}
\providecommand*{\sourcecodeautorefname}{Source code}
\addto\captionsbrazil{% portugues-brasil
    %% ajusta nomes padroes do babel
    \renewcommand{\sectionrefname}{se{\c c}\~ao~}   % \Sref{ }
    \renewcommand{\chapterrefname}{Cap{\'i}tulo~}   % \Cref{ }
    \renewcommand{\appendixrefname}{Ap{\^e}ndice~}  % \Aref{ }
    \renewcommand{\bookrefname}{Livro~}             % \Bref{ }
    \renewcommand{\partrefname}{Parte~}             % \Pref{ }
    \renewcommand{\figurerefname}{Figura}           % \fref{ }
    \renewcommand{\tablerefname}{Tabela}            % \tref{ }
    \renewcommand{\pagerefname}{p{\'a}gina}         % \pref{ }
    %% ajusta nomes usados com a macro \autoref
    \renewcommand{\chartautorefname}{Quadro}
    \renewcommand{\algorithmautorefname}{Algoritmo}
    \renewcommand{\sourcecodeautorefname}{Código-fonte}
}

% Corrige warning gerado pelo uso do comando \uppercase
\ifthenelse{\boolean{ABNTEXsumario-abnt-6027-2012}}{%v
    \renewcommand{\bibsection}{%
        \chapter*{\bibname}
        \bibmark
        \ifnobibintoc\else
        \phantomsection
        \addcontentsline{toc}{chapter}{\texorpdfstring{\MakeTextUppercase{\bibname}}{\bibname}}
        \fi
        \prebibhook
    }
}{}

% Níveis de subseções
\setcounter{tocdepth}{4}
\setcounter{secnumdepth}{4}

% Ajusta o espaçamento dos parágrafos do resumo
\setlength{\absparsep}{18pt}

% O tamanho do parágrafo é dado por:
\setlength{\parindent}{1.3cm}

% Controle do espaçamento entre um parágrafo e outro:
\setlength{\parskip}{0.2cm}  % tente também \onelineskip

% Pasta padrão para as imagens
\graphicspath{{images/}}

% Preenchimento padrão 
\universidade{Universidade Federal de Goi\'as}
\regional{Regional Catal\~ao}
\instituto{Unidade Acad\^emica Especial de Biotecnologia}
\curso{Ci\^encias da Computa{\c c}\~ao}
\local{Catal\~ao -- GO}


%================================================================================
% Configurações ao iniciar documento
%================================================================================

\AtBeginDocument{

    % Centraliza o caption das tabelas do IBGE, conforme recomendação da USP
    \renewcommand{\configurecaptions}{\captionstyle[singlelinecheck=on]{}}
    
    \pretextual
    
    % Capa
    \imprimircapa

    % Folha de Rosto
    \imprimirfolhaderosto*{} % (o * indica que haverá a ficha bibliográfica)
	\inserefichacatalografica
	\newpage
	
	% Folha de Aprovação
	\folhadeaprovacaoUFG

    \ifthenelse{\boolean{completo}}{
        % DEDICATÓRIA / AGRADECIMENTO / EPÍGRAFE
        \inserededicatoria
        \insereagradecimentos
        \insereepigrafe
    }{}

    % RESUMOS
    \insereresumo

    % Insere as Listas que forem setadas no documento
    \inserelistadefiguras
    \inserelistadequadros    
    \inserelistadealgoritmos
    \inserelistadecodigos
    \inserelistadetabelas
    \inserelistadesiglas
    \inserelistadesimbolos

	\pdfbookmark[0]{\contentsname}{toc}
    \tableofcontents*
    \cleardoublepage
    
    % Configuração dos caracteres especiais utilizados no pacote <listings>
    \lstset{%
        frame=top,
        frame=bottom,
        breakatwhitespace=false,         
        breaklines=true,                 
        keepspaces=true,                 
        numbers=left,                    
        numbersep=5pt,                  
        showspaces=false,                
        showstringspaces=false,
        showtabs=false,                  
        tabsize=2,
        inputencoding=utf8,
        extendedchars=true,
        literate=%
        {é}{{\'{e}}}1
        {è}{{\`{e}}}1
        {ê}{{\^{e}}}1
        {ë}{{\¨{e}}}1
        {É}{{\'{E}}}1
        {Ê}{{\^{E}}}1
        {û}{{\^{u}}}1
        {ù}{{\`{u}}}1
        {â}{{\^{a}}}1
        {à}{{\`{a}}}1
        {á}{{\'{a}}}1
        {ã}{{\~{a}}}1
        {Á}{{\'{A}}}1
        {Â}{{\^{A}}}1
        {Ã}{{\~{A}}}1
        {ç}{{\c{c}}}1
        {Ç}{{\c{C}}}1
        {õ}{{\~{o}}}1
        {ó}{{\'{o}}}1
        {ô}{{\^{o}}}1
        {Õ}{{\~{O}}}1
        {Ó}{{\'{O}}}1
        {Ô}{{\^{O}}}1
        {î}{{\^{i}}}1
        {Î}{{\^{I}}}1
        {í}{{\'{i}}}1
        {Í}{{\~{Í}}}1
        {Ú}{{\'U}}1
        {ú}{{\'u}}1
        {Ü}{{\"U}}1
        {ü}{{\"u}}1
    }
    
    % Numeração contínua do elementos do presentes no documento
    \counterwithout{quadro}{chapter}
    \counterwithout{lstlisting}{chapter}
    
    \mainmatter

    % Retira espaço extra obsoleto entre as frases.
    \frenchspacing
    
    \IfLangPortuguese{}{
        \bibliographystyle{packages/abntexalfenglish}
    }
}

% Finaliza comandos com @ no nome
\makeatother