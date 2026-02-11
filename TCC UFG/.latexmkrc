# Inclui a pasta packages/ no caminho de busca do LaTeX e do BibTeX
# para que abntex2cite.sty, abntex2-options.bib e outros arquivos locais sejam encontrados
$ENV{'TEXINPUTS'} = './packages//:.:' . ($ENV{'TEXINPUTS'} || '');
$ENV{'BIBINPUTS'} = './packages//:.:' . ($ENV{'BIBINPUTS'} || '');
