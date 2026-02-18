# Inclui a pasta packages/ no caminho de busca do LaTeX e do BibTeX
# para que abntex2cite.sty, abntex2-options.bib e outros arquivos locais sejam encontrados
$ENV{'TEXINPUTS'} = './packages//:.:' . ($ENV{'TEXINPUTS'} || '');
$ENV{'BIBINPUTS'} = './packages//:.:' . ($ENV{'BIBINPUTS'} || '');

# Só roda makeindex se thesis.idx existir (a classe ufgrc usa \makeindex,
# mas o .idx só é gerado após uma compilação pdflatex bem-sucedida)
$makeindex = 'test -f %S && makeindex -o %D %S; true';
