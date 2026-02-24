# LaTeX + BibTeX para artigo-revista (modelo SBC)
$pdf_mode = 1;
$bibtex_use = 2;
# -interaction=nonstopmode: continua apesar do erro normalsfcodes (babel/LaTeX 2020+)
# "; true" faz o comando retornar 0 para o latexmk completar bibtex e novas passagens
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S ; true';
$clean_ext = 'synctex.gz synctex(busy) run.xml tex.bak bcf fdb_latexmk run toc out fls';
