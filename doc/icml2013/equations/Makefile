LATEX = pdflatex
#LATEX = pdflatex -interaction=nonstopmode
OUTPUT=$(shell basename `pwd`)
SECTIONS = $(wildcard *.tex)

all: $(OUTPUT).pdf

$(OUTPUT).pdf: paper.tex $(SECTIONS)
	$(LATEX) $<
	mv $(subst .tex,.pdf,$<) $@

final: paper.tex $(SECTIONS)
	texify < $< > $(subst .tex,.tmp.tex,$<)
	$(LATEX) $(subst .tex,.tmp.tex,$<)
	mv $(subst .tex,.tmp.pdf,$<) $(OUTPUT).pdf

bib: paper.tex $(SECTIONS)
	$(LATEX) $< 
	bibtex $(subst .tex,,$<)
	$(LATEX) $< 
	$(LATEX) $< 

clean:
	rm -rf *.{aux,bbl,blg,log,out}
	rm -rf $(OUTPUT).pdf


.PHONY: clean bib final
