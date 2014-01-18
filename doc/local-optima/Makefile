
LATEX = pdflatex -interaction=nonstopmode
OUTPUT=$(shell basename `pwd`)
SECTIONS = 

all: $(OUTPUT).pdf

$(OUTPUT).pdf: paper.tex $(SECTIONS)
	texify < $< > $(subst .tex,.tmp.tex,$<)
	$(LATEX) $(subst .tex,.tmp.tex,$<)
	mv $(subst .tex,.tmp.pdf,$<) $(OUTPUT).pdf

bib: paper.tex $(SECTIONS)
	$(LATEX) $< 
	$(LATEX) $< 
	biblatex $(subst .tex,,$<)
	$(LATEX) $< 

clean:
	rm -rf *.{aux,bbl,blg,log,out}
	rm -rf $(OUTPUT).pdf


.PHONY: clean bib
