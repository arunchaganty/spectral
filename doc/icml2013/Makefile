
LATEX = pdflatex -interaction=nonstopmode
OUTPUT=$(shell basename `pwd`)
SECTIONS = abstract.tex intro.tex model.tex algo.tex evaluation.tex discussion.tex proofs.tex 

all: $(OUTPUT).pdf

$(OUTPUT).pdf: paper.tex $(SECTIONS)
	$(LATEX) $<
	mv $(subst .tex,.pdf,$<) $@

bib: paper.tex $(SECTIONS)
	$(LATEX) $< 
	$(LATEX) $< 
	bibtex $(subst .tex,,$<)
	$(LATEX) $< 

clean:
	rm -rf *.{aux,bbl,blg,log,out}
	rm -rf $(OUTPUT).pdf


.PHONY: clean bib
