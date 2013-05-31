default: main.pdf

texfiles := $(wildcard *.tex) $(wildcard figures/*.tex)

sfig:
	git clone https://github.com/percyliang/sfig

figures: sfig figures.js utils.js sfig/internal/sfig.js
	node figures.js
	touch figures

main.pdf: $(texfiles) figures main.bbl
	@pdflatex main < /dev/null > /dev/null || true
	@grep -A 4 "Emergency stop" main.log || grep Warning main.log || true
main.bbl: pliang.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

run:
	go main.pdf

clean:
	rm -f main.aux main.bbl main.blg main.log main.pdf

split: main.pdf
	pdftk main.pdf cat 1-9 output main-primary.pdf
	pdftk main.pdf cat 10-13 output main-supplementary.pdf
