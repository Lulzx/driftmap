.PHONY: all clean figures

all: main.pdf

main.pdf: main.tex references.bib
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main

figures:
	python3 generate_figures.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf *.synctex.gz

arxiv: main.pdf
	mkdir -p arxiv-submission
	cp main.tex arxiv-submission/
	cp main.bbl arxiv-submission/
	cp -r figures arxiv-submission/ 2>/dev/null || true
	cd arxiv-submission && zip -r ../arxiv.zip .
	rm -rf arxiv-submission
	@echo "Created arxiv.zip for submission"
