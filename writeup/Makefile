
MAIN = main
LATEX = pdflatex
BIBER = biber
 
.PHONY: all clean
 
all:
	$(LATEX) $(MAIN)
	$(BIBER) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
 
clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.toc
 