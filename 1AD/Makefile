LATEXMK ?= latexmk
MAIN    ?= main

.PHONY: pdf clean veryclean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEXMK) -pdf -interaction=nonstopmode -file-line-error $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex

veryclean:
	$(LATEXMK) -C $(MAIN).tex
