TEST_ENV=( [[ "$$CONDA_PREFIX" == */tennis-face ]] || \
		 (echo -e "\nERROR: environment not active: run '. ./activate'" && false) )

REQ=requirements/requirements
REQS=$(REQ).txt

PY=python
PIPC=pip-compile


.PHONY: create-env
create-env:
	conda create -n tennis-face -y python=3.9
	@echo "Run: '. ./activate' to finish setup"

# install dependencies
.PHONY: install-deps
install-deps: compile-deps
	@$(TEST_ENV)
	$(PY) -m pip install -U \
		-r $(REQ).txt

# compile (lock) dependencies
.PHONY: compile-deps
compile-deps: $(REQS)

# compile (lock) dependencies
.PHONY: compile-deps-clean
compile-deps-clean:
	rm -f $(REQS)


