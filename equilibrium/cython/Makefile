.PHONY: all
all:
	python setup.py build_ext --inplace

.PHONY: clean
clean:
	find . -type f -name  '*.pyc' -print0 | xargs -0 rm
	find . -type f -name  '*.c'   -print0 | xargs -0 rm
	rm -rf build/
