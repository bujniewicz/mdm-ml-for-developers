slides:
	rm index.html
	jupyter nbconvert ML\ in\ Python\ for\ developers.ipynb --to slides --output index
	mv index.slides.html index.html
