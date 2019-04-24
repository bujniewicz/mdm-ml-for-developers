# Machine Learning in Python for developers
### Codebase for the talk given at Mindspace Developers Meetup on 2019-04-25.

The presentation slides are available here: https://bujniewicz.github.io/mdm-ml-for-developers/

The repository consists of:

* github pages files: index.html which is the jupyter notebook converted to slides
  and mdm-ml-url.svg which is a QR code containing a link to this very repository.
* requirements.txt for dependency tracking.
* ML in Python for developers.ipynb - jupyter notebook containing the editable presentation.
* Two machine learning models and validators:
  * bayes - this is a naive bayes model of salary classification, using https://archive.ics.uci.edu/ml/datasets/adult.
    There are actually two models here, naive and regular version. They use different initial data processing.
  * knn - this is a knn model of altitude regression, using
    https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29

---

## Local setup

I was using Python 3.7.1 for running the code contained in this repository. However, I've verified it works
in 3.6.7 as well. It will not work before CPython 3.6 due to reliance on class parameter declaration order.

To work on the repo, create an environment of your preference (docker, virtualenv, venv module, etc.) and
`pip install -r requirements.txt`.

To start the jupyter notebook, run `jupyter notebook`.

To build the presentation, run `make slides`.

## Questions and comments

Feel free to contact me on https://twitter.com/bujniewicz.
