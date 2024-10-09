# Deep learning  methods for solving differential equations


## Introduction

Probability theory and statistical methods play a central role in
Science. Nowadays we are surrounded by huge amounts of data. For
example, there are more than one trillion web pages; more than one
hour of video is uploaded to YouTube every second, amounting to years
of content every day; the genomes of 1000s of people, each of which
has a length of more than a billion base pairs, have been sequenced by
various labs and so on. This deluge of data calls for automated
methods of data analysis, which is exactly what machine learning aims
at providing.

This series of lectures aim at giving you an understanding on how to
use deep learning methods like neural networks to solve differential
equations.

The online (zoom) session on October 10 will focus on the basics of
setting up a neural network and the elements you would need to build
your own code.
Thereafter, for the in-person week we will focus on
- Deep learning methods for solving ordinary differential equations
- Deep learning methods for solving partial differential equations
- Physics informed neural networks (PINNs)

You can access all learning material by going to the /doc/pub folder
and from navigate to the various sessions.  The material is provided
in forms of jupyter-notebooks and html-slides. Feel free to use the
format of your own choice.  During the lectures we will mainly use the
jupyter-notebooks. See below for instructions on how to install
various software elements.

The zoom link for the session at 10am Colombian time of October 10 is
https://us02web.zoom.us/j/88178994859?pwd=hlAM80sYPcSsFegbe5eGiEbyBCBHZT.1


## Prerequisites and background

Basic knowledge in programming and mathematics, with an emphasis on
linear algebra. Knowledge of Python or/and C++ as programming
languages is strongly recommended and experience with Jupyter
notebooks is recommended.


## Required Technologies

Course participants are expected to have their own laptops/PCs. We use _Git_ as version control software and the usage of providers like _GitHub_, _GitLab_ or similar are strongly recommended. If you are not familiar with Git as version control software, the following video may be of interest, see https://www.youtube.com/watch?v=RGOj5yH7evk&ab_channel=freeCodeCamp.org

We will make extensive use of Python as programming language and its
myriad of available libraries.  You will find
Jupyter notebooks invaluable in your work.

If you have Python installed and you feel
pretty familiar with installing different packages, we recommend that
you install the following Python packages via _pip_ as 

* pip install numpy scipy matplotlib ipython scikit-learn mglearn sympy pandas pillow 

For OSX users we recommend, after having installed Xcode, to
install _brew_. Brew allows for a seamless installation of additional
software via for example 

* brew install python


For Linux users, with its variety of distributions like for example the widely popular Ubuntu distribution,
you can use _pip_ as well and simply install Python as 

* sudo apt-get install python

You can specify the python version you wish to install. 

For various dependencies, we recommend installing a light variant of conda.

### Python installers

If you don't want to perform these operations separately and venture
into the hassle of exploring how to set up dependencies and paths, we
recommend two widely used distrubutions which set up all relevant
dependencies for Python, namely 

* Anaconda:https://docs.anaconda.com/, 

which is an open source
distribution of the Python and R programming languages for large-scale
data processing, predictive analytics, and scientific computing, that
aims to simplify package management and deployment. Package versions
are managed by the package management system _conda_. 

* Enthought canopy:https://www.enthought.com/product/canopy/ 

is a Python
distribution for scientific and analytic computing distribution and
analysis environment, available for free and under a commercial
license.

Furthermore, Google's Colab:https://colab.research.google.com/notebooks/welcome.ipynb is a free Jupyter notebook environment that requires 
no setup and runs entirely in the cloud. Try it out!

### Useful Python libraries
Here we list several useful Python libraries we strongly recommend (if you use anaconda many of these are already there)

* _NumPy_:https://www.numpy.org/ is a highly popular library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
* _The pandas_:https://pandas.pydata.org/ library provides high-performance, easy-to-use data structures and data analysis tools 
* _Xarray_:http://xarray.pydata.org/en/stable/ is a Python package that makes working with labelled multi-dimensional arrays simple, efficient, and fun!
* _Scipy_:https://www.scipy.org/ (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering. 
* _Matplotlib_:https://matplotlib.org/ is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.
* _Autograd_:https://github.com/HIPS/autograd can automatically differentiate native Python and Numpy code. It can handle a large subset of Python's features, including loops, ifs, recursion and closures, and it can even take derivatives of derivatives of derivatives
* _JAX_" https://jax.readthedocs.io/en/latest/index.html has now more or less replaced _Autograd_.
  JAX is Autograd and XLA, brought together for high-performance numerical computing and machine learning research.
  It provides composable transformations of Python+NumPy programs: differentiate, vectorize, parallelize, Just-In-Time compile to GPU/TPU, and more.
* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics.

* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics. 
* _scikit-learn_:https://scikit-learn.org/stable/ has simple and efficient tools for machine learning, data mining and data analysis
* _TensorFlow_:https://www.tensorflow.org/ is a Python library for fast numerical computing created and released by Google
* _Keras_:https://keras.io/ is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* And many more such as _pytorch_:https://pytorch.org/,  _Theano_:https://pypi.org/project/Theano/ etc 

## Textbooks

_Recommended textbooks_:


In addition to the electure notes, we recommend the books of
Goodfellow et al. and Raschka et al. We will follow these texts
closely.

- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. The different chapters are available for free at https://www.deeplearningbook.org/. Chapters 2-14 are highly recommended. The lectures follow to a good extent their chapters 6-8..
- Sebastian Raschka, Yuxi Lie, and Vahid Mirjalili,  Machine Learning with PyTorch and Scikit-Learn at https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312, see also https://sebastianraschka.com/blog/2022/ml-pytorch-book.html

The text by Raschka et al. is well-adapted to these lectures and contains many coding examples. We recommend reading chapter 11.

_Additional textbooks_:
- Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, https://www.springer.com/gp/book/9780387310732.  You can download for free the textbook in PDF format at https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
- Kevin Murphy, Probabilistic Machine Learning, an Introduction, https://probml.github.io/pml-book/book1.html
- Trevor Hastie, Robert Tibshirani, Jerome H. Friedman, The Elements of Statistical Learning, Springer, https://www.springer.com/gp/book/9780387848570. This is a well-known text and serves as additional literature.
- Aurelien Geron, Hands‑On Machine Learning with Scikit‑Learn and TensorFlow, O'Reilly, https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/. This text is very useful since it contains many code examples and hands-on applications of all algorithms discussed in this course.
- David Foster, Generative Deep Learning, https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/
- Babcock and Gavras, Generative AI with Python and TensorFlow, https://github.com/PacktPublishing/Hands-On-Generative-AI-with-Python-and-TensorFlow-2

_General learning book on statistical analysis_:
- Christian Robert and George Casella, Monte Carlo Statistical Methods, Springer
- Peter Hoff, A first course in Bayesian statistical models, Springer



