# WHITESPACE #

# blocks are delimited with idents
for i in [1, 2, 3, 4, 5]:
	print i
	for j in [1, 2, 3, 4, 5]:
		print j
		print i + j
	print i
print "done looping"


# white space ignored inside () and []
long_winded = (1 + 2 + 3 + 4 + 5
			   6 + 7 + 8 + 9 + 10)

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

easier_to_read = [ [1, 2, 3],
				   [4, 5, 6],
				   [7, 8, 9] ]

# can use \ to continue on next line
two_plus_three = 2 + \
				 3

# whitespace formatting can make pasting into the interpreter tricky
# IPython has a %paste command which fixes these issues

# MODULES #

# import a module; use module as prefix for accessing functions
import re 
my_regex = re.compile("[0-9]+", re.I)

# using an alias
import re as regex
import matplotlib.pyplot as plt 

# if you need only a few specific things
from collections import defaultdict, Counter
lookup = defaultdict(int)
my_counter = Counter()

# it is possible to import the entire contents of a module into your namespace:
# in general this is a TERRIBLE idea, as you can overwrite your own variabes unknowingly
match = 10
from re import *
print match 		# "<function re.match>"

# ARITHMETIC #

# integer addition by default, which can be replaced with decimal by
from __future__ import division
5 / 2 == 2.5
5 // 2 == 2 # can still do old division with //

# FUNCTIONS #

def double(x):
	"""optional docstring that explains function
	e.g. this function multiplies input by 2"""
	return x * 2

# python functions are first class, so they can be passed as variables for functional programming
def apply_to_one(f):
	"""calls the function f with 1 as arg"""
	return f(1)

# short anonymous functions (aka lambdas):
y = apply_to_one(lambda x: x + 4)

# you can assign lambdas to functions, but really you should just use def instead

# function parameters can be given default arguments, which only need to be specified when you want a value other than the default
def my_print(message="my default message"):
	print message

# it is sometimes useful to specify arguments by name:
def subtract(a=0, b=0):
	return a * b

subtract(0, 5) == subtract(b=5)

# STRINGS #

# can use '' or "", so long as they match

# use \ to encode special characters
tab_string = '\t'
len(tab_string) == 1

# if you want to use \ as \ (for ex. in regular expressions), you can create a raw string
not_tab_string = r"\t"
len(not_tab_string) == 2

# can create multiline strings with """example"""
multiline_string = """first line.
second line.
third line."""

# EXCEPTIONS #

# when something goes wrong, python raises an exception, which can cause your program to crash.
# you can handle them using try and except:
try:
	print 0/0
except ZeroDivisionError:
	print "cannot divide by zero"

# LISTS #

integer_list = [1, 2, 3]
heterogeneous_list = ["string", 0.1, True]
list_of_lists = [integer_list, heterogeneous_list, []]

len(integer_list) = 3
sum(integer_list) = 6

# accessing nth element:
x = range(10)	# [0, 1, ..., 9]
zero = x[0]
one = x[1]
nine = x[-1]
eight = [x-2]
x[0] = -1 		# [-1, 1, ..., 9]

# slicing (returns new list)
first_three = x[:3]
three_to_end = x[3:]
one_to_four = x[1:5]
last_three = x[-3:]
without_first_and_last = x[1:-1]
copy_of_x = x[:]

# in operator checks for membership (in O(n) time)
1 in [1, 2, 3] # True
0 in [1, 2, 3] # False

# concatenate
x = [1, 2, 3]
x.extend([4, 5, 6])

# returns a new list w/o modifying original
x = [1, 2, 3]
y = x + [4, 5, 6]

# more frequent: appending one item at a time
x = [1, 2, 3]
x.append(0)

# you can unpack lists if you know how many items they contain
x, y = [1, 2]

# you'll get a ValueError if the number of elements don't match, but there is a convention to use an underscore for values you intend to throw away
_, y = [1, 2]

# TUPLES #

# lists' immutable cousins. pretty much anything you can do to a list w/o modifying it applies to tuples as well
my_tuple = (1, 2)
other_tuple = 3, 4

try:
	my_tuple[1] = 3
except TypeError:
	print "cannot modify a tuple"

# tuples are convenient for returning multiple values from a function
def sum_and_product(x, y):
	return (x + y), (x * y)

sp = sum_and_product(2, 3)
s, p = sum_and_product(5, 10)

# multiple assignment
x, y = 1, 2
x, y = y, x # pythonic way to swap variables

# DICTIONARIES #

# associates values with keys
empty_dict = {}
grades = {"Joel": 80, "Tim": 95}

# look up value for key
joels_grade = grades["Joel"]

# will get KeyError if the key is not found in the dictionary

# can check for existence of a KEY with in
"Joel" in grades == True
"Kate" in grades == False

# get method returns value if key found, else a specified default, else None
grades.get("Joel", 0)  # 80
grades.get('Kate', 0)  # 0
grades.get('somebody') # None

# assigning values to keys
grades["Tim"] = 99
grades["Kate"] = 100

# dictionaries are frequently used as a simple way to structure data
tweet = {
	'user' : "greg",
	"text" : "Happy to learn data science",
	"retweet_count" : 100,
	"hashtags" : ["#data", '#science', '#datascience', '#yolo']
}

# besides looking for specific keys, we can also view all of them:
tweet.keys() # list of keys
tweet.values() # list of values
tweet.items()  # list of (key, value) tuples
tweet.iteritems() # uses lazy iteration (see generators section)

"greg" in tweet.values # True

# keys must be immutable bc dicts are implemented as hash tables under the hood. this means lists cannot be used as keys. if you need a multipart key, use a tuple or turn it into a string

# defaultdict #
# imagine you are making a dict of a document that records unique words as keys and the number of appearances of this word as the value. you can do this manually:
word_counts = {}
for word in document:
	if word in word_counts:
		word_counts[word] += 1
	else:
		word_counts[word] = 1

# a better approach is to use defaultdict. if you try to look up a key that isn't valid, it first adds the key and the value using a zero-argument function you provide when the dict is created
from collections import defaultdict

word_counts = defaultdict(int)			# int() produces 0
for word in document:
	word_counts[word] += 1

# some other functions to try:
dd_list = defaultdict(list)				# list() produces an empty list
dd_list[2].append(1)					# now dd_list contains {2: [1]}

dd_dict = defaultdict(dict)
dd_dict["Joel"]["City"] = "Seattle"		# {"Joel" : {"City" : "Seattle"}}

dd_pair = defaultdict(lambda: [0, 0])
dd_pair[2][1] = 1 						# {2: [0, 1]}

# Counter turns a sequence of values into a defaultdict(int)-like object mapping keys to counts. this can be used to make histograms.
from collections import Counter
c = Counter([0, 1, 2, 0])		# c is (basically) {0:2, 1:1, 2:1}

# this allows for a very simple solution to the lexicon problem above:
word_counts = Counter(document)

# a Counter instance has a most_common method that is frequently useful:
# print 10 most common words and their counts
for word, count in word_counts.most_common(10):
	print word, count


# SETS #

# set is a collection of distinct elements:
s = set()
s.add(1)
s.add(2)
s.add(2)	# {1, 2}
len(s)		# 2
2 in s 		# True

# in is a very fast operation on sets, making it more appropriate for some tasks than lists
words_list = ["a", "an", "at"] + thousands_of_other_words

"zip" in words_list		# False, but have to check every element in list

words_set = set(words_list)
"zip" in words_set		# fast to check

# finding the distinct items of a collection
items = [1, 2, 3, 1, 2, 3]
items_set = set(items)				# {1, 2, 3}
distinct_items = list(items_set)	# [1, 2, 3]

# CONTROL FLOW #

if 1 > 2:
	print ""
elif 1 > 3:
	print ""
else:
	print ""

# one-line ternary if-then-else
parity = "even" if x % 2 == 0 else "odd"

# while loop
x = 0
while x < 10:
	print x, "is less than 10"
	x += 1

# but more often we'll use for and in:
for x in range(10):
	print x, "is less than 10"

# more complex logic
for x in range(10):
	if x == 3:
		continue 	# skip to next iteration
	if x == 5:
		break 		# break out of loop
	print x
# above prints 0, 1, 2, 4

# TRUTHINESS #

1 < 2 # True

# None indicates a nonexistant value, similar to null in other languages
x = None
x == None # True, but not Pythonic
x is None # True and Pythonic

# "Falsy" values:
False
None
[]
{}
""
set()
0
0.0
# pretty much everything else is Truthy, so you can use if statements to test for empty lists, etc.

# and returns the second value when the first is truthy and the first when it is not:
s = function_returns_string()
first_char = s and s[0]

safe_x = x or 0 # guaranteed to return an actual number - not None

# all function takes a list and returns True when all elements are truthy
# any function returns True when any element is Truthy
all([True, 1, "a"])		# True
any([True, 0, ""])		# True


# SORTING #

# list sort method sorts in place
# sorted function returns new list
x = [4, 1, 2, 3]
y = sorted(x)
x.sort()

# can sort from largest to smallest with reverse parameter
# can compare results of function instead of the values themselves with key parameter

# sort by absolute value from largest to smallest
x = sorted([-4, 1, -2, 3], key=abs, reverse=True)	# [-4, 3, -2, 1]

# sort the words and counts from highest to lowest
wc = sorted(word_counts.items(),
			key=lambda (word, count): count,
			reverse=True)


# LIST COMPREHENSIONS #

# transform a list into another list
even_numbers = [x for x in range(5) if x % 2 == 0]		# [0, 2, 4]
squares		 = [x * x for x in range(5)]				# [0, 1, 4, 9, 16]
even_squares = [x * x for x in even_numbers]			# [0, 4, 16]

# creating dicts and sets
square_dict = { x : x * x for x in range(5) }	# {0:0, 1:1, 2:4, 3:9, 4:16}
square_set  = { x * x for x in [1, -1] }		# {1}

# if you don't need the values from the list, can use _
zeroes = [0 for _ in range(5)]

# multiple fors:
pairs = [(x, y)
		 for x in range(10)
		 for y in range(10)]	# 100 pairs (0,0),(0,1), ... ,(9,8),(9,9)

# later fors can use the results of earlier ones:
increasing_pairs = [(x, y)
					for x in range(10)
					for y in range(x + 1, 10)]

# GENERATORS AND ITERATORS #

# it is silly to create a giant list of values that takes up a tremendous amount of space when we are only intrested in one value at a time - say, for looping. A generator prodices values lazily (only when needed) for purposes of iteration.

for i in xrange(10):
	do_something(i)

# using for comprehensions wrapped in parens:
lazy_evens_below_20 = (i for i in xrange(20) if i % 2 == 0)

# you can only iterate over a generator once. if you need to iterate over something multiple times, you need to recreate the generator each time or use a list

# iteritems method for dictionaries returns key value pairs lazily


# RANDOMNESS #

import random

# random.random() produces numbers uniformly between 0 and 1
four_uniform_randoms = [random.random() for _ in range(4)]

# set the seed to get reproducable results
random.seed(10) == random.seed(10)

# random.randrange takes 1 or 2 arguments and returns an element from the chosen range():
random.randrange(10)	# chosen from range(10)
random.randrange(3, 6)	# chosen from range(3, 6)

# randomly reorder the elements of a list:
random.shuffle(range(10))	# ex: [2, 5, 1, ...]

# pick one element randomly from a list
random.choice(["Alice", "Bob", "Charlie"])

# choose sample with replacement
four_samples = [random.choice(range(10)) for _ in range(4)]

# randomly choose sample of elements without replacement
random.sample(range(10), 2)		# ex: [3, 8]


# REGULAR EXPRESSIONS #

# invaluable for searching text, but quite complicated

import re

print all([									# all of these are True:
	not re.match("a", "cat"),				# 'cat' doesn't start with 'a'
	re.search("a", "cat"),					# 'cat' has an 'a' in it
	not re.search("c", "dog"),
	3 == len(re.split("[ab]", "carbs")),	# split on 'a' or 'b' to ['c', 'r', 's']
	"R-D-" == re.sub("[0-9]", "-", "R2D2")  # replace digits with dashes 
	])


# OBJECT ORIENTED PROGRAMMING #

# let's construct our own Set class

class Set:
	
	# member functions take a first parameter "self" that refers to the particular Set object being used

	def __init__(self, values=None):
		"""constructor called when new set created, ex:
		s1 = Set()
		s2 = Set([1, 2, 3])"""

		self.dict = {}	# each instance of Set gets its own dict property

		if values is not None:
			for value in values:
				self.add(value)

	def __repr__(self):
		"""string representation of a Set object - if you type at Python promt or pass to str()"""
		return "Set: " + str(self.dict.keys())

	# membership in the Set is represented as a key in self.dict with value of True
	def add(self, value):
		self.dict[value] = True

	# value is in the Set if it's a key in the dictionary
	def contains(self, value):
		return value in self.dict

	def remove(self, value):
		del self.dict[value]

# usage
s = Set([1, 2, 3])
s.add(4)
s.contains(4)	# True
s.remove(3)
s.contains(3)	# False


# FUNCTIONAL TOOLS #

# sometimes we want to partially apply functions to create new functions:
def exp(base, power):
	return base ** power

def two_to_the(power):
	return exp(2, power)

# the above method can quickly become unwieldy, so there is another tool that allows us to do the equivolent
from functools import partial
two_to_the = partial(exp, 2)

# partial does get messy if you attempt to curry arguments in the middle of the function, so avoid doing that

# map, reduce, and filter are functional alternatives to list comprehensions
def double(x):
	return 2 * x

xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs]		# [2, 4, 6, 8]
twice_xs = map(double, xs)				# same as above
list_doubler = partial(map, double)		# *function* that doubles a list
twice_xs = list_doubler(xs)				# again [2, 4, 6, 8]

# map can be used with multiple-argument functions if you provide multiple lists:
def multiply(x, y): return x * y

products = map(multiply, [1, 2], [4,5])		# [1*4,2*5] = [4,10]

# filter does the work of a list comprehension "if":
def is_even(x):
	"""True if x is even, False if x is odd"""
	return x % 2 == 0

x_evens = [x for x in xs if is_even(x)]		# [2, 4]
x_evens = filter(is_even, xs)				# same as above
list_evener = partial(filter, is_even)		# *function* that filters a list
x_evens = list_evener(xs)					# again [2, 4]

# reduce combines the first two elements of a list, then that result with the third and so on:
x_product = reduce(multiply, xs)			# 1*2*3*4 = 24
list_product = partial(reduce, multiply)	# *function* that reduces a list
x_product = list_product(xs)				# again =24


# ENUMERATE #

# on occasion we want to iterate over a list and use both its elements and their indexes:

# NOT Pythonic:
for i in range(len(documents)):
	document = documents[i]
	do_something(i, document)

# the Pythonic solution is enumerate, which produces tuples (index,element):
for i, document in enumerate(documents):
	do something(i, document)

# if we just want the indexes:
for i in range(len(documents)): do_something(i)		# NOT Pythonic
for i, _ in enumerate(documents): do_something(i)	# Pythonic


# ZIP AND ARGUMENT UNPACKING #

# sometimes we need to zip two or more lists together. zip transforms multiple lists into a single list of tuples:
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
zip(list1,list2)		# [('a',1),('b',2),('c',3)]

# if the lists are different lengths, zip stops as soon as the first list ends

# can also unzip a list:
pairs = [('a',1),('b',2),('c',3)]
letters, numbers = zip(*pairs)

# the asterisk performs argument unpacking, which uses elements of pairs as individual arguments to zip.
# you can actually use argument unpacking with any function:
def add(a, b): return a + b

add(1, 2)		# 3
add([1, 2])		# TypeError
add(*[1,2])		# 3


# ARGS AND KWARGS #

# what if we want to specify a function that takes an arbitrary number of arguments? we use argument unpacking
def magic(*args, **kwargs):
	print "unnamed args:", args
	print "keyword args:", kwargs

magic(1, 2, key="word", key2="word2")

# above prints:
# unnamed args: (1, 2)
# keyword args: {'key2': 'word2', 'key': 'word'}

# can also be used the other way around to supply lists or dicts as arguments to a function:
def other_way_magic(x, y, z):
	return x + y + z

x_y_list = [1, 2]
z_dict = {"z" : 3}
print other_way_magic(*x_y_list, **z_dict)	# 6

# our most common usage for this stange witchcraft will be producing higher order functions whose inputs can accept arbitrary arguments:
def doubler_correct(f):
	"""works no matter what kind of inputs f expects"""
	def g(*args, **kwargs):
		"""whatever arguments g is supplied, pass them through to f"""
		return 2 * f(*args, **kwargs)
	return g

def f2(x, y):
	return x + y

g = doubler_correct(f2)
g(1, 2)		# 6





























