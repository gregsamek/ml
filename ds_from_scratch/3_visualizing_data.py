# the matplotlib library is tried and true, though showing its age. if you want elaborate interactive visualizations, it's probably not the right choice, but it's still fine for simple charts

# matplotlib.pyplot builds maintains an internal state in which you build up a vis step by step. once done, you can save it with savefig() or display with show()

from matplotlib import pyplot as plt 
from collections import Counter

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# create a line chart, years on x axis, gdp on y 
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# add a title
plt.title('Nominal GDP')

# add a label to the y-axis
plt.ylabel("Billions of $")
plt.show()


# BAR CHARTS #

# show some quanity varies among a discrete set of items
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# bars are by default width 0.8, so we'll add 0.1 to the left coordinates so that each bar is centered
xs = [i + 0.1 for i, _ in enumerate(movies)]

# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.bar(xs, num_oscars)

plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")

# label x-axis with movie names at bar centers
plt.xticks([i + 0.5 for i, _ in enumerate(movies)], movies)
plt.show()

# bar charts are also good for plotting histograms in order to explore how data is distributed:
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade: grade // 10 * 10
histogram = Counter(decile(grade) for grade in grades)

plt.bar([x - 4 for x in histogram.keys()], # shift each bar left by 4
	    histogram.values(),				   # give each bar its correct height
	    8)								   # give each bar a width of 8

plt.axis([-5, 105, 0, 5])				   # x-axis from -5 to 105, y-axis 0 to 5

plt.xticks([10 * i for i in range(11)])    # x-axis labels at 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
plt.show()

# another bar chart (with misleading axis)
mentions = [500, 505]
years = [2013, 2014]

plt.bar([2012.6, 2013.6], mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say 'data science'")

# if you don't do this, matplotlib will label the x-axis 0, 1 and then add a +2.013e3 in the corner (for the lolz)
plt.ticklabel_format(useOffset=False)

# misleading axis
plt.axis([2012.5, 2014.5, 499, 506])
plt.title("Look at the Huge Increase!")
plt.show()

# the less impressive (real) axes:
# plt.axis([2012.5, 2014.5, 0, 550])
# plt.title("Not So Huge Anymore")
# plt.show()


# LINE CHARTS #

# line charts are good for showing trends:
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256] 
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# we can make multiple calls to plt.plot to show multiple series on the same chart
plt.plot(xs, variance,     'g-',  label='variance')		# green solid line
plt.plot(xs, bias_squared, 'r-.', label='bias^2')		# red dot-dashed line
plt.plot(xs, total_error,  'b:',  label='total error')	# blue dotted line

# because we've assigned labels to each series, we can get a legend
# loc=9 means "top center"
plt.legend(loc=9)
plt.xlabel('model complexity')
plt.title('The Bias-Variance Tradeoff')
plt.show()


# SCATTERPLOTS #

# used for visualizing relationship between two paired sets of data:
friends = [ 70,  65,  72,  63,  71,  64,  60,  64,  67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# label each point
for label, friend_count, minute_count in zip(labels, friends, minutes):
	plt.annotate(label,
		xy=(friend_count, minute_count),	# put the label with its point
		xytext=(5,5),						# but slightly offset
		textcoords='offset points')

plt.title('Daily Minutes vs. Number of Friends')
plt.xlabel('# of friends')
plt.ylabel('daily minutes spent on site')
plt.show()

# if you're scattering comparable values, you might get a misleading picture if you let matplotlib choose the scale:
test_1_grades = [ 99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]
plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
# plt.axis("equal")
plt.show()

# the above graph incorrectly makes it seem like test 1 is more variable
# we can include a call to plt.axis("equal") to more accurately show the data
# now its clear that test 2 is morevariable