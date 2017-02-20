import tooploox as tp
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

input_file = 'data.csv'

# ---------- Exercise 1 ----------
video_views = tp.read_csv_data(input_file)

print "Basic statistics for:"
print "n = 24"
print "Mean: {:.2f}, Standard deviation: {:.2f}".format(video_views[:, 23].mean(), video_views[:, 23].std())
print "Q1: {:.2f}, Q2: {:.2f}, Q3: {:.2f}\n".format(np.percentile(video_views[:, 23], 25),
                                                    np.percentile(video_views[:, 23], 50),
                                                    np.percentile(video_views[:, 23], 75))

print "Basic statistics for:"
print "n = 72"
print "Mean: {:.2f}, Standard deviation: {:.2f}".format(video_views[:, 71].mean(), video_views[:, 71].std())
print "Q1: {:.2f}, Q2: {:.2f}, Q3: {:.2f}\n".format(np.percentile(video_views[:, 71], 25),
                                                    np.percentile(video_views[:, 71], 50),
                                                    np.percentile(video_views[:, 71], 75))

print "Basic statistics for:"
print "n = 168"
print "Mean: {:.2f}, Standard deviation: {:.2f}".format(video_views[:, 167].mean(), video_views[:, 167].std())
print "Q1: {:.2f}, Q2: {:.2f}, Q3: {:.2f}\n".format(np.percentile(video_views[:, 167], 25),
                                                    np.percentile(video_views[:, 167], 50),
                                                    np.percentile(video_views[:, 167], 75))

# ---------- Exercise 2 ----------
tp.plot_histogram_view_vs_videos(video_views[:, 167],
                                 "Plot of number of video views distribution after one week v(168)")

# ---------- Exercise 3 ----------
tp.plot_histogram_view_vs_videos(np.log(video_views[:, 167]),
                                 "Plot of number of video views distribution after one week v(168)")

# ---------- Exercise 4 ----------
data = np.log(video_views[:, 167])
data_mean = np.mean(data)
data_std = np.std(data)
data_idx = ~((data < data_mean - 3 * data_std) | (data > data_mean + 3 * data_std))
video_views = video_views[data_idx, :]
tp.plot_histogram_view_vs_videos(np.log(video_views[:, 167]),
                                 "Plot of number of video views distribution after one week without outliers v(168)")

# ---------- Exercise 5 ----------
print 'Correlations coefficients for v(1):v(24) and v(168)'
data_corr = np.concatenate((np.log(video_views[:, :24]), np.log(video_views[:, 167:])), 1)
data_corr[data_corr < 0] = 0
corr_matrix = np.corrcoef(data_corr, rowvar=0)
print corr_matrix

# ---------- Exercise 6 ----------
log_video_views = np.log(video_views)
log_video_views[log_video_views < 0] = 0
rand_sequence = np.random.permutation(log_video_views.shape[0])
test_log_video_views = log_video_views[rand_sequence[:int(0.1 * log_video_views.shape[0])], :]
train_log_video_views = log_video_views[rand_sequence[int(0.1 * log_video_views.shape[0]):], :]

# ---------- Exercise 7 & 8 & 9 & 10 ----------

test_log_video_views_X = test_log_video_views[:, :167]
test_log_video_views_Y = test_log_video_views[:, 167:]

train_log_video_views_X = train_log_video_views[:, :167]
train_log_video_views_Y = train_log_video_views[:, 167:]

regr_single_input = linear_model.LinearRegression()
regr_multiple_input = linear_model.LinearRegression()

mrse_single_input = []
mrse_multiple_input = []

for n in xrange(24):
    regr_single_input.fit(train_log_video_views_X[:, n, None], train_log_video_views_Y)
    regr_multiple_input.fit(train_log_video_views_X[:, :n + 1], train_log_video_views_Y)

    mrse_single_input.append(tp.compute_mrse(regr_single_input,
                                             test_log_video_views_X[:, n, None],
                                             test_log_video_views_Y))

    mrse_multiple_input.append(tp.compute_mrse(regr_multiple_input,
                                               test_log_video_views_X[:, :n + 1],
                                               test_log_video_views_Y))


plt.plot(mrse_single_input, 'r+-')
plt.plot(mrse_multiple_input, 'b.-')
plt.grid()
plt.xlabel("Reference time (n)")
plt.ylabel("mRSE")
plt.legend(('Linear Regression', 'Multiple-input Linear Regression'), loc='upper right')
plt.show()
