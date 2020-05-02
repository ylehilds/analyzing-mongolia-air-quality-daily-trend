import secrets
import logging
import math
import os
import tables

from influxdb import DataFrameClient
import numpy as np
import matplotlib.mlab as mlab
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.mlab as mlab

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Need to ssh tunnel for this to work
# ssh -L 8086:localhost:8086 aq.byu.edu -N
influx = DataFrameClient(
    host=secrets.HOST,
    port=secrets.PORT,
    username=secrets.USERNAME,
    password=secrets.PASSWORD,
    database=secrets.DATABASE,
)

from platform import python_version

print(python_version())

def large_query(influx, measurement, query, total=None, limit=100_000):
    if total is not None:
        total = math.ceil(total / limit)

    with tqdm(total=total) as pbar:
        offset = 0
        while True:
            new_query = query + " LIMIT {} OFFSET {}".format(limit, offset)
            data = influx.query(new_query)
            data = data[measurement]

            received = len(data)
            pbar.update(1)

            yield data

            offset += limit
            if received != limit:
                break


def load_data(filename):
    if os.path.exists(filename):
        LOGGER.info("Loading cached data...")
        return pd.read_hdf(filename)

    LOGGER.info("Downloading data...")
    result = influx.query(
        "SELECT COUNT(sequence) FROM air_quality_sensor WHERE time > '2019-10-01' AND time <= now()"
    )
    count = result["air_quality_sensor"].values[0][0]

    queries = large_query(
        influx,
        "air_quality_sensor",
        "SELECT * FROM air_quality_sensor WHERE time > '2019-10-01' AND time <= now()",
        count,
    )

    all_data = pd.concat(list(queries), sort=False)
    all_data.to_hdf(filename, "data")
    return all_data


data = load_data("aq_data.h5")
LOGGER.info("Done loading data...")


# https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
data.index = data.index.tz_convert('Asia/Ulaanbaatar')
data = data[data.pm2_5 >= 0]
data = data[data.location_name == 'Mongolia']

# temp = data[data.index < '15-02-2020'].groupby("name").get_group('AK')
# data.append(temp, ignore_index=True)
# https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas

# start clean up data mode when in office or switched from outdoor to indoor or vice versa

# in Office
ak = data[data.index < '2020-02-15'].groupby("name").get_group('AK')
co = data[data.index < '2020-02-15'].groupby("name").get_group('CO')
ky = data[data.index < '2020-02-15'].groupby("name").get_group('KY')

mb = data[data.index < '2020-02-15'].groupby("name").get_group('MB')
mb = mb[(mb.index < '2020-01-26') | (mb.index >= '2020-02-04')]

nj = data[(data.index < '2020-01-28') | (data.index >= '2020-02-04')].groupby("name").get_group('NJ')
nu = data[(data.index < '2020-01-26') | (data.index >= '2020-02-04')].groupby("name").get_group('NU')
oregon = data[(data.index < '2020-01-26') | (data.index >= '2020-02-04')].groupby("name").get_group('OR')
pe = data[(data.index < '2020-02-11')].groupby("name").get_group('PE') #outdoor sensor wea re no longer using these sensors data
wy = data[(data.index < '2020-02-11')].groupby("name").get_group('WY')

# Switched

# ab will commented out until we know the date
# ab = data[(data.index < '????-??-??')].groupby("name").get_group('AB') # outdoor sensor we are no longer using these sensors data before but will use after the switch to indoor
ns = data[(data.index >= '2020-01-28')].groupby("name").get_group('NS') # outdoor sensor we are no longer using these sensors data before but will use after the switch to indoor
oregon = oregon[(oregon.index >= '2020-01-28')] # outdoor sensor we are no longer using these sensors data before but will use after the switch to indoor
ut = data[(data.index >= '2020-01-29')].groupby("name").get_group('UT') # outdoor sensor we are no longer using these sensors data before but will use after the switch to indoor

# finish clean up data mode when in office or switched from outdoor to indoor or vice versa


# modified_gers = ['AL', 'AR', 'AZ', 'CA', 'CT', 'DE', 'IA', 'ID', 'IL', 'KS', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MS', 'MT', 'NC', 'NH', 'NM', 'GA', 'ND', 'NE']
# modified_gers_data = data[(data.name == modified_gers[0]) | (data.name == modified_gers[1]) | (data.name == modified_gers[2]) | (data.name == modified_gers[3]) | (data.name == modified_gers[4]) | (data.name == modified_gers[5]) | (data.name == modified_gers[6]) | (data.name == modified_gers[7]) | (data.name == modified_gers[8]) | (data.name == modified_gers[9]) | (data.name == modified_gers[10]) | (data.name == modified_gers[11]) | (data.name == modified_gers[12]) | (data.name == modified_gers[13]) | (data.name == modified_gers[14]) | (data.name == modified_gers[15]) | (data.name == modified_gers[16]) | (data.name == modified_gers[17]) | (data.name == modified_gers[18]) | (data.name == modified_gers[19]) | (data.name == modified_gers[20]) | (data.name == modified_gers[21]) | (data.name == modified_gers[22]) | (data.name == modified_gers[23])]

modified_gers = ['AL', 'AR', 'AZ', 'CA', 'CT', 'DE', 'IA', 'ID', 'IL', 'KS', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MS', 'MT', 'NC', 'NH', 'NM', 'GA', 'ND', 'NE']
modified_gers_data = data[(data.name == modified_gers[0]) | (data.name == modified_gers[1]) | (data.name == modified_gers[2]) | (data.name == modified_gers[3]) | (data.name == modified_gers[4]) | (data.name == modified_gers[5]) | (data.name == modified_gers[6]) | (data.name == modified_gers[7]) | (data.name == modified_gers[8]) | (data.name == modified_gers[9]) | (data.name == modified_gers[10]) | (data.name == modified_gers[11]) | (data.name == modified_gers[12]) | (data.name == modified_gers[13]) | (data.name == modified_gers[14]) | (data.name == modified_gers[15]) | (data.name == modified_gers[16]) | (data.name == modified_gers[17]) | (data.name == modified_gers[18]) | (data.name == modified_gers[19]) | (data.name == modified_gers[20]) | (data.name == modified_gers[21]) | (data.name == modified_gers[22]) | (data.name == modified_gers[23])]
modified_gers_data = modified_gers_data.append(ak, ignore_index=True)
modified_gers_data = modified_gers_data.append(co, ignore_index=True)
modified_gers_data = modified_gers_data.append(ky, ignore_index=True)

# unmodified_gers = ['NJ', 'NS', 'NU', 'OK', 'OR', 'PA', 'RI', 'SD', 'UT', 'VA', 'WI']
# unmodified_gers_data = data[(data.name == unmodified_gers[0]) | (data.name == unmodified_gers[1]) | (data.name == unmodified_gers[2]) | (data.name == unmodified_gers[3]) | (data.name == unmodified_gers[4]) | (data.name == unmodified_gers[5]) | (data.name == unmodified_gers[6]) | (data.name == unmodified_gers[7]) | (data.name == unmodified_gers[8]) | (data.name == unmodified_gers[9]) | (data.name == unmodified_gers[10])]

unmodified_gers = ['NJ', 'NU', 'OK', 'PA', 'RI', 'SD', 'VA', 'WI']
unmodified_gers_data = data[(data.name == unmodified_gers[0]) | (data.name == unmodified_gers[1]) | (data.name == unmodified_gers[2]) | (data.name == unmodified_gers[3]) | (data.name == unmodified_gers[4]) | (data.name == unmodified_gers[5]) | (data.name == unmodified_gers[6]) | (data.name == unmodified_gers[7])]
unmodified_gers_data = unmodified_gers_data.append(mb, ignore_index=True)
# unmodified_gers_data.append(ab, ignore_index=True)
unmodified_gers_data = unmodified_gers_data.append(ns, ignore_index=True)
unmodified_gers_data = unmodified_gers_data.append(oregon, ignore_index=True)
unmodified_gers_data = unmodified_gers_data.append(ut, ignore_index=True)


days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# for testing ...
for day in days_of_week:
    plt.xlabel('Sensors for {0}'.format(day))
    plt.ylabel('PM 2.5 Value')
    plt.title('Week PM 2.5 for sensors for {0}'.format(day))
    plt.grid(True)
    sensors_data = list()
    sensors_name = list()
    temp = data[data.index.day_name() == day]
    # temp = temp.groupby("name")
    for name, sensor_data in temp.groupby("name"):
        sensors_name.append(name)
        sensors_data.append(temp.groupby("name").get_group(name)["pm2_5"].between_time('01:01', '07:59').to_numpy().tolist())

    results = plt.boxplot(sensors_data, showfliers=False, labels=sensors_name, showmeans=True, meanline=True)
    print('whiskers: ', [item.get_ydata()[1] for item in results['whiskers']])
    print('caps: ', [item.get_ydata()[1] for item in results['caps']])
    print('boxes: ', [item.get_ydata()[1] for item in results['boxes']])
    print('medians: ', [item.get_ydata()[1] for item in results['medians']])
    print('means: ', [item.get_ydata()[1] for item in results['means']])
    print('fliers: ', [item.get_ydata()[1] for item in results['fliers']])
    plt.show()

monday = data[data.index.day_name() == 'Monday']['pm2_5']
tuesday = data[data.index.day_name() == 'Tuesday']['pm2_5']
wednesday = data[data.index.day_name() == 'Wednesday']['pm2_5']
thursday = data[data.index.day_name() == 'Thursday']['pm2_5']
friday = data[data.index.day_name() == 'Friday']['pm2_5']
saturday = data[data.index.day_name() == 'Saturday']['pm2_5']
sunday = data[data.index.day_name() == 'Sunday']['pm2_5']
all_days = [monday, tuesday, wednesday, thursday, friday, saturday, sunday]

print(data)

# https://pythonspot.com/matplotlib-histogram/

num_bins = 5
n, bins, patches = plt.hist(data.voltage.values, num_bins, facecolor='blue', alpha=0.5)
plt.show()

# n, bins, patches = plt.hist(x, num_bins, density=1, facecolor='blue', alpha=0.5)
# plt.show()

# example data
# mu = 100 # mean of distribution
# sigma = 15 # standard deviation of distribution
# # x = mu + sigma * np.random.randn(10000)
#
# num_bins = 20
# # the histogram of the data
# n, bins, patches = plt.hist(data.voltage.values, num_bins, normed=1, facecolor='blue', alpha=0.5)
#
# # add a 'best fit' line
# y = mlab.normpdf(bins, mu, sigma)
# plt.plot(bins, y, 'r--')
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
#
# # Tweak spacing to prevent clipping of ylabel
# plt.subplots_adjust(left=0.15)
# plt.show()
