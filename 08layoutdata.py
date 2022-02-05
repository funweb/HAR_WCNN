import numpy as np
import re
from datetime import datetime
from tqdm import tqdm
import os
import time
from collections import Counter
from keras.preprocessing import sequence

from tools import general

offset = 20
max_lenght = 2000

cookActivities = {"cairo": {"Other": offset,
                            "Work": offset + 1,
                            "Take_medicine": offset + 2,
                            "Sleep": offset + 3,
                            "Leave_Home": offset + 4,
                            "Eat": offset + 5,
                            "Bed_to_toilet": offset + 6,
                            "Bathing": offset + 7,
                            "Enter_home": offset + 8,
                            "Personal_hygiene": offset + 9,
                            "Relax": offset + 10,
                            "Cook": offset + 11},
                  "kyoto7": {"Other": offset,
                             "Work": offset + 1,
                             "Sleep": offset + 2,
                             "Relax": offset + 3,
                             "Personal_hygiene": offset + 4,
                             "Cook": offset + 5,
                             "Bed_to_toilet": offset + 6,
                             "Bathing": offset + 7,
                             "Eat": offset + 8,
                             "Take_medicine": offset + 9,
                             "Enter_home": offset + 10,
                             "Leave_home": offset + 11},
                  "kyoto8": {"Other": offset,
                             "Bathing": offset + 1,
                             "Cook": offset + 2,
                             "Sleep": offset + 3,
                             "Work": offset + 4,
                             "Bed_to_toilet": offset + 5,
                             "Personal_hygiene": offset + 6,
                             "Relax": offset + 7,
                             "Eat": offset + 8,
                             "Take_medicine": offset + 9,
                             "Enter_home": offset + 10,
                             "Leave_home": offset + 11}
    ,
                  "kyoto11": {"Other": offset,
                              "Work": offset + 1,
                              "Sleep": offset + 2,
                              "Relax": offset + 3,
                              "Personal_hygiene": offset + 4,
                              "Leave_Home": offset + 5,
                              "Enter_home": offset + 6,
                              "Eat": offset + 7,
                              "Cook": offset + 8,
                              "Bed_to_toilet": offset + 9,
                              "Bathing": offset + 10,
                              "Take_medicine": offset + 11},
                  "milan": {"Other": offset,
                            "Work": offset + 1,
                            "Take_medicine": offset + 2,
                            "Sleep": offset + 3,
                            "Relax": offset + 4,
                            "Leave_Home": offset + 5,
                            "Eat": offset + 6,
                            "Cook": offset + 7,
                            "Bed_to_toilet": offset + 8,
                            "Bathing": offset + 9,
                            "Enter_home": offset + 10,
                            "Personal_hygiene": offset + 11},
                  }
mappingActivities = {"cairo": {"": "Other",
                               "other": "Other",
                               "R1_wake": "Other",
                               "R2_wake": "Other",
                               "Night_wandering": "Other",
                               "R1_work_in_office": "Work",
                               "Laundry": "Work",
                               "R2_take_medicine": "Take_medicine",
                               "R1_sleep": "Sleep",
                               "R2_sleep": "Sleep",
                               "Leave_home": "Leave_Home",
                               "Breakfast": "Eat",
                               "Dinner": "Eat",
                               "Lunch": "Eat",
                               "Bed_to_toilet": "Bed_to_toilet"},
                     "kyoto7": {"R1_Bed_to_Toilet": "Bed_to_toilet",
                                "R2_Bed_to_Toilet": "Bed_to_toilet",
                                "Meal_Preparation": "Cook",
                                "R1_Personal_Hygiene": "Personal_hygiene",
                                "R2_Personal_Hygiene": "Personal_hygiene",
                                "Watch_TV": "Relax",
                                "R1_Sleep": "Sleep",
                                "R2_Sleep": "Sleep",
                                "Clean": "Work",
                                "R1_Work": "Work",
                                "R2_Work": "Work",
                                "Study": "Other",
                                "Wash_Bathtub": "Other",
                                "other": "Other"},
                     "kyoto8": {"R1_shower": "Bathing",
                                "R2_shower": "Bathing",
                                "Bed_toilet_transition": "Other",
                                "Cooking": "Cook",
                                "R1_sleep": "Sleep",
                                "R2_sleep": "Sleep",
                                "Cleaning": "Work",
                                "R1_work": "Work",
                                "R2_work": "Work",
                                "other": "Other",
                                "Grooming": "Other",
                                "R1_wakeup": "Other",
                                "R2_wakeup": "Other"},
                     "kyoto11": {"other": "Other",
                                 "R1_Wandering_in_room": "Other",
                                 "R2_Wandering_in_room": "Other",
                                 "R1_Work": "Work",
                                 "R2_Work": "Work",
                                 "R1_Housekeeping": "Work",
                                 "R1_Sleeping_Not_in_Bed": "Sleep",
                                 "R2_Sleeping_Not_in_Bed": "Sleep",
                                 "R1_Sleep": "Sleep",
                                 "R2_Sleep": "Sleep",
                                 "R1_Watch_TV": "Relax",
                                 "R2_Watch_TV": "Relax",
                                 "R1_Personal_Hygiene": "Personal_hygiene",
                                 "R2_Personal_Hygiene": "Personal_hygiene",
                                 "R1_Leave_Home": "Leave_Home",
                                 "R2_Leave_Home": "Leave_Home",
                                 "R1_Enter_Home": "Enter_home",
                                 "R2_Enter_Home": "Enter_home",
                                 "R1_Eating": "Eat",
                                 "R2_Eating": "Eat",
                                 "R1_Meal_Preparation": "Cook",
                                 "R2_Meal_Preparation": "Cook",
                                 "R1_Bed_Toilet_Transition": "Bed_to_toilet",
                                 "R2_Bed_Toilet_Transition": "Bed_to_toilet",
                                 "R1_Bathing": "Bathing",
                                 "R2_Bathing": "Bathing"},
                     "milan": {"": "Other",
                               "other": "Other",
                               "Master_Bedroom_Activity": "Other",
                               "Meditate": "Other",
                               "Chores": "Work",
                               "Desk_Activity": "Work",
                               "Morning_Meds": "Take_medicine",
                               "Eve_Meds": "Take_medicine",
                               "Sleep": "Sleep",
                               "Read": "Relax",
                               "Watch_TV": "Relax",
                               "Leave_Home": "Leave_Home",
                               "Dining_Rm_Activity": "Eat",
                               "Kitchen_Activity": "Cook",
                               "Bed_to_Toilet": "Bed_to_toilet",
                               "Master_Bathroom": "Bathing",
                               "Guest_Bathroom": "Bathing"},
                     }


# Input: Dictionary of X Series
# Return: dataset x, y, Dict_ Activities # because the order is not reversed, the activities are arranged in order. The order needs to be disrupted in the later stage
def load_data_from_dictX(dict_ids):
    # 1 Dictionary of tectonic activities
    dict_activities = {}
    index_activity = 0
    for activity_name in dict_ids:
        dict_activities.update({activity_name: index_activity})
        index_activity += 1

    # 2 only the sensor is encoded in the source data, and now the activity is also encoded. Then return the data
    dataX = []
    dataY = []
    for activity_name in dict_ids:
        for list_sensors in dict_ids[activity_name]:
            dataX.append(list_sensors)
            dataY.append(dict_activities[activity_name])

    return dataX, dataY, dict_activities


# Reclassify data
def convertActivities(X, Y, dictActivities, uniActivities, cookActivities):
    Yf = Y.copy()
    Xf = X.copy()
    activities = {}
    for i, y in enumerate(Y):
        convertact = [key for key, value in dictActivities.items() if value == y][0]
        activity = uniActivities[convertact]
        Yf[i] = int(cookActivities[activity] - offset)
        activities[activity] = Yf[i]

    return Xf, Yf, activities


if __name__ == '__main__':
    opts = general.load_config()

    for i in range(7):
        distant_int = i
        if i == 0:
            distant_int = 999
        elif i == 6:
            distant_int = 9999

        data_dir = os.path.join(opts["datasets"]["base_dir"], 'ende')
        data_names = ['cairo', 'kyoto7', 'kyoto8', 'kyoto11', 'milan']
        data_names = opts["datasets"]["names"]
        for data_name in data_names:
            print('\n\ndataset: %s' % (data_name))
            print('data address: %s' % (os.path.join(data_dir, data_name, str(distant_int), data_name + '-dict_ids.npy')))
            dict_ids = np.load(os.path.join(data_dir, data_name, str(distant_int), data_name + '-dict_ids.npy'), allow_pickle=True).item()
            dataX, dataY, dict_activities = load_data_from_dictX(dict_ids)

            print(dict_activities)

            dataX, dataY, dict_activities = convertActivities(dataX, dataY,
                                                              dict_activities,
                                                              mappingActivities[data_name],
                                                              cookActivities[data_name])

            print(sorted(dict_activities, key=dict_activities.get))
            print("n° instances post-filtering:\t" + str(len(dataX)))

            print(Counter(dataY))

            X = np.array(dataX, dtype=object)
            Y = np.array(dataY, dtype=object)

            X = sequence.pad_sequences(X, maxlen=max_lenght, dtype='int32', padding='pre')

            save_dir = os.path.join(data_dir, data_name, str(distant_int), 'npy')
            if not os.path.exists(save_dir):
                print('Create directory: %s' % (save_dir))
                time.sleep(3)
                os.makedirs(save_dir)
            np.save(os.path.join(save_dir, data_name + '-x.npy'), X)
            np.save(os.path.join(save_dir, data_name + '-y.npy'), Y)
            np.save(os.path.join(save_dir, data_name + '-labels.npy'), dict_activities)

    print('all success！')


# TODO: 日后完善
def getData(data_name, opts):
    ksplit = 3
    data_dir = os.path.join(os.getcwd(), 'ksplitdata', str(ksplit), data_name)

    X = np.load(os.path.join(data_dir, data_name + '-train-x-0.npy', allow_pickle=True))
    Y = np.load(os.path.join(data_dir, data_name + '-train-y-0.npy', allow_pickle=True))
    dictActivities = np.load(os.path.join(data_dir, data_name + '-labels.npy', allow_pickle=True).item())
    return X, Y, dictActivities
