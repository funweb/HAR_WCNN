# This file is a program for viewing the table of sensor contribution in the paper
# The generated files are CSV files located in: datasets\TFIDF\csvfiles\
import numpy as np
import os
import pandas as pd
from tools import general


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

opts = general.load_config()

data_dir = os.path.join(opts["datasets"]["base_dir"], 'tfidf')
data_names = ['cairo', 'kyoto7', 'kyoto8', 'kyoto11', 'milan']
data_names = opts["datasets"]["names"]

csv_dir = os.path.join(data_dir, 'csvFiles')

for data_name in data_names:
    df_data = pd.read_json(os.path.join(data_dir, data_name + '-norm'), encoding='utf-8')
    # display(df_data)

    df_rename = df_data.rename(columns=mappingActivities[data_name])

    df_copy = pd.DataFrame()
    for activity in list(set(df_rename.columns)):
        print(activity)
        df_copy[activity] = pd.DataFrame(df_rename[activity]).mean(axis=1)
    # display(df_copy)

    map_columns = {
        'Other': 0,
        'Personal_hygiene': 1,
        'Take_medicine': 2,
        'Sleep': 3,
        'Bed_to_toilet': 4,
        'Work': 5,
        'Leave_Home': 6,
        'Relax': 7,
        'Enter_home': 8,
        'Cook': 9,
        'Eat': 10,
        'Bathing': 11
    }
    df_final = df_copy.rename(columns=map_columns)
    df_final.sort_index(axis=1, inplace=True)
    for i in range(12):
        if i not in df_final.columns:
            df_final.insert(i, i, np.float16(0))

    df_final.drop(df_final.index[df_final.sum(axis=1) == 0], axis=0, inplace=True)
    # display(df_final)
    print(df_final)

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    csv_file_name = os.path.join(csv_dir, data_name + '_tfidf.csv')
    df_final.to_csv(csv_file_name, encoding='utf-8', float_format='%.2f')
    print('save in: %s' % csv_file_name)
