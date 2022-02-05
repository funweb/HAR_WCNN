from sklearn.preprocessing import OneHotEncoder


enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)

print(enc.categories_)

print(enc.transform([['Female', 1], ['Male', 4]]).toarray())


print(enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]))


print(enc.get_feature_names(['gender', 'group']))










